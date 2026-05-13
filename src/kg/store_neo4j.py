from __future__ import annotations

import json
import logging
from typing import Iterable

from neo4j import GraphDatabase, basic_auth

from src.kg.schema import AgentAssessment, BacktestOutcome, DecisionTrace, Entity, Evidence, Relation


_ALLOWED_REL_TYPES = {
    "HAS_SIGNAL",
    "MENTIONED_IN",
    "SUPPORTED_BY",
    "HAS_RISK",
    "RELATES_TO",
    "FOR_COMPANY",
    "HAS_ASSESSMENT",
    "MADE_BY",
    "SUPPORTS_DECISION",
    "OPPOSES_DECISION",
    "USES_EVIDENCE",
    "CONTAINS_EVIDENCE",
    "SUPPORTS_CLAIM",
    "CONTRADICTS_CLAIM",
    "CLAIM_USED_BY",
    "HAS_OUTCOME",
}


class Neo4jKGStore:
    """Thin wrapper around Neo4j for storing the financial KG."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def init_constraints(self) -> None:
        cypher_statements = [
            "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.entity_id IS UNIQUE",
            "CREATE CONSTRAINT indicator_id IF NOT EXISTS FOR (i:IndicatorSignal) REQUIRE i.entity_id IS UNIQUE",
            "CREATE CONSTRAINT news_id IF NOT EXISTS FOR (n:NewsEvent) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT risk_id IF NOT EXISTS FOR (r:RiskEvent) REQUIRE r.entity_id IS UNIQUE",
            "CREATE CONSTRAINT source_doc_id IF NOT EXISTS FOR (s:SourceDocument) REQUIRE s.entity_id IS UNIQUE",
            "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.entity_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_entity_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.entity_id IS UNIQUE",
            "CREATE CONSTRAINT decision_trace_id IF NOT EXISTS FOR (d:DecisionTrace) REQUIRE d.entity_id IS UNIQUE",
            "CREATE CONSTRAINT assessment_id IF NOT EXISTS FOR (aa:AgentAssessment) REQUIRE aa.entity_id IS UNIQUE",
            "CREATE CONSTRAINT backtest_outcome_id IF NOT EXISTS FOR (b:BacktestOutcome) REQUIRE b.entity_id IS UNIQUE",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in cypher_statements:
                session.run(stmt)

    def clear_generated_data_for_ticker(self, ticker: str) -> None:
        queries = [
            """
            MATCH (c:Company {ticker: $ticker})-[:HAS_SIGNAL]->(s:IndicatorSignal)
            DETACH DELETE s
            """,
            """
            MATCH (c:Company {ticker: $ticker})-[:MENTIONED_IN]->(n:NewsEvent)
            DETACH DELETE n
            """,
            """
            MATCH (src:SourceDocument)
            WHERE src.entity_id STARTS WITH ('source:news:' + $ticker + ':')
            DETACH DELETE src
            """,
            """
            MATCH (cl:Claim)
            WHERE cl.entity_id STARTS WITH ('claim:news:' + $ticker + ':')
            DETACH DELETE cl
            """,
            """
            MATCH (e:Evidence)
            WHERE e.evidence_id STARTS WITH ('news:' + $ticker + ':')
            DETACH DELETE e
            """,
            """
            MATCH (c:Company {ticker: $ticker})-[:HAS_SIGNAL]->(f:FundamentalSignal)
            DETACH DELETE f
            """,
            """
            MATCH (c:Company {ticker: $ticker})-[:HAS_RISK]->(r:RiskEvent)
            DETACH DELETE r
            """,
            """
            MATCH (d:DecisionTrace {ticker: $ticker})
            DETACH DELETE d
            """,
            """
            MATCH (aa:AgentAssessment {ticker: $ticker})
            DETACH DELETE aa
            """,
            """
            MATCH (b:BacktestOutcome {ticker: $ticker})
            DETACH DELETE b
            """,
        ]
        with self._driver.session(database=self._database) as session:
            for query in queries:
                session.run(query, ticker=ticker)

    def clear_all_generated_data(self) -> None:
        queries = [
            "MATCH (s:IndicatorSignal) DETACH DELETE s",
            "MATCH (n:NewsEvent) DETACH DELETE n",
            "MATCH (src:SourceDocument) DETACH DELETE src",
            "MATCH (cl:Claim) DETACH DELETE cl",
            "MATCH (e:Evidence) DETACH DELETE e",
            "MATCH (f:FundamentalSignal) DETACH DELETE f",
            "MATCH (r:RiskEvent) DETACH DELETE r",
            "MATCH (d:DecisionTrace) DETACH DELETE d",
            "MATCH (aa:AgentAssessment) DETACH DELETE aa",
            "MATCH (b:BacktestOutcome) DETACH DELETE b",
            "MATCH (a:Agent) DETACH DELETE a",
        ]
        with self._driver.session(database=self._database) as session:
            for query in queries:
                session.run(query)

    def upsert_entities(self, entities: Iterable[Entity]) -> None:
        by_label: dict[str, list[dict]] = {}
        for entity in entities:
            props = {
                "entity_id": entity.entity_id,
                "entity_type": entity.type,
                **(entity.properties or {}),
            }
            by_label.setdefault(entity.type, []).append(props)

        with self._driver.session(database=self._database) as session:
            for label, rows in by_label.items():
                if not rows:
                    continue
                query = f"""
                UNWIND $rows AS row
                MERGE (e:{label} {{entity_id: row.entity_id}})
                SET e += row
                """
                session.run(query, rows=rows)

    def upsert_evidences(self, evidences: Iterable[Evidence]) -> None:
        rows = []
        for evidence in evidences:
            data = evidence.model_dump()
            if data["published_at"] is not None:
                data["published_at"] = data["published_at"].isoformat()

            # Important: make Evidence compatible with generic Relation matching.
            data["entity_id"] = evidence.evidence_id
            data["entity_type"] = "Evidence"

            rows.append(data)

        if not rows:
            return

        cypher = """
        UNWIND $rows AS row
        MERGE (e:Evidence {evidence_id: row.evidence_id})
        SET e += row
        """
        with self._driver.session(database=self._database) as session:
            session.run(cypher, rows=rows)

    def upsert_relations(self, relations: Iterable[Relation]) -> None:
        grouped: dict[str, list[dict]] = {}
        for relation in relations:
            rel_type = relation.type
            if rel_type not in _ALLOWED_REL_TYPES:
                raise ValueError(f"Unsupported relationship type: {rel_type}")
            data = relation.model_dump()
            for key in ("as_of_date", "valid_from", "valid_to"):
                if data.get(key) is not None:
                    data[key] = data[key].isoformat()
            grouped.setdefault(rel_type, []).append(data)

        with self._driver.session(database=self._database) as session:
            for rel_type, rows in grouped.items():
                if not rows:
                    continue
                query = f"""
                UNWIND $rows AS row
                MATCH (s {{entity_id: row.start_id}})
                MATCH (t {{entity_id: row.end_id}})
                MERGE (s)-[r:{rel_type} {{as_of_date: row.as_of_date}}]->(t)
                SET r.confidence = row.confidence,
                    r.direction = row.direction,
                    r.valid_from = row.valid_from,
                    r.valid_to = row.valid_to,
                    r.evidence_ids = row.evidence_ids
                """
                session.run(query, rows=rows)

    def upsert_decision_trace(self, trace: DecisionTrace, assessments: Iterable[AgentAssessment]) -> None:
        trace_row = trace.model_dump()
        trace_row["trade_date"] = trace.trade_date.isoformat()
        trace_row["trace_json"] = json.dumps(trace_row.pop("trace", {}), ensure_ascii=False)

        assessment_rows = []
        for assessment in assessments:
            row = assessment.model_dump()
            row["trade_date"] = assessment.trade_date.isoformat()
            row["factors_json"] = json.dumps(row.pop("factors", []), ensure_ascii=False)
            assessment_rows.append(row)

        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (d:DecisionTrace {entity_id: $entity_id})
                SET d += $props
                """,
                entity_id=trace.decision_id,
                props={"entity_id": trace.decision_id, **trace_row},
            )
            session.run(
                """
                MATCH (d:DecisionTrace {entity_id: $decision_id})
                MATCH (c:Company {ticker: $ticker})
                MERGE (d)-[:FOR_COMPANY {as_of_date: $trade_date}]->(c)
                """,
                decision_id=trace.decision_id,
                ticker=trace.ticker,
                trade_date=trace.trade_date.isoformat(),
            )

            if assessment_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (aa:AgentAssessment {entity_id: row.assessment_id})
                    SET aa += row
                    """,
                    rows=[{"entity_id": r["assessment_id"], **r} for r in assessment_rows],
                )
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (a:Agent {entity_id: 'agent:' + row.agent_role})
                    SET a.role = row.agent_role
                    WITH a, row
                    MATCH (aa:AgentAssessment {entity_id: row.assessment_id})
                    MERGE (a)-[:MADE_BY {as_of_date: row.trade_date}]->(aa)
                    """,
                    rows=assessment_rows,
                )
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (d:DecisionTrace {entity_id: $decision_id})
                    MATCH (aa:AgentAssessment {entity_id: row.assessment_id})
                    MERGE (d)-[:HAS_ASSESSMENT {as_of_date: row.trade_date}]->(aa)
                    """,
                    decision_id=trace.decision_id,
                    rows=assessment_rows,
                )
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (d:DecisionTrace {entity_id: $decision_id})
                    MATCH (aa:AgentAssessment {entity_id: row.assessment_id})
                    FOREACH (_ IN CASE WHEN row.supports_decision THEN [1] ELSE [] END |
                        MERGE (aa)-[:SUPPORTS_DECISION {as_of_date: row.trade_date}]->(d)
                    )
                    FOREACH (_ IN CASE WHEN row.opposes_decision THEN [1] ELSE [] END |
                        MERGE (aa)-[:OPPOSES_DECISION {as_of_date: row.trade_date}]->(d)
                    )
                    """,
                    decision_id=trace.decision_id,
                    rows=assessment_rows,
                )
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (aa:AgentAssessment {entity_id: row.assessment_id})
                    UNWIND coalesce(row.evidence_refs, []) AS evidence_id
                    OPTIONAL MATCH (e:Evidence {evidence_id: evidence_id})
                    OPTIONAL MATCH (s {entity_id: evidence_id})
                    FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (aa)-[:USES_EVIDENCE {as_of_date: row.trade_date}]->(e)
                    )
                    FOREACH (_ IN CASE WHEN e IS NULL AND s IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (aa)-[:USES_EVIDENCE {as_of_date: row.trade_date}]->(s)
                    )
                    """,
                    rows=assessment_rows,
                )

            session.run(
                """
                MATCH (d:DecisionTrace {entity_id: $decision_id})
                UNWIND $evidence_ids AS evidence_id
                OPTIONAL MATCH (e:Evidence {evidence_id: evidence_id})
                OPTIONAL MATCH (s {entity_id: evidence_id})
                FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (d)-[:USES_EVIDENCE {as_of_date: $trade_date}]->(e)
                )
                FOREACH (_ IN CASE WHEN e IS NULL AND s IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (d)-[:USES_EVIDENCE {as_of_date: $trade_date}]->(s)
                )
                """,
                decision_id=trace.decision_id,
                evidence_ids=trace.evidence_ids,
                trade_date=trace.trade_date.isoformat(),
            )

    def upsert_backtest_outcome(self, outcome: BacktestOutcome) -> None:
        row = outcome.model_dump()

        row["trade_date"] = outcome.trade_date.isoformat()
        row["evaluated_at"] = outcome.evaluated_at.isoformat()
        row["metadata_json"] = json.dumps(row.pop("metadata", {}), ensure_ascii=False)

        props = {
            "entity_id": outcome.outcome_id,
            "entity_type": "BacktestOutcome",
            **row,
        }

        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (b:BacktestOutcome {entity_id: $entity_id})
                SET b += $props
                """,
                entity_id=outcome.outcome_id,
                props=props,
            )

            session.run(
                """
                MATCH (d:DecisionTrace {entity_id: $decision_id})
                MATCH (b:BacktestOutcome {entity_id: $outcome_id})
                MERGE (d)-[r:HAS_OUTCOME {as_of_date: $trade_date}]->(b)
                SET r.system = $system,
                    r.raw_return = $raw_return,
                    r.trade_executed = $trade_executed,
                    r.direction_outcome = $direction_outcome
                """,
                decision_id=outcome.decision_id,
                outcome_id=outcome.outcome_id,
                trade_date=outcome.trade_date.isoformat(),
                system=outcome.system,
                raw_return=outcome.raw_return,
                trade_executed=outcome.trade_executed,
                direction_outcome=outcome.direction_outcome,
            )

    def health_check(self) -> bool:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run("RETURN 1 AS ok").single()
                return bool(result and result["ok"] == 1)
        except Exception as exc:
            logging.error("Neo4j health check failed: %s", exc)
            return False
