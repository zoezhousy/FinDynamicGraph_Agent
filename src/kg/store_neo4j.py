from __future__ import annotations

import json
import logging
from datetime import datetime
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
    "CONFLICTS_WITH",
    "SUPERSEDES",
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
        # Suppress Neo4j driver INFO notifications during constraint creation
        neo4j_logger = logging.getLogger("neo4j")
        prev_level = neo4j_logger.level
        neo4j_logger.setLevel(logging.WARNING)
        try:
            with self._driver.session(database=self._database) as session:
                for stmt in cypher_statements:
                    session.run(stmt)
        finally:
            neo4j_logger.setLevel(prev_level)

    def clear_generated_data_for_ticker(
        self,
        ticker: str,
        preserve_historical_traces: bool = True,
    ) -> None:
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
            MATCH (cl:Claim)
            WHERE cl.entity_id STARTS WITH ('claim:fundamental:' + $ticker + ':')
            DETACH DELETE cl
            """,
            """
            MATCH (e:Evidence)
            WHERE e.evidence_id STARTS WITH ('fundamental_evidence:' + $ticker + ':')
            DETACH DELETE e
            """,
            """
            MATCH (c:Company {ticker: $ticker})-[:HAS_RISK]->(r:RiskEvent)
            DETACH DELETE r
            """,
        ]

        if not preserve_historical_traces:
            queries.extend(
                [
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
            )

        with self._driver.session(database=self._database) as session:
            for query in queries:
                session.run(query, ticker=ticker)

    def clear_old_indicators(
        self,
        ticker: str,
        older_than: str,
    ) -> int:
        """
        Delete IndicatorSignal nodes for a ticker older than the given date.
        Also cleans up their relations.

n        Args:
            ticker: Stock ticker, e.g. "0700.HK"
            older_than: ISO date string, e.g. "2025-01-01". Signals before this date are deleted.

        Returns:
            Number of deleted signals.
        """
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[r:HAS_SIGNAL]->(s:IndicatorSignal)
        WHERE s.as_of_date < $cutoff
        DETACH DELETE s
        RETURN count(s) AS deleted
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, ticker=ticker, cutoff=older_than).single()
            deleted = result["deleted"] if result else 0
            logging.info("Deleted %d old IndicatorSignal for %s (before %s)", deleted, ticker, older_than)
            return deleted

    def clear_old_news(self, ticker: str, older_than: str) -> int:
        """Delete NewsEvent + related Evidence/Claim/SourceDocument for a ticker older than cutoff.

        Returns:
            Number of deleted news events.
        """
        queries = [
            """
            MATCH (c:Company {ticker: $ticker})-[r:MENTIONED_IN]->(n:NewsEvent)
            WHERE n.published_at < $cutoff
            DETACH DELETE n
            RETURN count(n) AS deleted
            """,
            """
            MATCH (src:SourceDocument)
            WHERE src.entity_id STARTS WITH ('source:news:' + $ticker + ':')
              AND src.published_at < $cutoff
            DETACH DELETE src
            """,
            """
            MATCH (cl:Claim)
            WHERE cl.entity_id STARTS WITH ('claim:news:' + $ticker + ':')
              AND cl.as_of_date < $cutoff
            DETACH DELETE cl
            """,
            """
            MATCH (e:Evidence)
            WHERE e.evidence_id STARTS WITH ('news:' + $ticker + ':')
              AND e.published_at < $cutoff
            DETACH DELETE e
            """,
        ]
        total = 0
        with self._driver.session(database=self._database) as session:
            result = session.run(queries[0], ticker=ticker, cutoff=older_than).single()
            total = result["deleted"] if result else 0
            for q in queries[1:]:
                session.run(q, ticker=ticker, cutoff=older_than)
        logging.info("Deleted %d old NewsEvent for %s (before %s)", total, ticker, older_than)
        return total

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

    # ── Temporal / Dynamic KG methods ─────────────────────────────────────

    def upsert_entities_temporal(self, entities: Iterable[Entity]) -> list[str]:
        """Upsert entities with temporal awareness.

        MERGE on entity_id. If node already exists and is_active, bump version
        and update properties.  Returns list of entity_ids that were newly created
        (not already present in graph).
        """
        by_label: dict[str, list[dict]] = {}
        for entity in entities:
            now_iso = (entity.properties.get("ingested_at") or datetime.utcnow()).isoformat()
            props = {
                "entity_id": entity.entity_id,
                "entity_type": entity.type,
                "ingested_at": now_iso,
                "is_active": True,
                **(entity.properties or {}),
            }
            # Ensure temporal fields exist
            props.setdefault("version", 1)
            props.setdefault("valid_from", props.get("as_of_date", now_iso))
            props.setdefault("valid_to", None)
            props.setdefault("supersedes_id", None)
            by_label.setdefault(entity.type, []).append(props)

        created_ids: list[str] = []
        with self._driver.session(database=self._database) as session:
            for label, rows in by_label.items():
                if not rows:
                    continue
                query = f"""
                UNWIND $rows AS row
                MERGE (e:{label} {{entity_id: row.entity_id}})
                ON CREATE SET e += row, e.version = 1
                ON MATCH SET e += row,
                    e.version = CASE WHEN e.is_active = false THEN 1 ELSE coalesce(e.version, 1) + 1 END,
                    e.ingested_at = row.ingested_at
                """
                session.run(query, rows=rows)
        return created_ids

    def upsert_evidences_temporal(self, evidences: Iterable[Evidence]) -> None:
        """Upsert evidences with temporal fields."""
        rows = []
        for evidence in evidences:
            data = evidence.model_dump()
            now_iso = (data.get("ingested_at") or datetime.utcnow()).isoformat()
            if data["published_at"] is not None:
                data["published_at"] = data["published_at"].isoformat()
            data["entity_id"] = evidence.evidence_id
            data["entity_type"] = "Evidence"
            data.setdefault("ingested_at", now_iso)
            data.setdefault("valid_from", data.get("published_at", now_iso))
            data.setdefault("valid_to", None)
            data.setdefault("supersedes_id", None)
            data.setdefault("is_active", True)
            data.setdefault("version", 1)
            rows.append(data)

        if not rows:
            return

        cypher = """
        UNWIND $rows AS row
        MERGE (e:Evidence {evidence_id: row.evidence_id})
        ON CREATE SET e += row, e.version = 1
        ON MATCH SET e += row,
            e.version = CASE WHEN e.is_active = false THEN 1 ELSE coalesce(e.version, 1) + 1 END,
            e.ingested_at = row.ingested_at
        """
        with self._driver.session(database=self._database) as session:
            session.run(cypher, rows=rows)

    def upsert_relations_temporal(self, relations: Iterable[Relation]) -> None:
        """Upsert relations with temporal fields."""
        grouped: dict[str, list[dict]] = {}
        for relation in relations:
            rel_type = relation.type
            if rel_type not in _ALLOWED_REL_TYPES:
                raise ValueError(f"Unsupported relationship type: {rel_type}")
            data = relation.model_dump()
            now_iso = (data.get("ingested_at") or datetime.utcnow()).isoformat()
            for key in ("as_of_date", "valid_from", "valid_to"):
                if data.get(key) is not None:
                    data[key] = data[key].isoformat()
            data.setdefault("ingested_at", now_iso)
            data.setdefault("supersedes_id", None)
            data.setdefault("is_active", True)
            data.setdefault("version", 1)
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
                    r.evidence_ids = row.evidence_ids,
                    r.version = coalesce(r.version, 0) + 1,
                    r.ingested_at = row.ingested_at,
                    r.is_active = true
                """
                session.run(query, rows=rows)

    def expire_signals_for_ticker(self, ticker: str, cutoff_date: str) -> int:
        """Set valid_to on signals older than cutoff_date instead of deleting.

        Returns count of expired signals.
        """
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[r:HAS_SIGNAL]->(s:IndicatorSignal)
        WHERE s.as_of_date < $cutoff AND s.is_active = true
        SET s.is_active = false, s.valid_to = $cutoff
        SET r.is_active = false, r.valid_to = $cutoff
        RETURN count(s) AS expired
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, ticker=ticker, cutoff=cutoff_date).single()
            return result["expired"] if result else 0

    def expire_evidence_for_ticker(self, ticker: str, cutoff_date: str) -> int:
        """Set valid_to on evidence older than cutoff_date instead of deleting.

        Returns count of expired evidence nodes.
        """
        cypher = """
        MATCH (e:Evidence)
        WHERE (e.evidence_id STARTS WITH $prefix_news
            OR e.evidence_id STARTS WITH $prefix_tech
            OR e.evidence_id STARTS WITH $prefix_fund
            OR e.evidence_id STARTS WITH $prefix_risk
            OR e.evidence_id STARTS WITH $prefix_global)
          AND e.published_at < $cutoff
          AND e.is_active = true
        SET e.is_active = false, e.valid_to = $cutoff
        RETURN count(e) AS expired
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(
                cypher,
                prefix_news=f"news:{ticker}:",
                prefix_tech=f"tech_evidence:{ticker}:",
                prefix_fund=f"fundamental_evidence:{ticker}:",
                prefix_risk=f"risk_evidence:{ticker}:",
                prefix_global=f"global:{ticker}:",
                cutoff=cutoff_date,
            ).single()
            return result["expired"] if result else 0

    def find_conflicting_claims(self, ticker: str) -> list[dict]:
        """Find pairs of active claims with opposite polarity for a ticker."""
        cypher = """
        MATCH (a:Claim {ticker: $ticker})-[:CONFLICTS_WITH]-(b:Claim {ticker: $ticker})
        WHERE a.is_active = true AND b.is_active = true
          AND a.entity_id < b.entity_id
        RETURN a, b
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, ticker=ticker)
            pairs = []
            for record in result:
                pairs.append({
                    "claim_a": dict(record["a"]),
                    "claim_b": dict(record["b"]),
                })
            return pairs

    def create_conflict_between_claims(
        self,
        claim_a_id: str,
        claim_b_id: str,
        as_of_date: str,
    ) -> None:
        """Create a bidirectional CONFLICTS_WITH relation between two claims."""
        cypher = """
        MATCH (a:Claim {entity_id: $a_id})
        MATCH (b:Claim {entity_id: $b_id})
        MERGE (a)-[r:CONFLICTS_WITH {as_of_date: $as_of}]->(b)
        SET r.is_active = true, r.confidence = 0.8
        MERGE (b)-[r2:CONFLICTS_WITH {as_of_date: $as_of}]->(a)
        SET r2.is_active = true, r2.confidence = 0.8
        """
        with self._driver.session(database=self._database) as session:
            session.run(
                cypher, a_id=claim_a_id, b_id=claim_b_id, as_of=as_of_date
            )

    def health_check(self) -> bool:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run("RETURN 1 AS ok").single()
                return bool(result and result["ok"] == 1)
        except Exception as exc:
            logging.error("Neo4j health check failed: %s", exc)
            return False
