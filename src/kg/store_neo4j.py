from __future__ import annotations

import logging
from typing import Iterable

from neo4j import GraphDatabase, basic_auth

from src.kg.schema import Entity, Evidence, Relation


_ALLOWED_REL_TYPES = {
    "HAS_SIGNAL",
    "MENTIONED_IN",
    "SUPPORTED_BY",
    "HAS_RISK",
    "RELATES_TO",
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
            "CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in cypher_statements:
                session.run(stmt)

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

    def health_check(self) -> bool:
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run("RETURN 1 AS ok").single()
                return bool(result and result["ok"] == 1)
        except Exception as exc:
            logging.error("Neo4j health check failed: %s", exc)
            return False
