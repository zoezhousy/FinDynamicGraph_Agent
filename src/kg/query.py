from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from neo4j import GraphDatabase, basic_auth


class KGQueryClient:
    """Read-only query helper for agents."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def get_ticker_subgraph(
        self,
        ticker: str,
        as_of_date: datetime,
        max_news: int = 20,
        max_signals: int = 50,
        signal_window_days: int = 90,
        news_window_days: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        signal_from = as_of_date - timedelta(days=signal_window_days)
        news_from = as_of_date - timedelta(days=news_window_days)

        cypher = """
        MATCH (c:Company {ticker: $ticker})

        OPTIONAL MATCH (c)-[r1:HAS_SIGNAL]->(s:IndicatorSignal)
        WHERE datetime(r1.as_of_date) <= datetime($as_of)
        AND datetime(r1.as_of_date) >= datetime($signal_from)

        OPTIONAL MATCH (c)-[rf:HAS_SIGNAL]->(f:FundamentalSignal)
        WHERE datetime(rf.as_of_date) <= datetime($as_of)

        OPTIONAL MATCH (c)-[r2:MENTIONED_IN]->(n:NewsEvent)
        WHERE datetime(r2.as_of_date) <= datetime($as_of)
        AND datetime(r2.as_of_date) >= datetime($news_from)

        OPTIONAL MATCH (n)-[:SUPPORTED_BY]->(ne:Evidence)
        OPTIONAL MATCH (f)-[:SUPPORTED_BY]->(fe:Evidence)

        OPTIONAL MATCH (src_news:SourceDocument)-[:CONTAINS_EVIDENCE]->(ne)
        OPTIONAL MATCH (src_fund:SourceDocument)-[:CONTAINS_EVIDENCE]->(fe)

        OPTIONAL MATCH (ne)-[:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(nc:Claim)
        OPTIONAL MATCH (fe)-[:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(fc:Claim)

        WITH c,
            [x IN collect(DISTINCT s) WHERE x IS NOT NULL] AS raw_signals,
            [x IN collect(DISTINCT f) WHERE x IS NOT NULL] AS raw_fundamentals,
            [x IN collect(DISTINCT n) WHERE x IS NOT NULL] AS raw_news,
            [x IN collect(DISTINCT ne) + collect(DISTINCT fe) WHERE x IS NOT NULL] AS raw_evidences,
            [x IN collect(DISTINCT src_news) + collect(DISTINCT src_fund) WHERE x IS NOT NULL] AS raw_sources,
            [x IN collect(DISTINCT nc) + collect(DISTINCT fc) WHERE x IS NOT NULL] AS raw_claims

        RETURN c,
            raw_signals[0..$max_signals] AS signals,
            raw_fundamentals AS fundamentals,
            raw_news[0..$max_news] AS news,
            raw_evidences AS evidences,
            raw_sources AS sources,
            raw_claims AS claims
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(
                cypher,
                ticker=ticker,
                as_of=as_of_date.isoformat(),
                signal_from=signal_from.isoformat(),
                news_from=news_from.isoformat(),
                max_news=max_news,
                max_signals=max_signals,
            ).single()

            if not rec:
                return {
                    "company": [],
                    "signals": [],
                    "fundamentals": [],
                    "news": [],
                    "evidences": [],
                    "sources": [],
                    "claims": [],
                }

            company = rec["c"]
            signals = rec["signals"] or []
            fundamentals = rec["fundamentals"] or []
            news = rec["news"] or []
            evidences = rec["evidences"] or []
            sources = rec["sources"] or []
            claims = rec["claims"] or []

            signal_rows = [dict(node) for node in signals if node]
            fundamental_rows = [dict(node) for node in fundamentals if node]
            news_rows = [dict(node) for node in news if node]
            evidence_rows = [dict(node) for node in evidences if node]
            source_rows = [dict(node) for node in sources if node]
            claim_rows = [dict(node) for node in claims if node]

            def signal_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("as_of_date") or "")

            def news_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("published_at") or "")

            def evidence_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("published_at") or "")

            def source_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("published_at") or "")

            def claim_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("as_of_date") or "")

            signal_rows = sorted(signal_rows, key=signal_sort_key, reverse=True)[:max_signals]
            fundamental_rows = sorted(fundamental_rows, key=signal_sort_key, reverse=True)
            news_rows = sorted(news_rows, key=news_sort_key, reverse=True)[:max_news]
            evidence_rows = sorted(evidence_rows, key=evidence_sort_key, reverse=True)
            source_rows = sorted(source_rows, key=source_sort_key, reverse=True)
            claim_rows = sorted(claim_rows, key=claim_sort_key, reverse=True)

            return {
                "company": [dict(company)] if company else [],
                "signals": signal_rows,
                "fundamentals": fundamental_rows,
                "news": news_rows,
                "evidences": evidence_rows,
                "sources": source_rows,
                "claims": claim_rows,
            }

    def get_decision_trace(self, decision_id: str) -> dict[str, Any] | None:
        cypher = """
        MATCH (d:DecisionTrace {entity_id: $decision_id})
        OPTIONAL MATCH (d)-[:HAS_ASSESSMENT]->(aa:AgentAssessment)
        OPTIONAL MATCH (aa)-[:USES_EVIDENCE]->(ae)
        OPTIONAL MATCH (d)-[:USES_EVIDENCE]->(de)
        OPTIONAL MATCH (src:SourceDocument)-[:CONTAINS_EVIDENCE]->(de)
        OPTIONAL MATCH (de)-[:SUPPORTS_CLAIM|CONTRADICTS_CLAIM]->(cl:Claim)
        OPTIONAL MATCH (d)-[:HAS_OUTCOME]->(bo:BacktestOutcome)
        OPTIONAL MATCH (d)-[:FOR_COMPANY]->(c:Company)

        RETURN d, c,
            collect(DISTINCT aa) AS assessments,
            collect(DISTINCT ae) AS assessment_evidence,
            collect(DISTINCT de) AS decision_evidence,
            collect(DISTINCT src) AS sources,
            collect(DISTINCT cl) AS claims,
            collect(DISTINCT bo) AS outcomes
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(cypher, decision_id=decision_id).single()

            if not rec:
                return None

            return {
                "decision": dict(rec["d"]) if rec["d"] else None,
                "company": dict(rec["c"]) if rec["c"] else None,
                "assessments": [dict(x) for x in (rec["assessments"] or []) if x],
                "assessment_evidence": [dict(x) for x in (rec["assessment_evidence"] or []) if x],
                "decision_evidence": [dict(x) for x in (rec["decision_evidence"] or []) if x],
                "sources": [dict(x) for x in (rec["sources"] or []) if x],
                "claims": [dict(x) for x in (rec["claims"] or []) if x],
            }
    
    def get_backtest_outcome(self, outcome_id: str) -> dict[str, Any] | None:
        cypher = """
        MATCH (b:BacktestOutcome {entity_id: $outcome_id})
        OPTIONAL MATCH (d:DecisionTrace)-[:HAS_OUTCOME]->(b)
        RETURN b, d
        """

        with self._driver.session(database=self._database) as session:
            rec = session.run(cypher, outcome_id=outcome_id).single()

            if not rec:
                return None

            return {
                "outcome": dict(rec["b"]) if rec["b"] else None,
                "decision": dict(rec["d"]) if rec["d"] else None,
            }
