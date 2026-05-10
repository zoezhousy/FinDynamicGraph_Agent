from __future__ import annotations

from datetime import datetime
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
    ) -> dict[str, list[dict[str, Any]]]:
        cypher = """
        MATCH (c:Company {ticker: $ticker})
        OPTIONAL MATCH (c)-[r1:HAS_SIGNAL]->(s:IndicatorSignal)
          WHERE datetime(r1.as_of_date) <= datetime($as_of)
        OPTIONAL MATCH (c)-[r2:MENTIONED_IN]->(n:NewsEvent)
          WHERE datetime(r2.as_of_date) <= datetime($as_of)
        RETURN c,
               collect(DISTINCT s)[0..$max_signals] AS signals,
               collect(DISTINCT n)[0..$max_news] AS news
        """
        with self._driver.session(database=self._database) as session:
            rec = session.run(
                cypher,
                ticker=ticker,
                as_of=as_of_date.isoformat(),
                max_news=max_news,
                max_signals=max_signals,
            ).single()
            if not rec:
                return {"company": [], "signals": [], "news": []}
            company = rec["c"]
            signals = rec["signals"] or []
            news = rec["news"] or []
            return {
                "company": [dict(company)] if company else [],
                "signals": [dict(node) for node in signals if node],
                "news": [dict(node) for node in news if node],
            }
