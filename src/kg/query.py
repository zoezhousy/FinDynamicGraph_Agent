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

        OPTIONAL MATCH (c)-[r2:MENTIONED_IN]->(n:NewsEvent)
        WHERE datetime(r2.as_of_date) <= datetime($as_of)
          AND datetime(r2.as_of_date) >= datetime($news_from)

        WITH c,
             [x IN collect(DISTINCT s) WHERE x IS NOT NULL] AS raw_signals,
             [x IN collect(DISTINCT n) WHERE x IS NOT NULL] AS raw_news

        RETURN c,
               raw_signals[0..$max_signals] AS signals,
               raw_news[0..$max_news] AS news
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
                return {"company": [], "signals": [], "news": []}

            company = rec["c"]
            signals = rec["signals"] or []
            news = rec["news"] or []

            # 转成 Python dict
            signal_rows = [dict(node) for node in signals if node]
            news_rows = [dict(node) for node in news if node]

            # Python 端按时间排序，防止 collect 后顺序不稳定
            def signal_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("as_of_date") or "")

            def news_sort_key(x: dict[str, Any]) -> str:
                return str(x.get("published_at") or "")

            signal_rows = sorted(signal_rows, key=signal_sort_key, reverse=True)[:max_signals]
            news_rows = sorted(news_rows, key=news_sort_key, reverse=True)[:max_news]

            return {
                "company": [dict(company)] if company else [],
                "signals": signal_rows,
                "news": news_rows,
            }
