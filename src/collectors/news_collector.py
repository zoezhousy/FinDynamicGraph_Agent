from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from tavily import TavilyClient

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry


class NewsCollector:
    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing TAVILY_API_KEY in environment.")
        self.client = TavilyClient(api_key=api_key)

    def fetch_news(self, ticker: str, limit: int | None = None) -> pd.DataFrame:
        max_results = limit or self.config.news_limit_per_ticker
        query = f"{ticker} Hong Kong stock company news"

        @retry(
            max_retries=self.config.max_retries,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
            backoff_multiplier=self.config.backoff_multiplier,
            max_backoff_seconds=self.config.max_backoff_seconds,
        )
        def _search() -> Dict[str, Any]:
            self.rate_limiter.wait()
            return self.client.search(
                query=query,
                topic="news",
                max_results=max_results,
                search_depth="advanced",
            )

        resp = _search()
        rows = self._normalize(resp.get("results", []), ticker)
        frame = pd.DataFrame(rows)
        frame = frame.sort_values("published_time", ascending=False).reset_index(drop=True)
        logging.info("Fetched news for %s with %d rows", ticker, len(frame))
        return frame

    @staticmethod
    def _normalize(results: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        collected_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for item in results:
            rows.append(
                {
                    "ticker": ticker,
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                    "published_time": item.get("published_date"),
                    "content": item.get("content"),
                    "score": item.get("score"),
                    "collected_at": collected_at,
                }
            )

        return rows

