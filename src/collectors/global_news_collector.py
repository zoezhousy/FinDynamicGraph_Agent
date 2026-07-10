from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry


# Default macro search queries for Hong Kong market context.
# Tuned to cover: HK monetary policy, China economy, global risk events,
# sector-relevant macro themes, and the Hang Seng index itself.
DEFAULT_MACRO_QUERIES = [
    "Hang Seng index",
    "Hong Kong stock market",
    "China economy",
    "People's Bank of China interest rate",
    "US Federal Reserve rate decision",
    "Hong Kong monetary authority",
    "China GDP growth",
    "global inflation CPI",
    "US China trade tariffs",
    "Hong Kong property market",
    "China tech regulation",
    "global recession risk",
    "oil price OPEC",
    "US dollar Hong Kong dollar",
    "China stimulus package",
]


class GlobalNewsCollector:
    """Collect global/macro news using yfinance Search API.

    No API key required. Uses ``yf.Search`` to query multiple macro topics
    and deduplicates by title.  Returns a DataFrame compatible with the
    existing news pipeline.
    """

    def __init__(
        self,
        config: CollectionConfig,
        queries: List[str] | None = None,
        articles_per_query: int = 10,
    ) -> None:
        self.config = config
        self.queries = queries or DEFAULT_MACRO_QUERIES
        self.articles_per_query = articles_per_query
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

    def fetch_global_news(
        self,
        ticker: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch global macro news relevant to HK market.

        Args:
            ticker: Optional ticker for context (used in the ``ticker`` column).
                    If None, defaults to "GLOBAL".
            limit:  Max articles to return after dedup.

        Returns:
            DataFrame with columns matching ``NewsCollector`` output.
        """
        ticker = ticker or "GLOBAL"

        @retry(
            max_retries=self.config.max_retries,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
            backoff_multiplier=self.config.backoff_multiplier,
            max_backoff_seconds=self.config.max_backoff_seconds,
        )
        def _search_all() -> List[Dict[str, Any]]:
            seen_titles: set[str] = set()
            all_articles: List[Dict[str, Any]] = []

            for query in self.queries:
                self.rate_limiter.wait()
                try:
                    search = yf.Search(
                        query=query,
                        news_count=self.articles_per_query,
                        enable_fuzzy_query=True,
                    )
                except Exception as exc:
                    logging.warning("yf.Search failed for query=%r: %s", query, exc)
                    # Fast-fail on rate limit — remaining queries will also fail
                    if "429" in str(exc) or "Rate limited" in str(exc):
                        logging.warning("Rate limited by Yahoo, skipping remaining global news queries")
                        break
                    continue

                for article in search.news or []:
                    title = self._extract_title(article)
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    all_articles.append(article)

                if len(all_articles) >= limit:
                    break

            return all_articles

        raw = _search_all()
        rows = self._normalize(raw, ticker)
        frame = pd.DataFrame(rows)

        if frame.empty:
            logging.warning("Global news: no articles found for ticker=%s", ticker)
            return frame

        frame = frame.sort_values("published_time", ascending=False).reset_index(drop=True)
        frame = frame.head(limit)
        logging.info("Fetched %d global news articles for ticker=%s", len(frame), ticker)
        return frame

    # ── helpers ──

    @staticmethod
    def _extract_title(article: dict) -> str:
        if "content" in article:
            return (article["content"] or {}).get("title", "")
        return article.get("title", "")

    @staticmethod
    def _normalize(articles: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        collected_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for article in articles:
            if "content" in article:
                content = article["content"]
                title = content.get("title", "")
                summary = content.get("summary", "")
                provider = content.get("provider", {})
                source = provider.get("displayName", "Unknown")
                url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
                url = url_obj.get("url", "")
                pub_date_str = content.get("pubDate", "")
            else:
                title = article.get("title", "")
                summary = article.get("summary", "")
                source = article.get("publisher", "Unknown")
                url = article.get("link", "")
                # yfinance Search returns providerPublishTime as unix timestamp
                pub_ts = article.get("providerPublishTime")
                if pub_ts and isinstance(pub_ts, (int, float)) and pub_ts > 0:
                    pub_date_str = datetime.fromtimestamp(pub_ts, tz=timezone.utc).isoformat()
                else:
                    pub_date_str = ""

            rows.append(
                {
                    "ticker": ticker,
                    "title": title,
                    "url": url,
                    "source": source,
                    "published_time": pub_date_str,
                    "content": summary,
                    "score": None,
                    "collected_at": collected_at,
                    "news_type": "global_macro",
                }
            )

        return rows
