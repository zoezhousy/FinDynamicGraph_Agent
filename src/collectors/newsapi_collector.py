"""NewsAPI.org collector for Hong Kong stock news.

Uses the ``/v2/everything`` endpoint with keyword search.
Supports both English and Chinese articles.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry

_BASE_URL = "https://newsapi.org/v2/everything"

# Common HK stock name mappings for better search results.
_TICKER_ALIASES: Dict[str, str] = {
    "0005.HK": "HSBC",
    "0700.HK": "Tencent",
    "1299.HK": "AIA",
    "9988.HK": "Alibaba",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
    "2318.HK": "Ping An Insurance",
    "0941.HK": "China Mobile",
    "0883.HK": "CNOOC",
    "1398.HK": "ICBC",
    "0388.HK": "HKEX",
    "2020.HK": "ANTA Sports",
    "9618.HK": "JD.com",
    "9999.HK": "NetEase",
    "1024.HK": "Kuaishou",
}

# Chinese aliases for bilingual search.
_TICKER_ALIASES_ZH: Dict[str, str] = {
    "0005.HK": "汇丰",
    "0700.HK": "腾讯",
    "1299.HK": "友邦保险",
    "9988.HK": "阿里巴巴",
    "3690.HK": "美团",
    "1810.HK": "小米",
    "2318.HK": "平安保险",
    "0941.HK": "中国移动",
    "0883.HK": "中海油",
    "1398.HK": "工商银行",
    "0388.HK": "港交所",
    "2020.HK": "安踏体育",
    "9618.HK": "京东",
    "9999.HK": "网易",
    "1024.HK": "快手",
}


class NewsAPICollector:
    """Fetch Hong Kong stock news from NewsAPI.org."""

    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

        api_key = os.getenv("NEWSAPI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing NEWSAPI_API_KEY in environment.")
        self.api_key = api_key

    def fetch_news(
        self,
        ticker: str,
        limit: int | None = None,
        days_back: int = 7,
        languages: str | None = None,
    ) -> pd.DataFrame:
        """Fetch news for a given HK stock ticker.

        Args:
            ticker: Stock ticker, e.g. "0700.HK".
            limit:  Max articles to return. Defaults to config.news_limit_per_ticker.
            days_back: How many days back to search. Default 7.
            languages: Comma-separated language codes, e.g. "en,zh".
                       Defaults to "en,zh" for bilingual coverage.

        Returns:
            DataFrame with columns matching the existing news pipeline.
        """
        max_results = limit or self.config.news_limit_per_ticker
        languages = languages or "en,zh"

        # Build search query: use alias if available, otherwise ticker.
        alias = _TICKER_ALIASES.get(ticker, ticker.replace(".HK", ""))
        # Use company name + "Hong Kong" for better relevance.
        query = f"{alias} Hong Kong"

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        @retry(
            max_retries=self.config.max_retries,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
            backoff_multiplier=self.config.backoff_multiplier,
            max_backoff_seconds=self.config.max_backoff_seconds,
        )
        def _search() -> Dict[str, Any]:
            self.rate_limiter.wait()
            params: Dict[str, Any] = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": min(max_results * 2, 100),  # fetch extra for dedup
                "apiKey": self.api_key,
            }
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data = _search()
        articles = data.get("articles", [])
        rows = self._normalize(articles, ticker)

        # If English-only returned few results, do a Chinese pass.
        if len(rows) < max_results and "zh" in languages:
            rows_zh = self._fetch_chinese(ticker, alias, from_date, max_results - len(rows))
            seen_urls = {r["url"] for r in rows}
            rows.extend(r for r in rows_zh if r["url"] not in seen_urls)

        frame = pd.DataFrame(rows)
        if frame.empty:
            logging.warning("NewsAPI: no articles found for %s", ticker)
            return frame
        if "published_time" in frame.columns:
            frame = frame.sort_values("published_time", ascending=False).reset_index(drop=True)
        frame = frame.head(max_results)
        logging.info("NewsAPI: fetched %d articles for %s", len(frame), ticker)
        return frame

    def _fetch_chinese(
        self, ticker: str, alias: str, from_date: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Second pass targeting Chinese-language articles."""
        zh_alias = _TICKER_ALIASES_ZH.get(ticker, alias)
        query = zh_alias

        @retry(
            max_retries=2,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=8.0,
        )
        def _search_zh() -> Dict[str, Any]:
            self.rate_limiter.wait()
            params: Dict[str, Any] = {
                "q": query,
                "from": from_date,
                "language": "zh",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": self.api_key,
            }
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        try:
            data = _search_zh()
            return self._normalize(data.get("articles", []), ticker)
        except Exception as exc:
            logging.warning("NewsAPI Chinese pass failed for %s: %s", ticker, exc)
            return []

    @staticmethod
    def _normalize(articles: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        collected_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for item in articles:
            source_obj = item.get("source") or {}
            rows.append(
                {
                    "ticker": ticker,
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": source_obj.get("name", "Unknown"),
                    "published_time": item.get("publishedAt"),
                    "content": item.get("description") or item.get("content"),
                    "score": None,
                    "collected_at": collected_at,
                    "news_type": "newsapi",
                }
            )

        return rows
"""NewsAPI.org collector for Hong Kong stock news.

Uses the ``/v2/everything`` endpoint with keyword search.
Supports both English and Chinese articles.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry

_BASE_URL = "https://newsapi.org/v2/everything"

# Common HK stock name mappings for better search results.
_TICKER_ALIASES: Dict[str, str] = {
    "0005.HK": "HSBC",
    "0700.HK": "Tencent",
    "1299.HK": "AIA",
    "9988.HK": "Alibaba",
    "3690.HK": "Meituan",
    "1810.HK": "Xiaomi",
    "2318.HK": "Ping An Insurance",
    "0941.HK": "China Mobile",
    "0883.HK": "CNOOC",
    "1398.HK": "ICBC",
    "0388.HK": "HKEX",
    "2020.HK": "ANTA Sports",
    "9618.HK": "JD.com",
    "9999.HK": "NetEase",
    "1024.HK": "Kuaishou",
}

# Chinese aliases for bilingual search.
_TICKER_ALIASES_ZH: Dict[str, str] = {
    "0005.HK": "汇丰",
    "0700.HK": "腾讯",
    "1299.HK": "友邦保险",
    "9988.HK": "阿里巴巴",
    "3690.HK": "美团",
    "1810.HK": "小米",
    "2318.HK": "平安保险",
    "0941.HK": "中国移动",
    "0883.HK": "中海油",
    "1398.HK": "工商银行",
    "0388.HK": "港交所",
    "2020.HK": "安踏体育",
    "9618.HK": "京东",
    "9999.HK": "网易",
    "1024.HK": "快手",
}


class NewsAPICollector:
    """Fetch Hong Kong stock news from NewsAPI.org."""

    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

        api_key = os.getenv("NEWSAPI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing NEWSAPI_API_KEY in environment.")
        self.api_key = api_key

    def fetch_news(
        self,
        ticker: str,
        limit: int | None = None,
        days_back: int = 7,
        languages: str | None = None,
    ) -> pd.DataFrame:
        """Fetch news for a given HK stock ticker.

        Args:
            ticker: Stock ticker, e.g. "0700.HK".
            limit:  Max articles to return. Defaults to config.news_limit_per_ticker.
            days_back: How many days back to search. Default 7.
            languages: Comma-separated language codes, e.g. "en,zh".
                       Defaults to "en,zh" for bilingual coverage.

        Returns:
            DataFrame with columns matching the existing news pipeline.
        """
        max_results = limit or self.config.news_limit_per_ticker
        languages = languages or "en,zh"

        # Build search query: use alias if available, otherwise ticker.
        alias = _TICKER_ALIASES.get(ticker, ticker.replace(".HK", ""))
        # Use company name + "Hong Kong" for better relevance.
        query = f"{alias} Hong Kong"

        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        @retry(
            max_retries=self.config.max_retries,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
            backoff_multiplier=self.config.backoff_multiplier,
            max_backoff_seconds=self.config.max_backoff_seconds,
        )
        def _search() -> Dict[str, Any]:
            self.rate_limiter.wait()
            params: Dict[str, Any] = {
                "q": query,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": min(max_results * 2, 100),  # fetch extra for dedup
                "apiKey": self.api_key,
            }
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data = _search()
        articles = data.get("articles", [])
        rows = self._normalize(articles, ticker)

        # If English-only returned few results, do a Chinese pass.
        if len(rows) < max_results and "zh" in languages:
            rows_zh = self._fetch_chinese(ticker, alias, from_date, max_results - len(rows))
            seen_urls = {r["url"] for r in rows}
            rows.extend(r for r in rows_zh if r["url"] not in seen_urls)

        frame = pd.DataFrame(rows)
        if frame.empty:
            logging.warning("NewsAPI: no articles found for %s", ticker)
            return frame
        if "published_time" in frame.columns:
            frame = frame.sort_values("published_time", ascending=False).reset_index(drop=True)
        frame = frame.head(max_results)
        logging.info("NewsAPI: fetched %d articles for %s", len(frame), ticker)
        return frame

    def _fetch_chinese(
        self, ticker: str, alias: str, from_date: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Second pass targeting Chinese-language articles."""
        zh_alias = _TICKER_ALIASES_ZH.get(ticker, alias)
        query = zh_alias

        @retry(
            max_retries=2,
            initial_backoff_seconds=1.0,
            backoff_multiplier=2.0,
            max_backoff_seconds=8.0,
        )
        def _search_zh() -> Dict[str, Any]:
            self.rate_limiter.wait()
            params: Dict[str, Any] = {
                "q": query,
                "from": from_date,
                "language": "zh",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": self.api_key,
            }
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        try:
            data = _search_zh()
            return self._normalize(data.get("articles", []), ticker)
        except Exception as exc:
            logging.warning("NewsAPI Chinese pass failed for %s: %s", ticker, exc)
            return []

    @staticmethod
    def _normalize(articles: List[Dict[str, Any]], ticker: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        collected_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for item in articles:
            source_obj = item.get("source") or {}
            rows.append(
                {
                    "ticker": ticker,
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "source": source_obj.get("name", "Unknown"),
                    "published_time": item.get("publishedAt"),
                    "content": item.get("description") or item.get("content"),
                    "score": None,
                    "collected_at": collected_at,
                    "news_type": "newsapi",
                }
            )

        return rows
