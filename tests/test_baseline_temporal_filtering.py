"""Tests for point-in-time news filtering in baselines._load_raw_evidence.

Each test creates a temporary parquet file with controlled timestamps and
verifies that the filtering logic in _filter_news_by_trade_date and
_load_raw_evidence behaves correctly.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.eval.baselines import _filter_news_by_trade_date, _load_raw_evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_news_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal news DataFrame."""
    return pd.DataFrame(rows)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_ohlcv_df(n: int = 30, start: str = "2025-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with sequential trading days."""
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": [100.0] * n,
        "high": [105.0] * n,
        "low": [95.0] * n,
        "close": [102.0] * n,
        "volume": [1_000_000] * n,
    })


# ---------------------------------------------------------------------------
# Tests for _filter_news_by_trade_date
# ---------------------------------------------------------------------------


class TestFilterNewsByTradeDate:
    """Unit tests for the news filtering helper."""

    def test_future_news_excluded(self):
        """News published after trade_date must be excluded."""
        df = _make_news_df([
            {"title": "Past", "published_time": "2025-05-01T10:00:00Z", "source": "A"},
            {"title": "Future", "published_time": "2025-07-01T10:00:00Z", "source": "B"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Past"

    def test_historical_news_kept(self):
        """News published before trade_date must be kept."""
        df = _make_news_df([
            {"title": "Old", "published_time": "2025-01-15T08:00:00Z", "source": "A"},
            {"title": "Recent", "published_time": "2025-05-20T12:00:00Z", "source": "B"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 2

    def test_same_day_news_allowed(self):
        """News published on the same day as trade_date must be included."""
        df = _make_news_df([
            {"title": "Same day AM", "published_time": "2025-06-01T08:00:00Z", "source": "A"},
            {"title": "Same day PM", "published_time": "2025-06-01T20:00:00Z", "source": "B"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 2

    def test_invalid_time_excluded(self):
        """Rows with unparseable timestamps must be excluded."""
        df = _make_news_df([
            {"title": "Bad", "published_time": "not-a-date", "source": "A"},
            {"title": "Good", "published_time": "2025-05-15T10:00:00Z", "source": "B"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Good"

    def test_empty_time_excluded(self):
        """Rows with empty/NaT timestamps must be excluded."""
        df = _make_news_df([
            {"title": "Empty", "published_time": None, "source": "A"},
            {"title": "NaT", "published_time": pd.NaT, "source": "B"},
            {"title": "Good", "published_time": "2025-05-15T10:00:00Z", "source": "C"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "Good"

    def test_no_usable_news_returns_empty(self):
        """If all news are in the future or have no time, return empty DataFrame."""
        df = _make_news_df([
            {"title": "Future", "published_time": "2025-08-01T10:00:00Z", "source": "A"},
            {"title": "Bad time", "published_time": "garbage", "source": "B"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert result.empty

    def test_timezone_aware_no_crash(self):
        """tz-aware and tz-naive timestamps must not cause comparison errors."""
        df = _make_news_df([
            {"title": "TZ-aware", "published_time": "2025-05-20T10:00:00+08:00", "source": "A"},
            {"title": "UTC", "published_time": "2025-05-20T10:00:00Z", "source": "B"},
            {"title": "Naive", "published_time": "2025-05-20 10:00:00", "source": "C"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 3

    def test_sorted_descending(self):
        """Returned news must be sorted by published_time descending."""
        df = _make_news_df([
            {"title": "Oldest", "published_time": "2025-01-01T00:00:00Z", "source": "A"},
            {"title": "Middle", "published_time": "2025-03-01T00:00:00Z", "source": "B"},
            {"title": "Newest", "published_time": "2025-05-30T00:00:00Z", "source": "C"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        titles = result["title"].tolist()
        assert titles == ["Newest", "Middle", "Oldest"]

    def test_max_items_limit(self):
        """At most max_items rows must be returned."""
        rows = [
            {"title": f"N{i}", "published_time": f"2025-05-{i+1:02d}T00:00:00Z", "source": "S"}
            for i in range(20)
        ]
        df = _make_news_df(rows)
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade, max_items=10)
        assert len(result) == 10

    def test_published_at_column_used(self):
        """If published_time is absent but published_at exists, use published_at."""
        df = _make_news_df([
            {"title": "A", "published_at": "2025-05-15T10:00:00Z", "source": "S"},
            {"title": "B", "published_at": "2025-08-01T10:00:00Z", "source": "S"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "A"

    def test_date_column_used_as_fallback(self):
        """If only 'date' column exists, use it."""
        df = _make_news_df([
            {"title": "A", "date": "2025-05-15", "source": "S"},
            {"title": "B", "date": "2025-08-01", "source": "S"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert len(result) == 1
        assert result.iloc[0]["title"] == "A"

    def test_no_time_column_returns_empty(self):
        """If no time column exists at all, return empty DataFrame."""
        df = _make_news_df([
            {"title": "A", "source": "S"},
        ])
        trade = datetime(2025, 6, 1)
        result = _filter_news_by_trade_date(df, trade)
        assert result.empty


# ---------------------------------------------------------------------------
# Tests for _load_raw_evidence (integration with OHLCV)
# ---------------------------------------------------------------------------


class TestLoadRawEvidence:
    """Integration tests for _load_raw_evidence using temp directories."""

    def test_ohlcv_not_affected_by_news_filter(self, tmp_path):
        """OHLCV loading must work independently of news filtering."""
        ticker = "TEST.HK"
        ticker_dir = tmp_path / ticker
        ticker_dir.mkdir()

        ohlcv = _make_ohlcv_df(30, start="2025-04-01")
        _write_parquet(ohlcv, ticker_dir / "ohlcv_2021_now.parquet")

        # No news file — OHLCV should still work
        ohlcv_summary, news_text = _load_raw_evidence(
            ticker, datetime(2025, 6, 1), tmp_path
        )
        assert "No news available." == news_text
        assert ohlcv_summary != "No OHLCV data available."

    def test_future_news_excluded_integration(self, tmp_path):
        """Future news must be excluded in full _load_raw_evidence pipeline."""
        ticker = "TEST.HK"
        ticker_dir = tmp_path / ticker
        ticker_dir.mkdir()

        ohlcv = _make_ohlcv_df(30, start="2025-04-01")
        _write_parquet(ohlcv, ticker_dir / "ohlcv_2021_now.parquet")

        news = _make_news_df([
            {"title": "Past news", "published_time": "2025-05-15T10:00:00Z", "source": "A"},
            {"title": "Future news", "published_time": "2025-07-01T10:00:00Z", "source": "B"},
        ])
        _write_parquet(news, ticker_dir / "news_latest.parquet")

        _, news_text = _load_raw_evidence(ticker, datetime(2025, 6, 1), tmp_path)
        assert "Past news" in news_text
        assert "Future news" not in news_text

    def test_no_parquet_files(self, tmp_path):
        """When no data files exist, return default text."""
        ticker = "EMPTY.HK"
        ohlcv_summary, news_text = _load_raw_evidence(
            ticker, datetime(2025, 6, 1), tmp_path
        )
        assert ohlcv_summary == "No OHLCV data available."
        assert news_text == "No news available."

    def test_all_future_news_returns_no_news(self, tmp_path):
        """When all news are in the future, return 'No news available.'"""
        ticker = "TEST.HK"
        ticker_dir = tmp_path / ticker
        ticker_dir.mkdir()

        news = _make_news_df([
            {"title": "Future1", "published_time": "2025-08-01T00:00:00Z", "source": "A"},
            {"title": "Future2", "published_time": "2025-09-01T00:00:00Z", "source": "B"},
        ])
        _write_parquet(news, ticker_dir / "news_latest.parquet")

        _, news_text = _load_raw_evidence(ticker, datetime(2025, 6, 1), tmp_path)
        assert news_text == "No news available."
