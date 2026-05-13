from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry


class FundamentalCollector:
    """Collect lightweight fundamental information from Yahoo Finance / yfinance.

    The important implementation detail:
    - value is always stored as string for Parquet compatibility.
    - numeric_value stores float when conversion is possible.
    """

    FUNDAMENTAL_FIELDS = [
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "profitMargins",
        "operatingMargins",
        "grossMargins",
        "returnOnEquity",
        "returnOnAssets",
        "revenueGrowth",
        "earningsGrowth",
        "debtToEquity",
        "currentRatio",
        "quickRatio",
        "totalRevenue",
        "grossProfits",
        "ebitda",
        "freeCashflow",
        "operatingCashflow",
        "totalDebt",
        "totalCash",
        "bookValue",
        "dividendYield",
        "payoutRatio",
        "beta",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "recommendationMean",
        "recommendationKey",
        "targetMeanPrice",
        "currentPrice",
        "sector",
        "industry",
        "longName",
        "shortName",
        "currency",
        "exchange",
        "quoteType",
    ]

    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

    def fetch_fundamentals(self, ticker: str) -> pd.DataFrame:
        @retry(
            max_retries=self.config.max_retries,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
            backoff_multiplier=self.config.backoff_multiplier,
            max_backoff_seconds=self.config.max_backoff_seconds,
        )
        def _call() -> pd.DataFrame:
            self.rate_limiter.wait()

            yf_ticker = yf.Ticker(ticker)
            info: dict[str, Any] = yf_ticker.info or {}

            if not info:
                raise ValueError(f"No fundamental info returned for {ticker}")

            rows = []
            collected_at = datetime.utcnow().isoformat()

            for field in self.FUNDAMENTAL_FIELDS:
                raw_value = info.get(field)
                if raw_value is None:
                    continue

                rows.append(
                    {
                        "ticker": str(ticker),
                        "metric": str(field),
                        "value": _safe_string(raw_value),
                        "numeric_value": _safe_float(raw_value),
                        "value_type": type(raw_value).__name__,
                        "source": "yfinance.info",
                        "as_of_date": collected_at,
                    }
                )

            if not rows:
                raise ValueError(f"No usable fundamental fields returned for {ticker}")

            frame = pd.DataFrame(rows)

            # Make Parquet schema stable.
            string_cols = ["ticker", "metric", "value", "value_type", "source", "as_of_date"]
            for col in string_cols:
                frame[col] = frame[col].astype("string")

            frame["numeric_value"] = pd.to_numeric(frame["numeric_value"], errors="coerce")

            logging.info("Fetched fundamentals for %s with %d fields", ticker, len(frame))
            return frame

        return _call()


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _safe_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return str(value)
    return str(value)