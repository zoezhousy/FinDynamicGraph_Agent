from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry


class MarketCollector:
    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

    def _download(self, ticker: str) -> pd.DataFrame:
        @retry(
            max_retries=self.config.max_retries,
            initial_backoff_seconds=self.config.initial_backoff_seconds,
            backoff_multiplier=self.config.backoff_multiplier,
            max_backoff_seconds=self.config.max_backoff_seconds,
        )
        def _call() -> pd.DataFrame:
            self.rate_limiter.wait()
            frame = yf.download(
                ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if frame is None or frame.empty:
                raise ValueError(f"No OHLCV data returned for {ticker}")
            return frame

        return _call()

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        frame = self._download(ticker)
        frame = frame.reset_index()

        if "Date" not in frame.columns:
            raise ValueError(f"Missing Date column for {ticker}")

        keep_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        existing_cols = [c for c in keep_cols if c in frame.columns]
        frame = frame[existing_cols].copy()
        frame["ticker"] = ticker
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date

        rename_map = {"Date": "date", "Adj Close": "adj_close", "Volume": "volume"}
        frame = frame.rename(columns=rename_map)
        frame.columns = [c.lower() for c in frame.columns]

        # Stable ordering for downstream merge/join.
        frame = frame.sort_values("date").reset_index(drop=True)
        logging.info("Fetched OHLCV for %s with %d rows", ticker, len(frame))
        return frame

