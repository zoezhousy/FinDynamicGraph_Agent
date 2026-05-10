from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

from src.config import CollectionConfig
from src.utils.rate_limit import RateLimiter
from src.utils.retry import retry


class MarketCollector:
    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(config.request_interval_seconds)

    def _download_yfinance(self, ticker: str) -> pd.DataFrame:
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

    def _download_yahoo_chart(self, ticker: str) -> pd.DataFrame:
        self.rate_limiter.wait()
        period1 = int(pd.Timestamp(self.config.start_date).timestamp())
        period2 = int(pd.Timestamp(self.config.end_date).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "period1": period1,
            "period2": period2,
            "interval": "1d",
            "includeAdjustedClose": "true",
            "events": "div,splits",
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        }
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        result = ((payload.get("chart") or {}).get("result") or [None])[0]
        if not result:
            raise ValueError(f"No chart result returned for {ticker}")

        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [None])[0] or {}
        adjclose_block = ((result.get("indicators") or {}).get("adjclose") or [None])[0] or {}
        if not timestamps:
            raise ValueError(f"No chart timestamps returned for {ticker}")

        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(timestamps, unit="s"),
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Adj Close": adjclose_block.get("adjclose", quote.get("close")),
                "Volume": quote.get("volume"),
            }
        )
        frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])
        if frame.empty:
            raise ValueError(f"Yahoo chart returned only empty OHLCV rows for {ticker}")
        return frame

    def _download_stooq_csv(self, ticker: str) -> pd.DataFrame:
        symbol = ticker.lower()
        self.rate_limiter.wait()
        response = requests.get(
            f"https://stooq.com/q/d/l/?s={symbol}&i=d",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        text = response.text.strip()
        if not text or text.lower().startswith("no data"):
            raise ValueError(f"No OHLCV data returned from Stooq for {ticker}")
        frame = pd.read_csv(StringIO(text))
        if frame.empty or "Date" not in frame.columns:
            raise ValueError(f"Unexpected Stooq response format for {ticker}")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"]).copy()
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)
        frame = frame[(frame["Date"] >= start) & (frame["Date"] <= end)].copy()
        if frame.empty:
            raise ValueError(f"No OHLCV rows in configured date range for {ticker}")
        return frame

    def _download(self, ticker: str) -> pd.DataFrame:
        errors: list[str] = []
        for loader in (self._download_yahoo_chart, self._download_yfinance, self._download_stooq_csv):
            try:
                frame = loader(ticker)
                logging.info("Market data provider %s succeeded for %s", loader.__name__, ticker)
                return frame
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")
                logging.warning("%s failed for %s: %s", loader.__name__, ticker, exc)
        raise ValueError(f"All market data providers failed for {ticker}: {' | '.join(errors)}")

    def fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        frame = self._download(ticker)
        frame = frame.reset_index(drop=True)

        if "Date" not in frame.columns:
            raise ValueError(f"Missing Date column for {ticker}")

        keep_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        existing_cols = [c for c in keep_cols if c in frame.columns]
        frame = frame[existing_cols].copy()
        if "Adj Close" not in frame.columns and "Close" in frame.columns:
            frame["Adj Close"] = frame["Close"]
        frame["ticker"] = ticker
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date

        rename_map = {"Date": "date", "Adj Close": "adj_close", "Volume": "volume"}
        frame = frame.rename(columns=rename_map)
        frame.columns = [c.lower() for c in frame.columns]

        frame = frame.sort_values("date").reset_index(drop=True)
        logging.info("Fetched OHLCV for %s with %d rows", ticker, len(frame))
        return frame
