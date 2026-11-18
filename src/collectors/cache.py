"""Shared file operations. Caller explicitly chooses legacy ordering policy."""
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
from src.collectors.market_collector import MarketCollector

OHLCV_NAMES = ("ohlcv_2021_now.parquet", "ohlcv_2021_2025.parquet")

def resolve_ohlcv_path(root, ticker, *, legacy_first=False, current_only=False):
    names = OHLCV_NAMES[:1] if current_only else OHLCV_NAMES[::-1] if legacy_first else OHLCV_NAMES
    for name in names:
        path = Path(root) / ticker / name
        if path.exists():
            return path
    return Path(root) / ticker / names[-1]

def load_ohlcv(root, ticker, *, legacy_first=False):
    return pd.read_parquet(resolve_ohlcv_path(root, ticker, legacy_first=legacy_first))

def check_existing_data_dates(ticker_dir: Path, ticker: str) -> dict[str, str | None]:
    """Check the dates of existing data files for a ticker."""
    dates = {}

    # Check OHLCV
    ohlcv_path = ticker_dir / "ohlcv_2021_now.parquet"
    if not ohlcv_path.exists():
        ohlcv_path = ticker_dir / "ohlcv_2021_2025.parquet"
    if ohlcv_path.exists():
        try:
            df = pd.read_parquet(ohlcv_path)
            if "date" in df.columns and len(df) > 0:
                dates["ohlcv_last"] = str(pd.to_datetime(df["date"]).max().date())
        except Exception:
            dates["ohlcv_last"] = None

    # Check news
    news_path = ticker_dir / "news_combined_latest.parquet"
    if news_path.exists():
        try:
            df = pd.read_parquet(news_path)
            if "published_time" in df.columns and len(df) > 0:
                dates["news_last"] = str(df["published_time"].max())
        except Exception:
            dates["news_last"] = None

    # Check fundamentals
    fund_path = ticker_dir / "fundamentals_history.parquet"
    if fund_path.exists():
        try:
            df = pd.read_parquet(fund_path)
            if "as_of_date" in df.columns and len(df) > 0:
                dates["fundamentals_last"] = str(df["as_of_date"].max())
        except Exception:
            dates["fundamentals_last"] = None

    return dates


def dedup_news_frames(frames: list[pd.DataFrame], *, sort_by_time: bool = True) -> pd.DataFrame:
    """Merge and deduplicate multiple news DataFrames by URL, then title."""
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True)
    # Deduplicate by URL first (most reliable), then title as fallback
    if "url" in merged.columns:
        merged = merged.drop_duplicates(subset=["url"], keep="first")
    if "title" in merged.columns:
        merged = merged.drop_duplicates(subset=["title"], keep="first")
    if sort_by_time and "published_time" in merged.columns:
        merged = merged.sort_values("published_time", ascending=False).reset_index(drop=True)
    return merged


def merge_fundamentals_history(
    new_df: pd.DataFrame,
    history_path: Path,
    max_history_days: int = 365,
) -> pd.DataFrame:
    """Append new fundamentals snapshot to historical file, prune old data."""
    if history_path.exists():
        try:
            old = pd.read_parquet(history_path)
            combined = pd.concat([old, new_df], ignore_index=True)
            # Keep last N days of snapshots to prevent unbounded growth
            if "as_of_date" in combined.columns:
                combined["as_of_date_dt"] = pd.to_datetime(combined["as_of_date"], errors="coerce")
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_history_days)
                combined = combined[combined["as_of_date_dt"] >= cutoff].copy()
                combined = combined.drop(columns=["as_of_date_dt"])
            combined = combined.drop_duplicates(
                subset=["ticker", "metric", "as_of_date"], keep="last"
            ).reset_index(drop=True)
            return combined
        except Exception as exc:
            logging.warning("Failed to load fundamentals history, starting fresh: %s", exc)
    return new_df


def fetch_ohlcv_with_cache(
    ticker: str,
    market_collector: MarketCollector,
    ticker_dir: Path,
) -> pd.DataFrame | None:
    """Fetch OHLCV: load cached history first, then merge with incremental fetch."""
    cached_path = ticker_dir / "ohlcv_2021_now.parquet"
    if not cached_path.exists():
        cached_path = ticker_dir / "ohlcv_2021_2025.parquet"

    cached_df = None
    last_date = None
    if cached_path.exists():
        try:
            cached_df = pd.read_parquet(cached_path)
            if "date" in cached_df.columns and len(cached_df) > 0:
                last_date = pd.to_datetime(cached_df["date"]).max()
                logging.info(
                    "Loaded cached OHLCV for %s: %d rows, last date: %s",
                    ticker, len(cached_df), last_date.date(),
                )
        except Exception as exc:
            logging.warning("Failed to load cached OHLCV for %s: %s", ticker, exc)

    # Incremental fetch: only fetch from last_date + 1 day if cache exists
    try:
        if last_date is not None:
            # Only fetch new data from the day after last cached date
            next_day = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            today_str = datetime.now().strftime("%Y-%m-%d")
            if next_day >= today_str:
                # Next day is today or future — data not published yet; skip fetch
                logging.info(
                    "Skipping incremental OHLCV fetch for %s: next_day (%s) >= today (%s), using cached data",
                    ticker, next_day, today_str,
                )
                return cached_df
            logging.info("Incremental fetch for %s from %s to %s", ticker, next_day, market_collector.config.end_date)
            fresh_df = market_collector.fetch_ohlcv(ticker, start_date=next_day)
        else:
            # No cache, fetch full range
            fresh_df = market_collector.fetch_ohlcv(ticker)

        if cached_df is not None and last_date is not None:
            # Merge: append new data to cached data
            if not fresh_df.empty:
                fresh_df = pd.concat([cached_df, fresh_df], ignore_index=True)
                fresh_df = fresh_df.drop_duplicates(subset=["date"], keep="last")
                fresh_df = fresh_df.sort_values("date").reset_index(drop=True)
                logging.info(
                    "Merged cached + new OHLCV for %s: %d total rows",
                    ticker, len(fresh_df),
                )
            else:
                logging.info("No new OHLCV data for %s after %s", ticker, last_date.date())
                fresh_df = cached_df
        return fresh_df
    except Exception as exc:
        logging.warning("OHLCV fetch failed for %s, using cached data: %s", ticker, exc)
        return cached_df


