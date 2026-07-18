"""Daily price + news collection for HK stocks.

Fetches OHLCV (last 5 trading days) and news (Tavily + NewsAPI) for
each ticker, appends to rolling parquet files under ``data/daily/``.

Output structure::

    data/daily/
        status.json          # last run status
        collect.log          # rolling log
        {TICKER}/
            ohlcv.parquet    # cumulative OHLCV history
            news.parquet     # cumulative news history

Run manually or schedule via cron / Task Scheduler::

    python -m src.daily.collect_daily             # all tickers
    python -m src.daily.collect_daily 0700.HK     # single ticker
    python -m src.daily.collect_daily --check     # view last run status
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.collectors.news_collector import NewsCollector
from src.collectors.newsapi_collector import NewsAPICollector
from src.config import CollectionConfig

OUTPUT_ROOT = Path("data/daily")
STATUS_FILE = OUTPUT_ROOT / "status.json"
LOG_FILE = OUTPUT_ROOT / "collect.log"


def _setup_logging() -> logging.Logger:
    """Configure logging to both console and file."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # prevent duplicate messages via root logger

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = _setup_logging()


# ── OHLCV ──────────────────────────────────────────────────────────────


def fetch_ohlcv(ticker: str, days_back: int = 5) -> pd.DataFrame:
    """Download recent OHLCV from Yahoo Finance (last *days_back* days)."""
    end = date.today() + timedelta(days=1)  # yfinance end is exclusive
    start = date.today() - timedelta(days=days_back + 5)  # extra buffer for weekends
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        log.warning("OHLCV: no data for %s", ticker)
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["ticker"] = ticker
    # Keep only the columns we need
    keep = [c for c in ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def append_parquet(new_df: pd.DataFrame, path: Path, dedup_col: str = "date") -> pd.DataFrame:
    """Append *new_df* to parquet at *path*, deduplicate by *dedup_col*, return merged."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    if dedup_col in combined.columns:
        combined = combined.drop_duplicates(subset=[dedup_col], keep="last")
    combined = combined.sort_values(dedup_col).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined


# ── News ───────────────────────────────────────────────────────────────


def fetch_news(ticker: str, config: CollectionConfig, limit: int = 10) -> pd.DataFrame:
    """Fetch news from Tavily + NewsAPI, dedup, return combined."""
    frames: list[pd.DataFrame] = []

    # Tavily
    try:
        tavily = NewsCollector(config)
        df = tavily.fetch_news(ticker, limit=limit)
        if not df.empty:
            frames.append(df)
    except Exception as e:
        log.warning("Tavily failed for %s: %s", ticker, e)

    # NewsAPI (optional)
    try:
        newsapi = NewsAPICollector(config)
        df = newsapi.fetch_news(ticker, limit=limit)
        if not df.empty:
            frames.append(df)
    except ValueError:
        log.info("NewsAPI key not set, skipping")
    except Exception as e:
        log.warning("NewsAPI failed for %s: %s", ticker, e)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "url" in combined.columns:
        combined = combined.drop_duplicates(subset=["url"], keep="first")
    if "title" in combined.columns:
        combined = combined.drop_duplicates(subset=["title"], keep="first")
    if "published_time" in combined.columns:
        combined = combined.sort_values("published_time", ascending=False).reset_index(drop=True)
    return combined


# ── Main ───────────────────────────────────────────────────────────────


def collect_one_ticker(ticker: str, config: CollectionConfig) -> dict:
    """Collect OHLCV + news for one ticker.  Returns summary dict."""
    ticker_dir = OUTPUT_ROOT / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    summary = {"ticker": ticker}

    # ── OHLCV ──
    ohlcv_new = fetch_ohlcv(ticker, days_back=5)
    if not ohlcv_new.empty:
        ohlcv_all = append_parquet(ohlcv_new, ticker_dir / "ohlcv.parquet", dedup_col="date")
        summary["ohlcv_rows_total"] = len(ohlcv_all)
        summary["ohlcv_rows_new"] = len(ohlcv_new)
        log.info("OHLCV %s: %d new rows, %d total", ticker, len(ohlcv_new), len(ohlcv_all))
    else:
        summary["ohlcv_rows_total"] = 0
        summary["ohlcv_rows_new"] = 0

    # ── News ──
    news_new = fetch_news(ticker, config, limit=config.news_limit_per_ticker)
    if not news_new.empty:
        # Add collection date for dedup
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if "collected_date" not in news_new.columns:
            news_new["collected_date"] = today_str
        news_all = append_parquet(news_new, ticker_dir / "news.parquet", dedup_col="url")
        summary["news_rows_total"] = len(news_all)
        summary["news_rows_new"] = len(news_new)
        log.info("News %s: %d new rows, %d total", ticker, len(news_new), len(news_all))
    else:
        summary["news_rows_total"] = 0
        summary["news_rows_new"] = 0

    return summary


# ── Status ─────────────────────────────────────────────────────────────


def save_status(results: list[dict]) -> None:
    """Write run status to ``status.json``."""
    ok = sum(1 for r in results if "error" not in r)
    fail = len(results) - ok
    status = {
        "last_run": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers_ok": ok,
        "tickers_failed": fail,
        "results": results,
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def show_status() -> None:
    """Print last run status from ``status.json``."""
    if not STATUS_FILE.exists():
        print("No status file — daily collection has never run.")
        return

    status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    last_run = status.get("last_run", "?")
    ok = status.get("tickers_ok", 0)
    fail = status.get("tickers_failed", 0)

    print(f"Last run:  {last_run}")
    print(f"Status:    {ok} OK, {fail} failed")
    print()

    rows = []
    for r in status.get("results", []):
        ticker = r.get("ticker", "?")
        if "error" in r:
            rows.append({"ticker": ticker, "status": "FAIL", "detail": r["error"][:60]})
        else:
            rows.append({
                "ticker": ticker,
                "status": "OK",
                "ohlcv_new": r.get("ohlcv_rows_new", 0),
                "ohlcv_total": r.get("ohlcv_rows_total", 0),
                "news_new": r.get("news_rows_new", 0),
                "news_total": r.get("news_rows_total", 0),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Check if data files exist and are fresh
    print()
    for ticker_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for fname in ("ohlcv.parquet", "news.parquet"):
            fpath = ticker_dir / fname
            if fpath.exists():
                mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                age = datetime.now() - mtime
                size_kb = fpath.stat().st_size / 1024
                if age.days >= 2:
                    print(f"  ! {ticker}/{fname}: {age.days}d old ({size_kb:.0f} KB)")
                else:
                    h = age.seconds // 3600
                    m = (age.seconds % 3600) // 60
                    print(f"  OK {ticker}/{fname}: {h}h{m}m old ({size_kb:.0f} KB)")


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Daily price + news collection")
    parser.add_argument("tickers", nargs="*", help="Tickers to collect (default: from config)")
    parser.add_argument("--check", action="store_true", help="Show last run status and exit")
    args = parser.parse_args()

    if args.check:
        show_status()
        return

    config = CollectionConfig()
    tickers = args.tickers if args.tickers else config.tickers

    today = date.today()
    weekday = today.weekday()
    if weekday >= 5:
        log.info("Today is %s (weekend) — collecting anyway for news, price will be stale.", today.strftime("%A"))

    log.info("Daily collection for %s — tickers: %s", today.isoformat(), ", ".join(tickers))

    results = []
    for ticker in tickers:
        try:
            summary = collect_one_ticker(ticker, config)
            results.append(summary)
        except Exception as e:
            log.error("Failed for %s: %s", ticker, e)
            results.append({"ticker": ticker, "error": str(e)})

    # Summary table
    df = pd.DataFrame(results)
    log.info("\n%s", df.to_string(index=False))

    save_status(results)
    log.info("Status saved to %s", STATUS_FILE)


if __name__ == "__main__":
    main()
