from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.agents.technical_agent import TechnicalAgent
from src.collectors.fundamental_collector import FundamentalCollector
from src.collectors.global_news_collector import GlobalNewsCollector
from src.collectors.market_collector import MarketCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.newsapi_collector import NewsAPICollector
from src.config import CollectionConfig
from src.kg.schema import Entity
from src.kg.store_neo4j import Neo4jKGStore
from src.kg.update_pipeline import build_fundamentals_from_frame, build_global_news_from_frame, build_news_from_frame, build_risk_events_from_frame
from src.utils.io import ensure_dir, save_parquet


def setup_logging(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    # Suppress noisy Neo4j driver notifications
    logging.getLogger("neo4j").setLevel(logging.WARNING)


def _init_kg_store(config: CollectionConfig) -> Neo4jKGStore | None:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not (neo4j_uri and neo4j_user and neo4j_password):
        logging.info("Neo4j environment not configured, KG writes will be skipped.")
        return None

    kg_store = Neo4jKGStore(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        database=config.neo4j_database,
    )
    if not kg_store.health_check():
        logging.warning("Neo4j health check failed, KG writes will be skipped.")
        kg_store.close()
        return None

    kg_store.init_constraints()
    return kg_store


def build_company_entity(ticker: str) -> Entity:
    return Entity(
        entity_id=f"company:{ticker}",
        type="Company",
        properties={"ticker": ticker},
    )


def _check_existing_data_dates(ticker_dir: Path, ticker: str) -> dict[str, str | None]:
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


def _dedup_news_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
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
    if "published_time" in merged.columns:
        merged = merged.sort_values("published_time", ascending=False).reset_index(drop=True)
    return merged


def _merge_fundamentals_history(
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
                cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=max_history_days)
                combined = combined[combined["as_of_date_dt"] >= cutoff].copy()
                combined = combined.drop(columns=["as_of_date_dt"])
            combined = combined.drop_duplicates(
                subset=["ticker", "metric", "as_of_date"], keep="last"
            ).reset_index(drop=True)
            return combined
        except Exception as exc:
            logging.warning("Failed to load fundamentals history, starting fresh: %s", exc)
    return new_df


def _fetch_ohlcv_with_cache(
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


def run_collection(config: CollectionConfig) -> None:
    market_collector = MarketCollector(config)
    news_collector = NewsCollector(config)
    global_news_collector = GlobalNewsCollector(config)
    newsapi_collector = NewsAPICollector(config)
    fundamental_collector = FundamentalCollector(config)
    technical_agent = TechnicalAgent()
    kg_store = _init_kg_store(config)

    try:
        # Fetch OHLCV for all tickers in parallel (heaviest I/O)
        with ThreadPoolExecutor(max_workers=len(config.tickers)) as pool:
            ohlcv_futures = {
                pool.submit(
                    _fetch_ohlcv_with_cache, t, market_collector, config.output_root / t
                ): t
                for t in config.tickers
            }
            ohlcv_cache: dict[str, pd.DataFrame | None] = {}
            for fut in as_completed(ohlcv_futures):
                t = ohlcv_futures[fut]
                try:
                    ohlcv_cache[t] = fut.result()
                except Exception as exc:
                    logging.error("OHLCV parallel fetch failed for %s: %s", t, exc)
                    ohlcv_cache[t] = None

        # Process each ticker sequentially (news + KG updates)
        for ticker in config.tickers:
            ticker_dir = config.output_root / ticker
            ensure_dir(ticker_dir)
            logging.info("Start collecting ticker=%s", ticker)

            # Check existing data dates
            existing_dates = _check_existing_data_dates(ticker_dir, ticker)
            if existing_dates:
                logging.info("Existing data for %s: %s", ticker, existing_dates)
                print(f"[{ticker}] Existing data: {existing_dates}")

            ohlcv_df = ohlcv_cache.get(ticker)
            if ohlcv_df is not None:
                save_parquet(ohlcv_df, ticker_dir / "ohlcv_2021_now.parquet")

            news_df = None
            global_news_df = None
            fundamentals_df = None

            # ── News: fetch from both sources, then dedup ──
            news_frames: list[pd.DataFrame] = []
            newsapi_df = None
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(news_collector.fetch_news, ticker, config.news_limit_per_ticker): "tavily",
                    pool.submit(newsapi_collector.fetch_news, ticker, config.news_limit_per_ticker): "newsapi",
                }
                for fut in as_completed(futures):
                    source = futures[fut]
                    try:
                        df = fut.result()
                        if df is not None and not df.empty:
                            news_frames.append(df)
                            if source == "newsapi":
                                newsapi_df = df
                                save_parquet(df, ticker_dir / "newsapi_latest.parquet")
                            else:
                                save_parquet(df, ticker_dir / "news_latest.parquet")
                    except Exception as exc:
                        logging.exception("News (%s) failed for %s: %s", source, ticker, exc)

            news_df = _dedup_news_frames(news_frames)
            if not news_df.empty:
                save_parquet(news_df, ticker_dir / "news_combined_latest.parquet")
                logging.info("Combined deduped news for %s: %d articles", ticker, len(news_df))

            # ── Global news ──
            try:
                global_news_df = global_news_collector.fetch_global_news(ticker=ticker, limit=50)
                if global_news_df is not None and not global_news_df.empty:
                    save_parquet(global_news_df, ticker_dir / "global_news_latest.parquet")
            except Exception as exc:
                logging.exception("Global news collection failed for %s: %s", ticker, exc)

            # ── Fundamentals: append to historical file ──
            if config.collect_fundamentals:
                try:
                    fund_df = fundamental_collector.fetch_fundamentals(ticker)
                    history_path = ticker_dir / "fundamentals_history.parquet"
                    fundamentals_df = _merge_fundamentals_history(fund_df, history_path)
                    save_parquet(fundamentals_df, history_path)
                    save_parquet(fund_df, ticker_dir / "fundamentals_latest.parquet")
                except Exception as exc:
                    logging.warning("Fundamental fetch failed for %s, trying cached: %s", ticker, exc)
                    cached_fund = ticker_dir / "fundamentals_latest.parquet"
                    if cached_fund.exists():
                        try:
                            fundamentals_df = pd.read_parquet(cached_fund)
                            logging.info("Loaded cached fundamentals for %s: %d rows", ticker, len(fundamentals_df))
                        except Exception as exc2:
                            logging.error("Cached fundamentals also failed for %s: %s", ticker, exc2)
                    else:
                        logging.error("No cached fundamentals for %s", ticker)

            if kg_store and config.reset_graph_before_collection:
                try:
                    kg_store.clear_generated_data_for_ticker(
                        ticker,
                        preserve_historical_traces=config.preserve_historical_traces_on_reset,
                    )
                    logging.info(
                        "Reset generated graph data for ticker=%s; preserve_historical_traces=%s",
                        ticker,
                        config.preserve_historical_traces_on_reset,
                    )
                except Exception as exc:
                    logging.exception("Failed resetting graph data for %s: %s", ticker, exc)

            if kg_store:
                try:
                    kg_store.upsert_entities([build_company_entity(ticker)])
                except Exception as exc:
                    logging.exception("Company KG upsert failed for %s: %s", ticker, exc)

            # update Techinical KG
            if kg_store and ohlcv_df is not None:
                try:
                    signal_entities, signal_relations = technical_agent.build_signal_entities_from_ohlcv(
                        ticker,
                        ohlcv_df,
                    )

                    kg_store.upsert_entities(signal_entities)
                    kg_store.upsert_relations(signal_relations)
                    logging.info("Technical graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Technical KG update failed for %s: %s", ticker, exc)

            # Update News KG
            if kg_store and news_df is not None:
                try:
                    news_entities, evidences, news_relations = build_news_from_frame(ticker, news_df)
                    kg_store.upsert_entities(news_entities)
                    kg_store.upsert_evidences(evidences)
                    kg_store.upsert_relations(news_relations)
                    logging.info("News graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("News KG update failed for %s: %s", ticker, exc)

            if kg_store and newsapi_df is not None and not newsapi_df.empty:
                try:
                    na_entities, na_evidences, na_relations = build_news_from_frame(ticker, newsapi_df)
                    kg_store.upsert_entities(na_entities)
                    kg_store.upsert_evidences(na_evidences)
                    kg_store.upsert_relations(na_relations)
                    logging.info("NewsAPI graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("NewsAPI KG update failed for %s: %s", ticker, exc)

            if kg_store and global_news_df is not None and not global_news_df.empty:
                try:
                    gn_entities, gn_evidences, gn_relations = build_global_news_from_frame(ticker, global_news_df)
                    kg_store.upsert_entities(gn_entities)
                    kg_store.upsert_evidences(gn_evidences)
                    kg_store.upsert_relations(gn_relations)
                    logging.info("Global news graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Global news KG update failed for %s: %s", ticker, exc)

            # Update Fundamental KG 
            if kg_store and fundamentals_df is not None:
                try:
                    fundamental_entities, fundamental_evidences, fundamental_relations = (
                        build_fundamentals_from_frame(ticker, fundamentals_df)
                    )
                    kg_store.upsert_entities(fundamental_entities)
                    kg_store.upsert_evidences(fundamental_evidences)
                    kg_store.upsert_relations(fundamental_relations)
                    logging.info("Fundamental graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Fundamental KG update failed for %s: %s", ticker, exc)
            
            # Update risk events in KG based on OHLCV and fundamentals.
            if kg_store and ohlcv_df is not None:
                try:
                    risk_entities, risk_evidences, risk_relations = build_risk_events_from_frame(
                        ticker=ticker,
                        ohlcv=ohlcv_df,
                        fundamentals=fundamentals_df,
                    )
                    kg_store.upsert_entities(risk_entities)
                    kg_store.upsert_evidences(risk_evidences)
                    kg_store.upsert_relations(risk_relations)
                    logging.info("Risk graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Risk KG update failed for %s: %s", ticker, exc)

            logging.info("Done ticker=%s", ticker)
    finally:
        if kg_store is not None:
            kg_store.close()


def main() -> None:
    load_dotenv()

    today_str = datetime.now().strftime("%Y-%m-%d")

    config = CollectionConfig(
        tickers=["0005.HK", "0700.HK", "1299.HK"],
        start_date="2021-01-01",
        end_date=today_str,
        news_limit_per_ticker=10,
        reset_graph_before_collection=False,
        preserve_historical_traces=True,
        collect_fundamentals=True,
    )

    print(f"Collect range: {config.start_date} -> {config.end_date}")
    print(f"Reset graph before collection: {config.reset_graph_before_collection}")
    print(f"Collect fundamentals: {config.collect_fundamentals}")

    setup_logging(config.log_file)
    run_collection(config)


if __name__ == "__main__":
    main()