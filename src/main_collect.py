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
from src.kg.ingestion import company_entity, write_graph_batch
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
    from src.kg.connections import Neo4jSettings, initialized_store
    settings = Neo4jSettings.from_env(config.neo4j_database, required=False)
    return initialized_store(settings, Neo4jKGStore, optional=True)


def build_company_entity(ticker: str) -> Entity:
    return company_entity(ticker)


from src.collectors.cache import (
    check_existing_data_dates as _check_existing_data_dates,
    dedup_news_frames as _dedup_news_frames,
    merge_fundamentals_history as _merge_fundamentals_history,
    fetch_ohlcv_with_cache as _fetch_ohlcv_with_cache,
)


def run_collection(config: CollectionConfig, **kwargs) -> None:
    from src.collectors.service import run_collection as collect
    kwargs.setdefault("store_factory", _init_kg_store)
    return collect(config, **kwargs)


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