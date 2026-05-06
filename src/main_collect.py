from __future__ import annotations

import logging
from pathlib import Path

from src.collectors.market_collector import MarketCollector
from src.collectors.news_collector import NewsCollector
from src.config import CollectionConfig
from src.utils.io import ensure_dir, save_parquet
from src.kg.store_neo4j import Neo4jKGStore
from src.kg.update_pipeline import build_kg_batch_for_ticker


def setup_logging(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


def run_collection(config: CollectionConfig) -> None:
    market_collector = MarketCollector(config)
    news_collector = NewsCollector(config)

    # Optional: initialize KG store if Neo4j is configured in environment.
    kg_store = None
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if neo4j_uri and neo4j_user and neo4j_password:
        kg_store = Neo4jKGStore(neo4j_uri, neo4j_user, neo4j_password)
        if kg_store.health_check():
            kg_store.init_constraints()
        else:
            logging.warning("Neo4j health check failed, KG writes will be skipped.")
            kg_store = None

    for ticker in config.tickers:
        ticker_dir = config.output_root / ticker
        ensure_dir(ticker_dir)
        logging.info("Start collecting ticker=%s", ticker)

        ohlcv_df = None
        news_df = None
        try:
            ohlcv_df = market_collector.fetch_ohlcv(ticker)
            save_parquet(ohlcv_df, ticker_dir / "ohlcv_2021_2025.parquet")
        except Exception as exc:
            logging.exception("OHLCV collection failed for %s: %s", ticker, exc)

        try:
            news_df = news_collector.fetch_news(ticker, limit=config.news_limit_per_ticker)
            save_parquet(news_df, ticker_dir / "news_latest.parquet")
        except Exception as exc:
            logging.exception("News collection failed for %s: %s", ticker, exc)

        if kg_store and ohlcv_df is not None and news_df is not None:
            try:
                batch = build_kg_batch_for_ticker(ticker, ohlcv_df, news_df)
                kg_store.upsert_entities(batch.entities)
                kg_store.upsert_evidences(batch.evidences)
                kg_store.upsert_relations(batch.relations)
            except Exception as exc:
                logging.exception("KG update failed for %s: %s", ticker, exc)

        logging.info("Done ticker=%s", ticker)


def main() -> None:
    config = CollectionConfig(
        # You can replace with your full Hang Seng constituents.
        tickers=["0005.HK", "0700.HK", "1299.HK"],
        start_date="2021-01-01",
        end_date="2025-12-31",
        news_limit_per_ticker=10,
    )
    setup_logging(config.log_file)
    run_collection(config)


    if kg_store is not None:
        kg_store.close()


if __name__ == "__main__":
    main()

