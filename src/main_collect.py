from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.agents.technical_agent import TechnicalAgent
from src.collectors.fundamental_collector import FundamentalCollector
from src.collectors.market_collector import MarketCollector
from src.collectors.news_collector import NewsCollector
from src.config import CollectionConfig
from src.kg.schema import Entity
from src.kg.store_neo4j import Neo4jKGStore
from src.kg.update_pipeline import build_fundamentals_from_frame, build_news_from_frame
from src.utils.io import ensure_dir, save_parquet


def setup_logging(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )


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


def run_collection(config: CollectionConfig) -> None:
    market_collector = MarketCollector(config)
    news_collector = NewsCollector(config)
    fundamental_collector = FundamentalCollector(config)
    technical_agent = TechnicalAgent()
    kg_store = _init_kg_store(config)

    try:
        for ticker in config.tickers:
            ticker_dir = config.output_root / ticker
            ensure_dir(ticker_dir)
            logging.info("Start collecting ticker=%s", ticker)

            ohlcv_df = None
            news_df = None
            fundamentals_df = None

            try:
                ohlcv_df = market_collector.fetch_ohlcv(ticker)
                save_parquet(ohlcv_df, ticker_dir / "ohlcv_2021_now.parquet")
            except Exception as exc:
                logging.exception("OHLCV collection failed for %s: %s", ticker, exc)

            try:
                news_df = news_collector.fetch_news(ticker, limit=config.news_limit_per_ticker)
                save_parquet(news_df, ticker_dir / "news_latest.parquet")
            except Exception as exc:
                logging.exception("News collection failed for %s: %s", ticker, exc)

            if config.collect_fundamentals:
                try:
                    fundamentals_df = fundamental_collector.fetch_fundamentals(ticker)
                    save_parquet(fundamentals_df, ticker_dir / "fundamentals_latest.parquet")
                except Exception as exc:
                    logging.exception("Fundamental collection failed for %s: %s", ticker, exc)

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

            if kg_store and news_df is not None:
                try:
                    news_entities, evidences, news_relations = build_news_from_frame(ticker, news_df)
                    kg_store.upsert_entities(news_entities)
                    kg_store.upsert_evidences(evidences)
                    kg_store.upsert_relations(news_relations)
                    logging.info("News graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("News KG update failed for %s: %s", ticker, exc)

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