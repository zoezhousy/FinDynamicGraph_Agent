"""Import news parquet files from a data directory into Neo4j.

Usage:
    python -m src.scripts.import_news_to_neo4j [--data-dir PATH]

Default data-dir: C:/Users/zoezh/Downloads/data

Processes per ticker:
    - news_combined_latest.parquet  (Tavily + NewsAPI, deduped)
    - global_news_latest.parquet    (global/macro news)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.kg.store_neo4j import Neo4jKGStore
from src.kg.update_pipeline import build_global_news_from_frame, build_news_from_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.getLogger("neo4j").setLevel(logging.WARNING)

TICKERS = ["0005.HK", "0700.HK", "1299.HK"]


def _init_store() -> Neo4jKGStore:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        raise RuntimeError("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD must be set in .env")
    store = Neo4jKGStore(uri, user, password)
    if not store.health_check():
        raise RuntimeError("Neo4j health check failed — is the DB running?")
    store.init_constraints()
    return store


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logging.warning("File not found, skipping: %s", path)
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logging.info("Empty file, skipping: %s", path)
            return None
        return df
    except Exception as exc:
        logging.error("Failed to read %s: %s", path, exc)
        return None


def import_news(data_dir: Path) -> None:
    store = _init_store()
    news_dir = data_dir / "raw" / "market_news"

    if not news_dir.exists():
        raise FileNotFoundError(f"News directory not found: {news_dir}")

    total_entities = 0
    total_evidences = 0
    total_relations = 0

    try:
        for ticker in TICKERS:
            ticker_dir = news_dir / ticker
            if not ticker_dir.exists():
                logging.warning("Ticker directory not found: %s", ticker_dir)
                continue

            logging.info("=== Processing %s ===", ticker)

            # Ensure company node exists
            from src.kg.update_pipeline import build_company_entity
            store.upsert_entities([build_company_entity(ticker)])

            # 1. Combined news (Tavily + NewsAPI, deduped)
            combined_df = _load_parquet(ticker_dir / "news_combined_latest.parquet")
            if combined_df is not None:
                logging.info("  news_combined_latest: %d rows", len(combined_df))
                entities, evidences, relations = build_news_from_frame(ticker, combined_df)
                store.upsert_entities(entities)
                store.upsert_evidences(evidences)
                store.upsert_relations(relations)
                total_entities += len(entities)
                total_evidences += len(evidences)
                total_relations += len(relations)
                logging.info("  -> %d entities, %d evidences, %d relations", len(entities), len(evidences), len(relations))

            # 2. Global/macro news
            global_df = _load_parquet(ticker_dir / "global_news_latest.parquet")
            if global_df is not None:
                logging.info("  global_news_latest: %d rows", len(global_df))
                entities, evidences, relations = build_global_news_from_frame(ticker, global_df)
                store.upsert_entities(entities)
                store.upsert_evidences(evidences)
                store.upsert_relations(relations)
                total_entities += len(entities)
                total_evidences += len(evidences)
                total_relations += len(relations)
                logging.info("  -> %d entities, %d evidences, %d relations", len(entities), len(evidences), len(relations))

            logging.info("=== Done %s ===", ticker)

        logging.info(
            "Import complete. Total: %d entities, %d evidences, %d relations",
            total_entities, total_evidences, total_relations,
        )
    finally:
        store.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Import news parquet files into Neo4j")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("C:/Users/zoezh/Downloads/data"),
        help="Root data directory (default: C:/Users/zoezh/Downloads/data)",
    )
    args = parser.parse_args()
    import_news(args.data_dir)


if __name__ == "__main__":
    main()
