from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CollectionConfig:
    tickers: List[str] = field(
        default_factory=lambda: ["0005.HK", "0700.HK", "1299.HK"]
    )
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    news_limit_per_ticker: int = 10
    output_root: Path = Path("data/raw/market_news")
    log_file: Path = Path("data/logs/data_collection.log")

    # Retry and rate limit controls
    max_retries: int = 4
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 16.0
    request_interval_seconds: float = 1.0

    # Runtime behavior
    market_data_provider: str = "yahoo"
    neo4j_database: str = "neo4j"
