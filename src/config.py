from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CollectionConfig:

    # data collection settings
    # tickers for simple implementation in the early stage
    # TODO: get tickers from API or user's input or external watchlist file
    tickers: List[str] = field(
        default_factory=lambda: ["0005.HK", "0700.HK", "1299.HK"]
    )
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"
    news_limit_per_ticker: int = 10
    output_root: Path = Path("data/raw/market_news")
    log_file: Path = Path("data/logs/data_collection.log")

    # retry settings for data collection
    max_retries: int = 4
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 16.0
    request_interval_seconds: float = 1.0


    # market data & graph database settings
    market_data_provider: str = "yahoo"
    neo4j_database: str = "neo4j"

    # do not destroy historical decision traces by default.
    reset_graph_before_collection: bool = False
    preserve_historical_traces: bool = True

    # Fundamental data collection.
    collect_fundamentals: bool = True

    # llm api setting
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model: str = ""