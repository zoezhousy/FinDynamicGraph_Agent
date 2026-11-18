from dataclasses import dataclass, field
from datetime import date
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
    end_date: str = field(default_factory=lambda: date.today().isoformat())
    news_limit_per_ticker: int = 10
    output_root: Path = Path("data/raw/market_news")
    report_root: Path = Path("data/reports")
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

@dataclass(frozen=True)
class ExperimentConfig:
    protocol_version: str = "legacy"
    output_dir: Path = Path("data/experiments")
    run_id: str | None = None
    resume: bool = False
    holding_days: int = 5
    transaction_cost_bp: float = 5.0
    evaluation_version: str = "legacy"

    def __post_init__(self):
        if self.protocol_version not in {"legacy", "pit_v1"}:
            raise ValueError("protocol_version must be legacy or pit_v1")
        if self.evaluation_version not in {"legacy", "v2"}:
            raise ValueError("evaluation_version must be legacy or v2")
        if self.evaluation_version == "v2" and self.protocol_version != "pit_v1":
            raise ValueError("evaluation v2 requires pit_v1")
        if self.holding_days < 1 or self.transaction_cost_bp < 0:
            raise ValueError("Invalid backtest configuration")
        if self.resume and not self.run_id:
            raise ValueError("resume requires run_id")
