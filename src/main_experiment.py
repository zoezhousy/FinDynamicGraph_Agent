from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from src.agents.kg_tools import KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator
from src.collectors.market_collector import MarketCollector
from src.config import CollectionConfig
from src.eval.baselines import (
    baseline_evidence_no_kg,
    baseline_no_kg_no_evidence,
    baseline_static_kg,
)
from src.eval.metrics import directional_accuracy, summarize_returns
from src.kg.query import KGQueryClient
from src.sim.backtest import BacktestConfig, compute_trade_return


def load_ohlcv_from_disk(root: Path, ticker: str) -> pd.DataFrame:
    path = root / ticker / "ohlcv_2021_2025.parquet"
    return pd.read_parquet(path)


def run_experiment_for_tickers(
    tickers: List[str],
    trade_dates: List[str],
    config: CollectionConfig,
) -> None:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not (neo4j_uri and neo4j_user and neo4j_password):
        raise RuntimeError("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD must be set.")

    kg_client = KGQueryClient(neo4j_uri, neo4j_user, neo4j_password)
    kg_ctx = KGAgentContext(kg_client)
    orchestrator = KGBasedOrchestrator(kg_ctx)

    bt_cfg = BacktestConfig()
    results_rows = []

    for ticker in tickers:
        ohlcv = load_ohlcv_from_disk(config.output_root, ticker)
        for d in trade_dates:
            trade_dt = datetime.fromisoformat(d)

            # Our KG-based system
            kg_decision = orchestrator.run_for_ticker(ticker, trade_dt)
            bt_res = compute_trade_return(ohlcv, d, kg_decision["action"], bt_cfg)
            row = {**kg_decision, **bt_res, "system": "kg_dynamic"}
            results_rows.append(row)

            # Baselines
            for fn in (
                baseline_no_kg_no_evidence,
                baseline_evidence_no_kg,
                baseline_static_kg,
            ):
                b_decision = fn(ticker, trade_dt)
                b_bt = compute_trade_return(ohlcv, d, b_decision["action"], bt_cfg)
                row = {**b_decision, **b_bt, "system": b_decision["baseline"]}
                results_rows.append(row)

    df = pd.DataFrame(results_rows)
    out_dir = Path("data") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "trades.parquet", index=False)

    # 简单汇总
    summary = df.groupby("system").apply(summarize_returns)
    da = df.groupby("system").apply(directional_accuracy)
    print("Return summary by system:")
    print(summary)
    print("Directional accuracy by system:")
    print(da)


def main() -> None:
    cfg = CollectionConfig()
    # 示例：每只股票选几个交易日
    trade_dates = ["2023-01-10", "2023-03-15", "2023-06-20"]
    run_experiment_for_tickers(cfg.tickers, trade_dates, cfg)


if __name__ == "__main__":
    main()

