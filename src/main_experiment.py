from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv

from src.agents.kg_tools import KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator
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
) -> pd.DataFrame:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not (neo4j_uri and neo4j_user and neo4j_password):
        raise RuntimeError("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD must be set.")

    kg_client = KGQueryClient(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        database=config.neo4j_database,
    )
    kg_ctx = KGAgentContext(kg_client)
    orchestrator = KGBasedOrchestrator(kg_ctx)

    bt_cfg = BacktestConfig()
    results_rows = []

    try:
        for ticker in tickers:
            ohlcv = load_ohlcv_from_disk(config.output_root, ticker)
            for trade_date_str in trade_dates:
                trade_dt = datetime.fromisoformat(trade_date_str)

                kg_decision = orchestrator.run_for_ticker(ticker, trade_dt)
                bt_res = compute_trade_return(ohlcv, trade_date_str, kg_decision["action"], bt_cfg)
                results_rows.append({**kg_decision, **bt_res, "system": "kg_dynamic"})

                for baseline_fn in (
                    baseline_no_kg_no_evidence,
                    baseline_evidence_no_kg,
                    baseline_static_kg,
                ):
                    baseline_decision = baseline_fn(ticker, trade_dt)
                    baseline_bt = compute_trade_return(ohlcv, trade_date_str, baseline_decision["action"], bt_cfg)
                    results_rows.append(
                        {**baseline_decision, **baseline_bt, "system": baseline_decision["baseline"]}
                    )
    finally:
        kg_client.close()

    df = pd.DataFrame(results_rows)
    out_dir = Path("data") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "trades.parquet", index=False)
    return df


def main() -> None:
    load_dotenv()
    cfg = CollectionConfig()
    trade_dates = ["2023-01-10", "2023-03-15", "2023-06-20"]
    df = run_experiment_for_tickers(cfg.tickers, trade_dates, cfg)

    summary = df.groupby("system").apply(summarize_returns)
    da = df.groupby("system").apply(directional_accuracy)
    print("Return summary by system:")
    print(summary)
    print("Directional accuracy by system:")
    print(da)


if __name__ == "__main__":
    main()
