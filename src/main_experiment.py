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

def generate_trade_dates(
    start_date: str,
    end_date: str,
    mode: str = "weekly",
) -> list[str]:
    """
    mode:
    - daily -> every business day
    - weekly -> every Wednesday
    - monthly -> first calendar day of each month
    """
    if mode == "daily":
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
    elif mode == "weekly":
        dates = pd.date_range(start=start_date, end=end_date, freq="W-WED")
    elif mode == "monthly":
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return [d.strftime("%Y-%m-%d") for d in dates]

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

                # KG dynamic system
                kg_decision = orchestrator.run_for_ticker(ticker, trade_dt)
                bt_res = compute_trade_return(ohlcv, trade_date_str, kg_decision["action"], bt_cfg)
                results_rows.append({**kg_decision, **bt_res, "system": "kg_dynamic"})

                # Baselines
                for baseline_fn in (
                    baseline_no_kg_no_evidence,
                    baseline_evidence_no_kg,
                    baseline_static_kg,
                ):
                    baseline_decision = baseline_fn(ticker, trade_dt)
                    baseline_bt = compute_trade_return(
                        ohlcv,
                        trade_date_str,
                        baseline_decision["action"],
                        bt_cfg,
                    )
                    results_rows.append(
                        {**baseline_decision, **baseline_bt, "system": baseline_decision["baseline"]}
                    )
    finally:
        kg_client.close()

    df = pd.DataFrame(results_rows)

    out_dir = Path("data") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    # df.to_parquet(out_dir / "trades.parquet", index=False)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parquet_path = out_dir / f"trades_{timestamp}.parquet"
    csv_path = out_dir / f"trades_{timestamp}.csv"

    latest_parquet_path = out_dir / "trades_latest.parquet"
    latest_csv_path = out_dir / "trades_latest.csv"

    df_to_save = df.copy()

    for col in df_to_save.columns:
        if df_to_save[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_to_save[col] = df_to_save[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )

    df_to_save.to_parquet(parquet_path, index=False)
    df_to_save.to_csv(csv_path, index=False, encoding="utf-8-sig")

    df_to_save.to_parquet(latest_parquet_path, index=False)
    df_to_save.to_csv(latest_csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved parquet: {parquet_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Updated latest parquet: {latest_parquet_path}")
    print(f"Updated latest csv: {latest_csv_path}")


    return df

def main() -> None:
    load_dotenv()

    cfg = CollectionConfig()
    # ===== Experiment configuration =====
    experiment_start_date = "2025-01-01"
    experiment_end_date = "2026-04-30"
    experiment_mode = "monthly" # daily / weekly / monthly
    # ====================================

    trade_dates = generate_trade_dates(
        start_date=experiment_start_date,
        end_date=experiment_end_date,
        mode=experiment_mode,
    )

    print(f"Experiment mode: {experiment_mode}")
    print(f"Trade dates range: {experiment_start_date} -> {experiment_end_date}")
    print(f"Number of trade dates: {len(trade_dates)}")
    print("Sample trade dates:", trade_dates[:10])

    df = run_experiment_for_tickers(cfg.tickers, trade_dates, cfg)

    # Debug output
    print(df[["system", "ticker", "trade_date", "action", "trade_executed", "raw_return"]].head(50))
    print("\nAction distribution:")
    print(df.groupby(["system", "action"]).size())

    cols_for_decision_debug = [
        c for c in ["system", "decision_reason", "confidence", "conflict_level", "evidence_refs"]
        if c in df.columns
    ]
    if cols_for_decision_debug:
        print("\nDecision debug sample:")
        print(df[cols_for_decision_debug].head(20))

    # Summary without pandas FutureWarning
    summary_rows = []
    da_rows = []

    for system_name, group in df.groupby("system"):
        summary_series = summarize_returns(group)
        summary_series.name = system_name
        summary_rows.append(summary_series)

        da_rows.append(
            {
                "system": system_name,
                "directional_accuracy": directional_accuracy(group),
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.index.name = "system"

    da = pd.DataFrame(da_rows).set_index("system") if da_rows else pd.DataFrame()

    print("\nReturn summary by system:")
    print(summary)

    print("\nDirectional accuracy by system:")
    print(da)

if __name__ == "__main__":
    main()
