from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from src.agents.kg_tools import KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator
from src.config import CollectionConfig
from src.eval.baselines import (
    baseline_evidence_no_kg,
    baseline_no_kg_no_evidence,
    baseline_static_kg,
)
from src.eval.metrics import directional_accuracy, summarize_returns, full_summary_by_system, paired_significance_tests
from src.eval.grounding import grounding_metrics_by_system
from src.kg.query import KGQueryClient
from src.sim.backtest import BacktestConfig, compute_trade_return
from src.kg.schema import BacktestOutcome
from src.kg.store_neo4j import Neo4jKGStore

def load_ohlcv_from_disk(root: Path, ticker: str) -> pd.DataFrame:
    from src.collectors.cache import load_ohlcv
    return load_ohlcv(root, ticker)


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

from src.eval.outcomes import build_backtest_outcome


def run_experiment_for_tickers(
    tickers: List[str], trade_dates: List[str], config: CollectionConfig,
    experiment_start_date: str = "2025-01-01", *, experiment_config=None,
    query_client=None, store=None, orchestrator=None, model_client=None, llm=None,
    runners=None, ohlcv_loader=None,
) -> pd.DataFrame:
    from src.eval.runner import run_grid
    return run_grid(tickers, trade_dates, config, experiment_start_date,
        experiment_config=experiment_config, query_client=query_client, store=store,
        orchestrator=orchestrator, model_client=model_client, llm=llm, runners=runners,
        ohlcv_loader=ohlcv_loader or load_ohlcv_from_disk,
        query_factory=KGQueryClient, store_factory=Neo4jKGStore,
        orchestrator_factory=KGBasedOrchestrator)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    load_dotenv()

    import argparse
    from src.config import ExperimentConfig
    parser = argparse.ArgumentParser(description="Run the four-system experiment")
    parser.add_argument("--protocol", choices=("legacy", "pit_v1"), default="legacy")
    parser.add_argument("--evaluation-version", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments"))
    args = parser.parse_args()
    experiment_config = ExperimentConfig(protocol_version=args.protocol, evaluation_version=args.evaluation_version, output_dir=args.output_dir, run_id=args.run_id, resume=args.resume)
    cfg = CollectionConfig()
    # ===== Experiment configuration =====
    experiment_start_date = "2025-01-01"
    experiment_end_date = datetime.now().strftime("%Y-%m-%d")
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

    total_iterations = len(cfg.tickers) * len(trade_dates)
    logger.info(
        "Starting experiment: %d tickers × %d dates = %d iterations (4 systems each)",
        len(cfg.tickers), len(trade_dates), total_iterations,
    )
    logger.info("LLM calls will be retried up to 3 times on timeout/failure.")
    logger.info("Progress will be logged per iteration to stderr.")

    df = run_experiment_for_tickers(cfg.tickers, trade_dates, cfg, experiment_start_date, experiment_config=experiment_config)

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

    # Use the same eligibility policy as exported new-protocol reports.
    from src.eval.runner import comparable_rows
    metric_df = comparable_rows(df) if args.protocol == "pit_v1" else df
    if metric_df.empty:
        print("No successful, settled tasks available for summary.")
        return
    # Full summary table
    summary = full_summary_by_system(metric_df, evaluation_version=args.evaluation_version)
    print("\n=== Experiment Summary by System ===")
    print(summary.to_string())

    # Grounding metrics
    grounding = grounding_metrics_by_system(metric_df, evaluation_version=args.evaluation_version)
    print("\n=== Grounding Metrics by System ===")
    print(grounding.to_string())

    # Paired significance tests
    sig = paired_significance_tests(metric_df, evaluation_version=args.evaluation_version)
    if not sig.empty:
        print("\n=== Paired Significance Tests (alpha=0.05) ===")
        print(sig.to_string(index=False))


# ── Case Study Candidate Selection ─────────────────────────────────────


from src.eval.reporting import _safe_list_len, _select_case_study_candidates, _make_serializable


if __name__ == "__main__":
    main()
