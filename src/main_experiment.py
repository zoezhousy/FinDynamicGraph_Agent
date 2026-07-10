from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from src.eval.metrics import directional_accuracy, summarize_returns, full_summary_by_system
from src.eval.grounding import grounding_metrics_by_system
from src.kg.query import KGQueryClient
from src.sim.backtest import BacktestConfig, compute_trade_return
from src.kg.schema import BacktestOutcome
from src.kg.store_neo4j import Neo4jKGStore

def load_ohlcv_from_disk(root: Path, ticker: str) -> pd.DataFrame:
    # Try both naming conventions
    path = root / ticker / "ohlcv_2021_now.parquet"
    if not path.exists():
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

def build_backtest_outcome(
    ticker: str,
    trade_dt: datetime,
    action: str,
    kg_decision: dict,
    bt_res: dict,
    bt_cfg: BacktestConfig,
) -> BacktestOutcome:
    decision_id = kg_decision.get("decision_trace", {}).get(
        "decision_id",
        f"decision:{ticker}:{trade_dt.date().isoformat()}",
    )
    outcome_id = f"outcome:{ticker}:{trade_dt.date().isoformat()}:kg_dynamic"

    raw_return = float(bt_res.get("raw_return") or 0.0)
    trade_executed = bool(bt_res.get("trade_executed"))

    if not trade_executed:
        direction_outcome = "not_executed"
        is_profitable = None
    elif raw_return > 0:
        direction_outcome = "correct"
        is_profitable = True
    elif raw_return < 0:
        direction_outcome = "incorrect"
        is_profitable = False
    else:
        direction_outcome = "flat"
        is_profitable = False

    return BacktestOutcome(
        outcome_id=outcome_id,
        decision_id=decision_id,
        ticker=ticker,
        trade_date=trade_dt,
        action=action,
        system="kg_dynamic",
        raw_return=raw_return,
        holding_days=int(bt_res.get("holding_days") or 0),
        trade_executed=trade_executed,
        direction_outcome=direction_outcome,
        is_profitable=is_profitable,
        transaction_cost_bp=bt_cfg.transaction_cost_bp,
        metadata={
            "decision_action": action,
            "final_score": kg_decision.get("final_score"),
            "confidence": kg_decision.get("confidence"),
            "conflict_level": kg_decision.get("conflict_level"),
            "entry_date": bt_res.get("entry_date"),
            "exit_date": bt_res.get("exit_date"),
            "entry_price": bt_res.get("entry_price"),
            "exit_price": bt_res.get("exit_price"),
        },
    )

def run_experiment_for_tickers(
    tickers: List[str],
    trade_dates: List[str],
    config: CollectionConfig,
    experiment_start_date: str = "2025-01-01",
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

    kg_store = Neo4jKGStore(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        database=config.neo4j_database,
    )
    kg_store.init_constraints()

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

                outcome = build_backtest_outcome(
                    ticker=ticker,
                    trade_dt=trade_dt,
                    action=kg_decision["action"],
                    kg_decision=kg_decision,
                    bt_res=bt_res,
                    bt_cfg=bt_cfg,
                )
                kg_store.upsert_backtest_outcome(outcome)

                results_rows.append(
                    {
                        **kg_decision,
                        **bt_res,
                        "system": "kg_dynamic",
                        "outcome_id": outcome.outcome_id,
                        "direction_outcome": outcome.direction_outcome,
                        "is_profitable": outcome.is_profitable,
                    }
                )

                # --- Baseline 1: no_kg_no_evidence ---
                bl1 = baseline_no_kg_no_evidence(ticker, trade_dt)
                bl1_bt = compute_trade_return(ohlcv, trade_date_str, bl1["action"], bt_cfg)
                results_rows.append(
                    {**bl1, **bl1_bt, "system": "no_kg_no_evidence"}
                )

                # --- Baseline 2: evidence_no_kg ---
                bl2 = baseline_evidence_no_kg(ticker, trade_dt, data_root=config.output_root)
                bl2_bt = compute_trade_return(ohlcv, trade_date_str, bl2["action"], bt_cfg)
                results_rows.append(
                    {**bl2, **bl2_bt, "system": "evidence_no_kg"}
                )

                # --- Baseline 3: static_kg ---
                static_cutoff = datetime.fromisoformat(experiment_start_date)
                bl3 = baseline_static_kg(
                    ticker, trade_dt,
                    kg_context=kg_ctx,
                    static_cutoff=static_cutoff,
                )
                bl3_bt = compute_trade_return(ohlcv, trade_date_str, bl3["action"], bt_cfg)
                results_rows.append(
                    {**bl3, **bl3_bt, "system": "static_kg"}
                )
    finally:
        kg_client.close()
        kg_store.close()
        orchestrator.close()

    df = pd.DataFrame(results_rows)

    out_dir = Path("data") / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Save timestamped + latest trades ──
    parquet_path = out_dir / f"trades_{timestamp}.parquet"
    csv_path = out_dir / f"trades_{timestamp}.csv"
    latest_parquet_path = out_dir / "trades_latest.parquet"
    latest_csv_path = out_dir / "trades_latest.csv"

    df_to_save = _make_serializable(df)

    df_to_save.to_parquet(parquet_path, index=False)
    df_to_save.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_to_save.to_parquet(latest_parquet_path, index=False)
    df_to_save.to_csv(latest_csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved parquet: {parquet_path}")
    print(f"Saved csv: {csv_path}")
    print(f"Updated latest parquet: {latest_parquet_path}")
    print(f"Updated latest csv: {latest_csv_path}")

    # ── Save results summary ──
    summary = full_summary_by_system(df)
    summary_path = out_dir / "results_summary_latest.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"Saved summary: {summary_path}")

    # ── Save case study candidates ──
    candidates = _select_case_study_candidates(df)
    if not candidates.empty:
        candidates_path = out_dir / "case_study_candidates.csv"
        candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
        print(f"Saved case study candidates: {candidates_path}")

    return df


def main() -> None:
    load_dotenv()

    cfg = CollectionConfig()
    # ===== Experiment configuration =====
    experiment_start_date = "2025-01-01"
    # TODO： change experiment_end_date to 2026-05-15 after 2026-05-15
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

    df = run_experiment_for_tickers(cfg.tickers, trade_dates, cfg, experiment_start_date)

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

    # Full summary table
    summary = full_summary_by_system(df)
    print("\n=== Experiment Summary by System ===")
    print(summary.to_string())

    # Grounding metrics
    grounding = grounding_metrics_by_system(df)
    print("\n=== Grounding Metrics by System ===")
    print(grounding.to_string())


# ── Case Study Candidate Selection ─────────────────────────────────────


def _safe_list_len(val: Any) -> int:
    """Get length of a list that might be a string-encoded list."""
    if isinstance(val, list):
        return len(val)
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return len(parsed)
            except Exception:
                pass
        if val and val not in ("[]", "None", "nan", ""):
            return 1
    return 0


def _select_case_study_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Select interesting decisions for dissertation case studies.

    Selection criteria (any match):
    1. kg_dynamic abstains but baseline buys/sells (conflict-driven abstain)
    2. High conflict_level (>= 0.5)
    3. High stale evidence in static_kg
    4. kg_dynamic has more fresh evidence than static_kg
    5. Decision has clear evidence_ids and claim_ids
    6. Outcome correct while baseline incorrect
    """
    if df.empty:
        return pd.DataFrame()

    candidates = []
    systems = df["system"].unique()

    # Group by (ticker, trade_date) for cross-system comparison
    grouped = df.groupby(["ticker", "trade_date"])

    for (ticker, trade_date), group in grouped:
        kg_dyn = group[group["system"] == "kg_dynamic"]
        if kg_dyn.empty:
            continue

        kg_row = kg_dyn.iloc[0]
        kg_action = str(kg_row.get("action", "abstain"))
        kg_conflict = float(kg_row.get("conflict_level", 0) or 0)
        kg_ev_refs = _safe_list_len(kg_row.get("evidence_refs"))
        kg_claim_refs = _safe_list_len(kg_row.get("claim_refs"))
        kg_fresh = int(kg_row.get("fresh_evidence_count", 0) or 0)
        kg_stale = int(kg_row.get("stale_evidence_count", 0) or 0)

        reasons = []

        # Criterion 1: kg_dynamic abstains but baselines act
        for sys_name in ("no_kg_no_evidence", "evidence_no_kg", "static_kg"):
            bl = group[group["system"] == sys_name]
            if bl.empty:
                continue
            bl_action = str(bl.iloc[0].get("action", "abstain"))
            if kg_action == "abstain" and bl_action in ("buy", "sell"):
                reasons.append(f"kg_dynamic abstains but {sys_name}={bl_action}")

        # Criterion 2: High conflict
        if kg_conflict >= 0.5:
            reasons.append(f"high conflict_level={kg_conflict:.2f}")

        # Criterion 3: High stale evidence in static_kg
        static_kg = group[group["system"] == "static_kg"]
        if not static_kg.empty:
            static_stale = int(static_kg.iloc[0].get("stale_evidence_count", 0) or 0)
            if static_stale > 3:
                reasons.append(f"static_kg stale_evidence={static_stale}")

        # Criterion 4: kg_dynamic has more fresh evidence
        if not static_kg.empty:
            static_fresh = int(static_kg.iloc[0].get("fresh_evidence_count", 0) or 0)
            if kg_fresh > static_fresh and kg_fresh > 0:
                reasons.append(f"kg_dynamic fresh={kg_fresh} > static_kg fresh={static_fresh}")

        # Criterion 5: Clear evidence and claims
        if kg_ev_refs > 0 and kg_claim_refs > 0:
            reasons.append(f"well-grounded: {kg_ev_refs} evidence, {kg_claim_refs} claims")

        # Criterion 6: Outcome correctness
        kg_outcome = kg_row.get("direction_outcome")
        if kg_outcome == "correct":
            # Check if any baseline was incorrect
            for sys_name in ("no_kg_no_evidence", "evidence_no_kg", "static_kg"):
                bl = group[group["system"] == sys_name]
                if bl.empty:
                    continue
                bl_outcome = bl.iloc[0].get("direction_outcome")
                if bl_outcome == "incorrect":
                    reasons.append(f"kg_dynamic correct but {sys_name} incorrect")

        if reasons:
            candidates.append({
                "ticker": ticker,
                "trade_date": trade_date,
                "system": "kg_dynamic",
                "action": kg_action,
                "final_score": kg_row.get("final_score"),
                "confidence": kg_row.get("confidence"),
                "conflict_level": kg_conflict,
                "evidence_count": kg_ev_refs,
                "claim_count": kg_claim_refs,
                "fresh_evidence_count": kg_fresh,
                "stale_evidence_count": kg_stale,
                "reason": str(kg_row.get("decision_reason", ""))[:200],
                "why_candidate": "; ".join(reasons),
            })

    return pd.DataFrame(candidates)


def _make_serializable(df: pd.DataFrame) -> pd.DataFrame:
    """Make DataFrame safe for parquet/csv by converting list/dict columns to strings."""
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_out[col] = df_out[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
    return df_out


if __name__ == "__main__":
    main()
