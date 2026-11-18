"""Pure case selection and legacy-compatible result export."""
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
from src.eval.metrics import full_summary_by_system, paired_significance_tests, _check_evaluation_version

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

    Returns candidates with a ``case_group`` column:

    * **mechanism** — chosen on process signals only (conflict, evidence
      freshness, graph completeness); ``direction_outcome`` is *not* inspected.
    * **outcome_best** — kg_dynamic achieved the best return among all systems
      on this date (at least one per run).
    * **outcome_worst** — kg_dynamic achieved the worst return among all
      systems on this date (at least one per run).
    """
    if df.empty:
        return pd.DataFrame()

    mechanism_rows: list[dict] = []
    outcome_scores: list[tuple[float, dict]] = []  # (score, row_dict)

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
        kg_return = float(kg_row.get("raw_return", 0) or 0)

        reasons = []

        # ── Mechanism criteria (no outcome inspection) ────────────────

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

        base_row = {
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
        }

        if reasons:
            mechanism_rows.append({**base_row, "case_group": "mechanism", "why_candidate": "; ".join(reasons)})

        # ── Outcome scoring (deferred selection) ──────────────────────
        # Score = return advantage of kg_dynamic over the best baseline.
        # Higher is better for best-case; lower is worse for worst-case.
        baseline_returns = []
        for sys_name in ("no_kg_no_evidence", "evidence_no_kg", "static_kg"):
            bl = group[group["system"] == sys_name]
            if bl.empty:
                continue
            baseline_returns.append(float(bl.iloc[0].get("raw_return", 0) or 0))

        best_baseline = max(baseline_returns) if baseline_returns else 0.0
        worst_baseline = min(baseline_returns) if baseline_returns else 0.0

        # best-case score: how much kg_dynamic beats the best baseline
        best_score = kg_return - best_baseline
        # worst-case score: how much kg_dynamic trails the worst baseline
        worst_score = kg_return - worst_baseline

        outcome_scores.append((best_score, worst_score, {**base_row}))

    # ── Assemble final DataFrame ─────────────────────────────────────

    rows = list(mechanism_rows)

    if outcome_scores:
        # Best-case: highest (kg_return - best_baseline)
        best_idx = max(range(len(outcome_scores)), key=lambda i: outcome_scores[i][0])
        best_row = outcome_scores[best_idx][2]
        best_adv = outcome_scores[best_idx][0]
        rows.append({
            **best_row,
            "case_group": "outcome_best",
            "why_candidate": f"best return advantage over baselines: {best_adv:+.4f}",
        })

        # Worst-case: lowest (kg_return - worst_baseline)
        worst_idx = min(range(len(outcome_scores)), key=lambda i: outcome_scores[i][1])
        worst_row = outcome_scores[worst_idx][2]
        worst_gap = outcome_scores[worst_idx][1]
        rows.append({
            **worst_row,
            "case_group": "outcome_worst",
            "why_candidate": f"worst return gap vs baselines: {worst_gap:+.4f}",
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Ensure consistent column order
    col_order = [
        "case_group", "ticker", "trade_date", "system", "action",
        "final_score", "confidence", "conflict_level",
        "evidence_count", "claim_count", "fresh_evidence_count",
        "stale_evidence_count", "reason", "why_candidate",
    ]
    result = result[[c for c in col_order if c in result.columns]]
    return result


def _make_serializable(df: pd.DataFrame) -> pd.DataFrame:
    """Make DataFrame safe for parquet/csv by converting list/dict columns to strings."""
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_out[col] = df_out[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
    return df_out



def export_results(df: pd.DataFrame, out_dir: Path, *, metrics_df=None, evaluation_version="legacy") -> None:
    metric_trades = df if metrics_df is None else metrics_df
    _check_evaluation_version(df, evaluation_version)
    _check_evaluation_version(metric_trades, evaluation_version)
    out_dir = Path(out_dir)
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
    summary = full_summary_by_system(metric_trades, evaluation_version=evaluation_version) if not metric_trades.empty else pd.DataFrame()
    summary_path = out_dir / "results_summary_latest.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"Saved summary: {summary_path}")

    # ── Save case study candidates ──
    candidates = _select_case_study_candidates(metric_trades)
    if not candidates.empty:
        candidates_path = out_dir / "case_study_candidates.csv"
        candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
        print(f"Saved case study candidates: {candidates_path}")

    # ── Save paired significance tests ──
    sig = paired_significance_tests(metric_trades, evaluation_version=evaluation_version)
    if not sig.empty:
        sig_path = out_dir / "significance_tests.csv"
        sig.to_csv(sig_path, index=False, encoding="utf-8-sig")
        print(f"Saved significance tests: {sig_path}")

