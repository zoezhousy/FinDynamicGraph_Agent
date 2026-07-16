from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.grounding import (
    citation_precision,
    claim_coverage,
    compute_grounding_metrics,
    evidence_coverage,
    stale_evidence_rate,
    unsupported_claim_rate,
)


def directional_accuracy(trades: pd.DataFrame) -> float:
    if "raw_return" not in trades.columns or "action" not in trades.columns:
        raise ValueError("trades must have 'raw_return' and 'action'.")
    executed = trades[trades["trade_executed"]]
    if executed.empty:
        return 0.0
    correct = (executed["raw_return"] > 0).sum()
    return float(correct) / len(executed)


def summarize_returns(trades: pd.DataFrame) -> pd.Series:
    executed = trades[trades["trade_executed"]]
    if executed.empty:
        return pd.Series({"mean_return": 0.0, "median_return": 0.0, "n_trades": 0})
    return pd.Series(
        {
            "mean_return": executed["raw_return"].mean(),
            "median_return": executed["raw_return"].median(),
            "n_trades": len(executed),
        }
    )


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of executed trades with positive raw_return."""
    executed = trades[trades["trade_executed"]]
    if executed.empty:
        return 0.0
    return float((executed["raw_return"] > 0).sum()) / len(executed)


def abstain_rate(trades: pd.DataFrame) -> float:
    """Fraction of all decisions that were abstain."""
    if trades.empty:
        return 0.0
    return float((trades["action"] == "abstain").sum()) / len(trades)


def trade_execution_rate(trades: pd.DataFrame) -> float:
    """Fraction of all decisions that were actually executed."""
    if trades.empty:
        return 0.0
    return float(trades["trade_executed"].sum()) / len(trades)


def mean_confidence(trades: pd.DataFrame) -> float:
    if trades.empty or "confidence" not in trades.columns:
        return 0.0
    return float(trades["confidence"].mean())


def mean_conflict_level(trades: pd.DataFrame) -> float:
    if trades.empty or "conflict_level" not in trades.columns:
        return 0.0
    return float(trades["conflict_level"].mean())


def mean_fresh_evidence_count(trades: pd.DataFrame) -> float:
    if trades.empty or "fresh_evidence_count" not in trades.columns:
        return 0.0
    return float(trades["fresh_evidence_count"].mean())


def mean_stale_evidence_count(trades: pd.DataFrame) -> float:
    if trades.empty or "stale_evidence_count" not in trades.columns:
        return 0.0
    return float(trades["stale_evidence_count"].mean())


def full_summary_by_system(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute all metrics grouped by system.

    Includes trade metrics, grounding metrics, and evidence freshness.
    """
    rows = []
    for system_name, group in trades.groupby("system"):
        ret = summarize_returns(group)
        gnd = compute_grounding_metrics(group)
        row = {
            "system": system_name,
            "n_decisions": len(group),
            "n_trades": int(ret["n_trades"]),
            "trade_execution_rate": trade_execution_rate(group),
            "win_rate": win_rate(group),
            "directional_accuracy": directional_accuracy(group),
            "abstain_rate": abstain_rate(group),
            "mean_return": ret["mean_return"],
            "median_return": ret["median_return"],
            "mean_confidence": mean_confidence(group),
            "mean_conflict_level": mean_conflict_level(group),
            "mean_fresh_evidence": mean_fresh_evidence_count(group),
            "mean_stale_evidence": mean_stale_evidence_count(group),
            "evidence_coverage": gnd["evidence_coverage"],
            "citation_precision": gnd["citation_precision"],
            "claim_coverage": gnd["claim_coverage"],
            "unsupported_claim_rate": gnd["unsupported_claim_rate"],
            "stale_evidence_rate": gnd["stale_evidence_rate"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("system")
    return df


def paired_significance_tests(
    trades: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run pairwise paired-significance tests between all systems.

    Uses the same (ticker, trade_date) pairs to ensure matched samples.

    Tests:
    - **Wilcoxon signed-rank** on raw_return (paired, non-parametric)
    - **McNemar's test** on directional correctness (correct / incorrect)
      using the exact binomial test on the off-diagonal counts.

    Returns:
        DataFrame with columns:
        system_a, system_b, test, statistic, p_value, significant

        Returns an empty DataFrame if fewer than 2 systems exist.
    """
    if trades.empty or "system" not in trades.columns:
        return pd.DataFrame()

    systems = sorted(trades["system"].unique())
    if len(systems) < 2:
        return pd.DataFrame()

    # Build per-system raw_return series aligned on (ticker, trade_date)
    return_pivot = trades.pivot_table(
        index=["ticker", "trade_date"],
        columns="system",
        values="raw_return",
        aggfunc="first",
    )

    # Build per-system correctness aligned on (ticker, trade_date)
    executed = trades[trades["trade_executed"]].copy()
    if "raw_return" in executed.columns:
        executed = executed.assign(
            _correct=(executed["raw_return"] > 0).astype(int)
        )
    correct_pivot = executed.pivot_table(
        index=["ticker", "trade_date"],
        columns="system",
        values="_correct",
        aggfunc="first",
    )

    rows: list[dict] = []

    for i, sys_a in enumerate(systems):
        for sys_b in systems[i + 1:]:
            # ── Wilcoxon signed-rank on raw_return ──────────────────
            if sys_a in return_pivot.columns and sys_b in return_pivot.columns:
                pair = return_pivot[[sys_a, sys_b]].dropna()
                a_vals = pair[sys_a].values
                b_vals = pair[sys_b].values
                diffs = a_vals - b_vals
                n_pairs = len(pair)

                if n_pairs < 2 or np.all(diffs == 0):
                    p_wilcoxon = float("nan")
                    stat_wilcoxon = float("nan")
                else:
                    # zero_diff="pratt" keeps zero-diff pairs in ranking
                    # but excludes them from the signed-rank sum
                    try:
                        stat_wilcoxon, p_wilcoxon = stats.wilcoxon(
                            a_vals, b_vals, zero_method="pratt",
                        )
                    except ValueError:
                        # All non-zero diffs have same sign → cannot compute
                        stat_wilcoxon, p_wilcoxon = float("nan"), float("nan")

                rows.append({
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "test": "wilcoxon_raw_return",
                    "statistic": stat_wilcoxon,
                    "p_value": p_wilcoxon,
                    "significant": bool(p_wilcoxon < alpha) if not np.isnan(p_wilcoxon) else False,
                })

            # ── McNemar's test on directional correctness ───────────
            if sys_a in correct_pivot.columns and sys_b in correct_pivot.columns:
                cpair = correct_pivot[[sys_a, sys_b]].dropna()
                a_ok = cpair[sys_a].astype(int).values
                b_ok = cpair[sys_b].astype(int).values

                # b = a correct & b incorrect; c = a incorrect & b correct
                b_count = int(((a_ok == 1) & (b_ok == 0)).sum())
                c_count = int(((a_ok == 0) & (b_ok == 1)).sum())

                n_discordant = b_count + c_count

                if n_discordant == 0:
                    # All pairs agree → no evidence of difference
                    p_mcnemar = float("nan")
                    stat_mcnemar = float("nan")
                else:
                    # Exact binomial test: is b/(b+c) ≠ 0.5?
                    stat_mcnemar = float(b_count)
                    p_mcnemar = float(
                        stats.binomtest(b_count, n_discordant, 0.5).pvalue
                    )

                rows.append({
                    "system_a": sys_a,
                    "system_b": sys_b,
                    "test": "mcnemar_directional",
                    "statistic": stat_mcnemar,
                    "p_value": p_mcnemar,
                    "significant": bool(p_mcnemar < alpha) if not np.isnan(p_mcnemar) else False,
                })

    return pd.DataFrame(rows)
