from __future__ import annotations

from typing import Iterable, List

import pandas as pd


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
    """Compute all metrics grouped by system."""
    rows = []
    for system_name, group in trades.groupby("system"):
        ret = summarize_returns(group)
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
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("system")
    return df

