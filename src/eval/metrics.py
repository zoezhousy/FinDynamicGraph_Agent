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

