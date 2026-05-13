from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd


@dataclass
class BacktestConfig:
    holding_days: int = 5
    transaction_cost_bp: float = 5.0  # basis points per trade


def compute_trade_return(
    ohlcv: pd.DataFrame,
    trade_date: str,
    action: str,
    cfg: BacktestConfig,
) -> Dict[str, float | int | str]:
    """Next-period execution: enter at next day's open, exit after holding_days at close."""
    if "date" not in ohlcv.columns or "open" not in ohlcv.columns or "close" not in ohlcv.columns:
        raise ValueError("OHLCV must contain 'date', 'open', 'close' columns.")

    frame = ohlcv.sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    t0 = pd.to_datetime(trade_date)
    idx_candidates = frame.index[frame["date"] > t0]
    if len(idx_candidates) == 0:
        return {"raw_return": 0.0, "holding_days": 0, "trade_executed": False}

    entry_idx = int(idx_candidates[0])
    exit_idx = min(entry_idx + cfg.holding_days - 1, len(frame) - 1)
    entry_price = float(frame.loc[entry_idx, "open"])
    exit_price = float(frame.loc[exit_idx, "close"])

    if action == "buy":
        gross = (exit_price - entry_price) / entry_price
    elif action == "sell":
        gross = (entry_price - exit_price) / entry_price
    else:
        return {
            "raw_return": 0.0,
            "holding_days": 0,
            "trade_executed": False,
            "entry_date": None,
            "exit_date": None,
            "entry_price": None,
            "exit_price": None,
        }

    cost = 2 * cfg.transaction_cost_bp / 10000.0
    net = gross - cost
    return {
        "raw_return": net,
        "holding_days": int(exit_idx - entry_idx + 1),
        "trade_executed": True,
        "entry_date": frame.loc[entry_idx, "date"].date().isoformat(),
        "exit_date": frame.loc[exit_idx, "date"].date().isoformat(),
        "entry_price": entry_price,
        "exit_price": exit_price,
    }

