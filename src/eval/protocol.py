"""Explicit point-in-time policy; legacy query helpers stay untouched."""
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
import pandas as pd

HK = timezone(timedelta(hours=8))

def decision_cutoff(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, time.min)
    if value.tzinfo is None:
        if value.time() == time.min:
            value = datetime.combine(value.date(), time.max)
        value = value.replace(tzinfo=HK)
    return value.astimezone(timezone.utc)

def available_time(value):
    if value is None or (isinstance(value, str) and not value):
        return None
    try:
        result = pd.Timestamp(value)
        if pd.isna(result):
            return None
        if result.tzinfo is None:
            result = result.tz_localize(HK)
        return result.tz_convert("UTC").to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None

AVAILABILITY = ("effective_at", "published_at", "published_time", "as_of_date", "valid_from")

def eligible_record(row, cutoff, *, strict=False):
    dates = [available_time(row.get(k)) for k in AVAILABILITY if row.get(k) is not None]
    # Unknown or unparseable availability is never historical evidence.
    if not dates or any(d is None or d > cutoff or (strict and d == cutoff) for d in dates):
        return False
    expiry = available_time(row.get("valid_to"))
    if row.get("valid_to") is not None and not (isinstance(row.get("valid_to"), str) and not row["valid_to"]) and expiry is None:
        return False
    return expiry is None or expiry > cutoff

def strict_snapshot(snapshot, cutoff):
    result = deepcopy(snapshot)
    for key in ("signals", "fundamentals", "risks", "news", "evidences", "claims", "sources"):
        if key in result:
            result[key] = [dict(row) for row in result[key] or [] if eligible_record(row, cutoff)]
            for row in result[key]:
                for field in (*AVAILABILITY, "valid_to"):
                    if row.get(field) is not None:
                        parsed = available_time(row[field])
                        if parsed is not None:
                            row[field] = parsed.isoformat()
    # Conflict edges inherit availability from both endpoints, not their own timestamp.
    retained = {str(row.get("claim_id") or row.get("entity_id"))
                for key in ("claims", "signals", "fundamentals", "risks")
                for row in result.get(key, []) if row.get("claim_id") or row.get("entity_id")}
    if "conflicts" in result:
        result["conflicts"] = [dict(edge) for edge in result["conflicts"] or []
                               if edge.get("claim_a") in retained and edge.get("claim_b") in retained]
    return result

class FrozenSnapshotClient:
    """Owns data only. The caller retains ownership of the live query client."""
    def __init__(self, ticker, cutoff, snapshot, *, request_date=None):
        self.ticker = ticker
        self.cutoff = cutoff
        self.snapshot = deepcopy(snapshot)
        self.request_date = request_date or cutoff
        self.violations = []

    def load_subgraph(self, ticker, as_of_date, **kwargs):
        accepted = {self.cutoff, self.request_date}
        if ticker != self.ticker or as_of_date not in accepted:
            from src.eval.execution import note_issue
            self.violations.append({"ticker": ticker, "date": str(as_of_date)})
            note_issue("tool_boundary", reason="request_outside_fixed_snapshot")
            raise ValueError("Request is outside the fixed ticker/date boundary")
        return deepcopy(self.snapshot)

    def close(self):
        pass

def backtest_details(ohlcv, trade_date, action, cfg, *, protocol_version="legacy"):
    from src.sim.backtest import compute_trade_return
    result = compute_trade_return(ohlcv, trade_date, action, cfg)
    executed = bool(result.get("trade_executed"))
    complete = not executed or result.get("holding_days", 0) == cfg.holding_days
    settled = complete and (not executed or action in {"buy", "sell"})
    cost = 2 * cfg.transaction_cost_bp / 10000 if executed else 0.0
    result.update(gross_return=float(result.get("raw_return", 0)) + cost, transaction_cost=cost, window_complete=complete, settled=settled)
    if protocol_version == "pit_v1" and not complete:
        result.update(raw_return=None, gross_return=None, settled=False)
    if action in {"buy", "sell"} and not executed:
        result["settled"] = False
    return result
