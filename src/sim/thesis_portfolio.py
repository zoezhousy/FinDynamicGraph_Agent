"""Calendar-based five-session holdings and equal-weight long/short portfolios."""
from bisect import bisect_right
from datetime import date
from math import isfinite

HOLDING_SESSIONS = 5
COST_BP = 5
TAIL_PERCENT = 35


class MarketTape:
    def __init__(self, payload):
        self.sessions = sorted(date.fromisoformat(day).isoformat() for day in payload["sessions"])
        if not self.sessions or len(set(self.sessions)) != len(self.sessions):
            raise ValueError("A non-empty, unique exchange session calendar is required")
        sessions = set(self.sessions)
        self.prices = {}
        for row in payload["prices"]:
            day = date.fromisoformat(row["date"]).isoformat()
            key = (row["ticker"], day)
            if day not in sessions or key in self.prices:
                raise ValueError("Market row is off-calendar or duplicated")
            for field in ("open", "close"):
                value = row.get(field)
                if value is not None and (type(value) not in (float, int) or not isfinite(value) or value <= 0):
                    raise ValueError("Market prices must be finite positive numbers or null")
            self.prices[key] = {"open": row.get("open"), "close": row.get("close")}
        self.tickers = {ticker for ticker, _ in self.prices}

    def window(self, decision_date):
        index = bisect_right(self.sessions, decision_date)
        return self.sessions[index:index + HOLDING_SESSIONS]

    def marks(self, ticker, window):
        entry = self.prices.get((ticker, window[0]), {}).get("open") if window else None
        exit_price = (self.prices.get((ticker, window[-1]), {}).get("close")
                      if len(window) == HOLDING_SESSIONS else None)
        return entry, exit_price


def settle_portfolios(tasks, tickers, dates, market, mode="independent"):
    if mode not in {"independent", "rolling"}:
        raise ValueError("Invalid portfolio mode")
    rows, active = [], []
    for day in sorted(dates):
        daily = {ticker: tasks.get((ticker, day), {"status": "failed"}) for ticker in tickers}
        failed = sorted(t for t, row in daily.items() if row["status"] != "success")
        ranked = sorted(((t, row["score"]) for t, row in daily.items()
                         if row["status"] == "success" and not row["gate"]),
                        key=lambda item: (-item[1], item[0]))
        count = len(ranked) * TAIL_PERCENT // 100
        row = {"trade_date": day, "candidate_count": len(ranked), "ranked": ranked,
               "gates": {t: r["gate"] for t, r in daily.items() if r["status"] == "success" and r["gate"]},
               "failed_tickers": failed, "status": "no_trade", "positions": [], "net_return": None}
        rows.append(row)
        if failed:
            row["status"] = "selection_failed"
            continue
        if not count:
            continue
        window = market.window(day)
        first = window[0] if window else None
        last = window[-1] if len(window) == HOLDING_SESSIONS else None
        if first is not None:
            active = [end for end in active if end is None or end >= first]
        if mode == "rolling" and len(active) >= HOLDING_SESSIONS:
            row["status"] = "capacity_skipped"
            continue
        fraction = 1 / HOLDING_SESSIONS if mode == "rolling" else 1.0
        row.update(entry_date=first, exit_date=last, gross_target=fraction, status="unsettled")
        for side, selection in ((1, ranked[:count]), (-1, ranked[-count:])):
            for ticker, _ in selection:
                weight = side * fraction / (2 * count)
                entry, exit_price = market.marks(ticker, window)
                position = {"ticker": ticker, "target_weight": weight, "entry_price": entry,
                            "exit_price": exit_price, "gross_pnl": None, "cost": None, "net_pnl": None}
                row["positions"].append(position)
                if entry is not None and exit_price is not None:
                    gross = weight * (exit_price / entry - 1)
                    cost = abs(weight) * 2 * COST_BP / 10000
                    position.update(gross_pnl=gross, cost=cost, net_pnl=gross - cost)
        entered = all(p["entry_price"] is not None for p in row["positions"])
        settled = all(p["net_pnl"] is not None for p in row["positions"])
        if not entered:
            row["unsettled_reason"] = "missing_entry"
            for position in row["positions"]:
                position.update(gross_pnl=None, cost=None, net_pnl=None)
        elif settled:
            row.update(status="settled", net_return=sum(p["net_pnl"] for p in row["positions"]))
        else:
            row["unsettled_reason"] = "incomplete_holding_period" if last is None else "missing_exit"
        for position in row["positions"]:
            position["executed_entry"] = entered
            position["executed_exit"] = entered and position["exit_price"] is not None
            charge = abs(position["target_weight"]) * COST_BP / 10000
            position["entry_cost"] = charge if entered else 0.0
            position["exit_cost"] = charge if position["executed_exit"] else 0.0
        if mode == "rolling" and entered:
            active.append(last if settled else None)
    return rows
