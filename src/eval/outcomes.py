from datetime import datetime
from src.kg.schema import BacktestOutcome
from src.sim.backtest import BacktestConfig

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

