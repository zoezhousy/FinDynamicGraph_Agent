"""Deterministic research decision, semantic score and backtest replay."""
from datetime import datetime
import pandas as pd
from src.agents.models import AgentReport
from src.agents.pm_deterministic import portfolio_manager_decide
from src.eval.protocol import backtest_details
from src.eval.records import digest
from src.eval.research import risk_conditions, apply_risk_policy, VARIANTS
from src.eval.semantic_grounding import JudgeResponse, score_judgment
from src.sim.backtest import BacktestConfig


def replay_research_record(record):
    inputs = record["inputs"]
    if digest(inputs) != record["input_digest"]:
        raise ValueError("Saved input digest does not match")
    snapshot = record["decision_snapshot"]
    if digest(snapshot) != record["decision_snapshot_digest"] or digest(snapshot) != inputs["snapshot_digest"]:
        raise ValueError("Saved research snapshot digest does not match")
    if digest(inputs["corpus"]) != inputs["corpus_digest"] or inputs["variant"] != VARIANTS[record["system"]]:
        raise ValueError("Saved research corpus or variant identity differs")
    when = datetime.fromisoformat(inputs["decision_time"])
    reports = [AgentReport(**r) for r in record["decision"]["agent_reports"]]
    decision = portfolio_manager_decide(record["ticker"], when, reports, snapshot, evaluation_version="v2")
    decision = apply_risk_policy(decision, risk_conditions(snapshot, when), inputs["risk_policy"])
    if digest(decision) != digest(record["decision"]):
        raise ValueError("Saved research decision differs from deterministic aggregation")
    response = JudgeResponse.model_validate(record["row"]["semantic_grounding"]["response"])
    semantic = score_judgment(response, inputs["corpus"], sorted(set(decision["evidence_refs"])),
                              key_evidence_ids=inputs.get("key_evidence_ids"))
    for field, value in semantic.items():
        if digest(value) != digest(record["row"]["semantic_grounding"][field]):
            raise ValueError("Saved semantic score differs from quote-validated replay")
    backtest = backtest_details(pd.DataFrame(inputs["ohlcv"]), record["trade_date"], decision["action"],
                               BacktestConfig(**inputs["backtest_config"]), protocol_version="pit_v1")
    if digest(backtest) != digest(record["backtest"]):
        raise ValueError("Saved research backtest differs")
    return {"status": record["status"], "decision": decision, "backtest": backtest, "semantic_grounding": semantic}
