"""Replay saved decisions and backtests without network or credentials."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.agents.models import AgentReport
from src.agents.pm_deterministic import portfolio_manager_decide
from src.eval.protocol import backtest_details
from src.eval.records import digest
from src.sim.backtest import BacktestConfig

def replay_record(record):
    if digest(record["inputs"]) != record["input_digest"]:
        raise ValueError("Saved input digest does not match")
    if record["status"] == "failed" or record.get("decision") is None:
        return {"status": record["status"], "decision": None, "backtest": None}
    inputs = record["inputs"]
    if inputs.get("research_version") == "research_v1":
        from src.eval.research_replay import replay_research_record
        return replay_research_record(record)
    snapshot = record.get("decision_snapshot")
    if snapshot is not None and digest(snapshot) != record["decision_snapshot_digest"]:
        raise ValueError("Saved decision snapshot digest does not match")
    if snapshot is None:
        snapshot = inputs.get("snapshot")
    decision = record["decision"]
    # Full KG decisions are re-aggregated from captured structured agent reports.
    if record["system"] in {"kg_dynamic", "static_kg"} and decision.get("agent_reports"):
        from datetime import datetime
        date = datetime.fromisoformat(inputs["decision_time"])
        decision = portfolio_manager_decide(record["ticker"], date, [AgentReport(**r) for r in decision["agent_reports"]], snapshot, evaluation_version=inputs.get("evaluation_version", "legacy"))
        if record["system"] == "static_kg":
            decision["baseline"] = "static_kg"
        original = record["decision"]
        if original["decision_trace"]["decision_id"].endswith(":" + record["run_id"]):
            decision["decision_trace"]["decision_id"] += ":" + record["run_id"]
            for assessment in decision["agent_assessments"]:
                assessment["assessment_id"] += ":" + record["run_id"]
    if record["system"] in {"no_kg_no_evidence", "evidence_no_kg"}:
        from datetime import datetime
        from src.eval.baselines import decision_no_evidence, decision_evidence
        parsed = [call["parsed_response"] for call in record["calls"] if "parsed_response" in call]
        if parsed:
            builder = decision_no_evidence if record["system"] == "no_kg_no_evidence" else decision_evidence
            decision = builder(parsed[-1], record["ticker"], datetime.fromisoformat(inputs["decision_time"]))
    cfg = BacktestConfig(**inputs["backtest_config"])
    backtest = backtest_details(pd.DataFrame(inputs["ohlcv"]), record["trade_date"], decision["action"], cfg, protocol_version=inputs["protocol_version"])
    return {"status": record["status"], "decision": decision, "backtest": backtest}

def replay_run(run_dir):
    rows = []
    versions = set()
    for path in sorted((Path(run_dir) / "tasks").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        versions.add(record["inputs"].get("evaluation_version", "legacy"))
        replayed = replay_record(record)
        if replayed["decision"] is not None:
            row = {**(record["row"] or {}), **replayed["decision"], "system": record["system"]}
            for key, value in replayed["backtest"].items():
                if key in row or record["inputs"]["protocol_version"] == "pit_v1":
                    row[key] = value
            if record["inputs"]["protocol_version"] == "pit_v1":
                row["execution_status"] = record["status"]
            rows.append(row)
    if len(versions) > 1:
        raise ValueError("Cannot mix evaluation versions in replay")
    result = pd.DataFrame(rows)
    result.attrs["evaluation_version"] = next(iter(versions), "legacy")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    df = replay_run(args.run_dir)
    from src.eval.reporting import _make_serializable
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _make_serializable(df).to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.report_dir:
        if not df.empty and "research_version" in df:
            from src.eval.research_reporting import export_research
            export_research(df, args.report_dir, updates=pd.DataFrame())
            return
        from src.eval.reporting import export_results
        metrics = df
        if not df.empty and "execution_status" in df:
            from src.eval.runner import comparable_rows
            metrics = comparable_rows(df)
        export_results(df, args.report_dir, metrics_df=metrics, evaluation_version=df.attrs.get("evaluation_version", "legacy"))

if __name__ == "__main__":
    main()
