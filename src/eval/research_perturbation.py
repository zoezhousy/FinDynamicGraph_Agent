"""Evidence-dependence diagnostics on explicitly selected research snapshots."""
from copy import deepcopy
from pathlib import Path

from src.eval.execution import capture_execution
from src.eval.protocol import decision_cutoff
from src.eval.records import code_identity, digest, model_identity, write_json
from src.eval.research import ResearchConfig, research_decision
from src.eval.research_safety import public_value
from src.eval.research_sources import ResearchSnapshotSource
from src.eval.temporal_audit import assert_no_future, audit_subgraph


def _without_evidence(snapshot, evidence_id):
    result = deepcopy(snapshot)
    result["evidences"] = [row for row in result["evidences"] if row["evidence_id"] != evidence_id]
    removed_claims = {row["entity_id"] for row in result["claims"] if evidence_id in row["evidence_ids"]}
    result["claims"] = [row for row in result["claims"] if row["entity_id"] not in removed_claims]
    removed_nodes = removed_claims | {evidence_id}
    result["relations"] = [
        row for row in result["relations"]
        if row["start_id"] not in removed_nodes and row["end_id"] not in removed_nodes
    ]
    result["conflicts"] = [
        row for row in result.get("conflicts", [])
        if row["claim_a"] not in removed_claims and row["claim_b"] not in removed_claims
    ]
    return result


def _views(snapshot, case):
    key, control = case["key_evidence_id"], case["control_evidence_id"]
    evidence_ids = {row["evidence_id"] for row in snapshot["evidences"]}
    if key == control or not {key, control} <= evidence_ids:
        raise ValueError("Select distinct key and control evidence IDs in the frozen snapshot")
    edge = case["critical_relation"]
    fields = ("start_id", "end_id", "type")
    matching = [row for row in snapshot["relations"] if all(row[field] == edge[field] for field in fields)]
    if len(matching) != 1:
        raise ValueError("Critical relation must identify exactly one frozen edge")
    relation_only = deepcopy(snapshot)
    relation_only["relations"].remove(matching[0])
    if edge["type"] == "CONFLICTS_WITH":
        relation_only["conflicts"] = [
            row for row in relation_only.get("conflicts", [])
            if (row["claim_a"], row["claim_b"]) != (edge["start_id"], edge["end_id"])
        ]
    return {
        "original": snapshot,
        "without_key_evidence": _without_evidence(snapshot, key),
        "without_critical_relation": relation_only,
        "without_control_evidence": _without_evidence(snapshot, control),
    }


def run_evidence_perturbation(run_dir, output_dir, cases, *, analyst_client):
    output_dir = Path(output_dir).resolve()
    source = ResearchSnapshotSource(run_dir, output_dir)
    manifest, identity = source.manifest, source.identity
    if not isinstance(cases, list) or not cases:
        raise ValueError("Supply a non-empty list of annotated cases")
    config = ResearchConfig(response_token_budget=identity["config"]["response_token_budget"],
                            risk_policy=identity["config"]["risk_policy"])
    prepared = {}
    for case in cases:
        if set(case) != {"ticker", "trade_date", "key_evidence_id", "control_evidence_id", "critical_relation"}:
            raise ValueError("Each case must contain only the documented selection fields")
        if set(case["critical_relation"]) != {"start_id", "end_id", "type"}:
            raise ValueError("Critical relation requires start_id, end_id and type")
        ticker, date = case["ticker"], case["trade_date"]
        if (ticker, date) in prepared:
            raise ValueError("Cases must be unique ticker/date keys")
        record = source.load_task(ticker, date)
        snapshot = record["decision_snapshot"]
        when = decision_cutoff(date)
        views = _views(snapshot, case)
        for view in views.values():
            assert_no_future(audit_subgraph(view, when), context="evidence perturbation")
        prepared[ticker, date] = (case, views, record["input_digest"])

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", {
        "source_run_id": manifest["run_id"], "source_identity_digest": manifest["identity_digest"],
        "code": code_identity(), "analyst": model_identity(analyst_client), "risk_policy": config.risk_policy,
        "response_token_budget": config.response_token_budget, "cases": cases,
    })
    rows = []
    for (ticker, date), (case, views, source_digest) in prepared.items():
        baseline = None
        for scenario, view in views.items():
            row = {"ticker": ticker, "trade_date": date, "scenario": scenario,
                   "snapshot_digest": digest(view), "source_input_digest": source_digest}
            with capture_execution() as capture:
                try:
                    decision, conditions, _ = research_decision(
                        ticker, decision_cutoff(date), view, "full_graph", analyst_client, config
                    )
                    row.update(status=capture.status, action=decision["action"],
                               ungated_action=conditions["ungated_action"], confidence=decision["confidence"],
                               risk_policy_trigger=decision.get("risk_policy_trigger"))
                    if row["status"] == "success":
                        if scenario == "original":
                            baseline = row
                        if baseline is not None:
                            row.update(confidence_delta=row["confidence"] - baseline["confidence"],
                                       action_changed=row["action"] != baseline["action"],
                                       ungated_action_changed=row["ungated_action"] != baseline["ungated_action"])
                except Exception as exc:
                    row.update(status="failed", error_type=type(exc).__name__)
            rows.append(row)
            write_json(output_dir / (digest([ticker, date, scenario]) + ".json"),
                       {**row, "selection": case, "calls": public_value(capture.calls), "issues": capture.issues})
    write_json(output_dir / "perturbation_results.json", rows)
    return rows
