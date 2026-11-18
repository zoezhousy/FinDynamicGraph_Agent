"""Research reports and read-only historical review with traceable case selection."""
from pathlib import Path
import json
from collections import Counter

import pandas as pd

from src.agents.evidence import extract_retrieved_ids
from src.eval.metrics import full_summary_by_system, paired_significance_tests
from src.eval.records import digest, write_json, json_value
from src.eval.replay import replay_record
from src.eval.reporting import _make_serializable
from src.eval.research_safety import public_value


def export_research(df, output_dir, *, updates):
    output_dir = Path(output_dir)
    if not df.empty:
        if df.duplicated(["ticker", "trade_date", "system"]).any():
            raise ValueError("Duplicate research decision keys")
        if set(df.get("research_version", [])) != {"research_v1"}:
            raise ValueError("Research report cannot mix protocol versions")
        successful = df.loc[df.execution_status == "success"]
        full_corpus = successful.loc[successful.system == "full_graph"].set_index(["ticker", "trade_date"])
        for variant in ("text_same_evidence", "no_grounding", "single_agent"):
            other = successful.loc[successful.system == variant].set_index(["ticker", "trade_date"])
            for key in full_corpus.index.intersection(other.index):
                if full_corpus.loc[key, "corpus_digest"] != other.loc[key, "corpus_digest"]:
                    raise ValueError("Controlled ablation corpus mismatch")
                if "snapshot_digest" in successful and full_corpus.loc[key, "snapshot_digest"] != other.loc[key, "snapshot_digest"]:
                    raise ValueError("Controlled ablation snapshot mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not updates.empty:
        updates.to_csv(output_dir / "kg_update_latency.csv", index=False)
    if df.empty:
        write_json(output_dir / "research_report.json", {"status": "no_successful_rows", "claims": "No research effect established."})
        return
    _make_serializable(df).to_csv(output_dir / "research_decisions.csv", index=False)
    valid = df.loc[(df.execution_status == "success") & df.settled.fillna(False)].copy()
    full_summary_by_system(valid, evaluation_version="v2").to_csv(output_dir / "research_summary.csv")
    paired_significance_tests(valid, evaluation_version="v2").to_csv(output_dir / "research_significance.csv", index=False)
    semantic_rows, risk_rows, performance_rows = [], [], []
    for system, group in df.loc[df.execution_status == "success"].groupby("system"):
        semantic = list(group.semantic_grounding)
        units = sum(s["n_units"] for s in semantic)
        supported = sum(s["n_supported"] for s in semantic)
        citations = sum(s["n_citations"] for s in semantic)
        factual = sum(s["n_factual_units"] for s in semantic)
        unsupported_factual = sum(s["n_unsupported_factual_units"] for s in semantic)
        relevant = sum(sum(c["relevant"] for c in s["response"]["citations"]) for s in semantic)
        annotated = [s["key_evidence_coverage"] for s in semantic if s["coverage_status"] == "annotated"]
        semantic_rows.append({"system": system, "n_decisions": len(group), "n_units": units,
                              "faithfulness_score": supported / units if units else None,
                              "hallucination_rate": (units - supported) / units if units else None,
                              "n_factual_units": factual, "factual_hallucination_rate": unsupported_factual / factual if factual else None,
                              "semantic_citation_precision": relevant / citations if citations else None,
                              "n_citations": citations, "n_coverage_annotations": len(annotated),
                              "mean_key_evidence_coverage": sum(annotated) / len(annotated) if annotated else None})
        conditions = list(group.risk_conditions)
        conservative = list(group.action.isin(["hold", "abstain"]))
        risk_row = {"system": system, "n_decisions": len(group)}
        for condition in ("weak_evidence", "conflicting_evidence"):
            positions = [i for i, value in enumerate(conditions) if value[condition]]
            risk_row["n_" + condition] = len(positions)
            risk_row["conservative_rate_" + condition] = sum(conservative[i] for i in positions) / len(positions) if positions else None
        risk_rows.append(risk_row)
        performance_rows.append({"system": system, "n_decisions": len(group), "mean_decision_latency": group.decision_latency.mean(),
                                 "mean_retrieval_latency": group.retrieval_latency.mean() if "retrieval_latency" in group else None,
                                 "mean_analysis_calls": group.analysis_calls.map(len).mean(),
                                 "mean_prompt_characters": group.analysis_calls.map(lambda calls: sum(c["prompt_characters"] for c in calls)).mean(),
                                 "max_response_token_cap": group.analysis_calls.map(lambda calls: sum(c["max_response_tokens"] for c in calls)).max()})
    pd.DataFrame(semantic_rows).to_csv(output_dir / "semantic_grounding_summary.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output_dir / "risk_behavior.csv", index=False)
    pd.DataFrame(performance_rows).to_csv(output_dir / "retrieval_and_decision_latency.csv", index=False)
    # Verify matched evidence for each controlled comparison; static data freshness is intentionally different.
    full = valid.loc[valid.system == "full_graph"].set_index(["ticker", "trade_date"])
    comparisons = []
    for variant in ("text_same_evidence", "no_grounding", "single_agent", "static_graph"):
        other = valid.loc[valid.system == variant].set_index(["ticker", "trade_date"])
        matched = full.index.intersection(other.index)
        same_corpus = all(full.loc[key, "corpus_digest"] == other.loc[key, "corpus_digest"] for key in matched)
        if variant != "static_graph" and not same_corpus:
            raise ValueError("Controlled ablation corpus mismatch")
        baseline, comparison = full.loc[matched], other.loc[matched]
        support_pairs = [
            a["n_supported"] / a["n_units"] - b["n_supported"] / b["n_units"]
            for a, b in zip(baseline.semantic_grounding, comparison.semantic_grounding)
            if a["n_units"] and b["n_units"]
        ]
        comparisons.append({"full_variant": "full_graph", "ablation": variant, "n_pairs": len(matched),
                            "n_support_pairs": len(support_pairs),
                            "mean_support_rate_delta": sum(support_pairs) / len(support_pairs) if support_pairs else None,
                            "mean_net_return_delta": (baseline.raw_return - comparison.raw_return).mean() if len(matched) else None,
                            "abstention_rate_delta": (
                                baseline.action.eq("abstain").mean() - comparison.action.eq("abstain").mean()
                                if len(matched) else None
                            ),
                            "same_corpus": same_corpus, "changes": {"text_same_evidence": "representation", "no_grounding": "citation_requirement_and_validation",
                            "single_agent": "one_integrated_call_instead_of_four_role_calls", "static_graph": "frozen_pre_experiment_state"}[variant]})
    write_json(output_dir / "research_report.json", {"protocol": "research_v1", "evaluation_version": "v2",
               "delta_direction": "full_graph_minus_ablation", "comparisons": comparisons,
               "limitations": ["Judge scores require independent human agreement validation.", "Input-token use and call counts differ for architecture ablation; response-token caps are matched.",
                               "Static ablation changes evidence freshness/coverage and does not isolate representation.", "Conservative risk gates measure policy behavior, not standalone model risk understanding.",
                               "No human key-evidence annotations means coverage recall remains unmeasured."]})


def historical_review(run_dir, output_dir):
    """Explicit user-authorized review only; original artifacts are never rewritten."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Historical review must use a separate output directory")
    paths = sorted((run_dir / "tasks").glob("*.json"))
    if not paths:
        raise ValueError("No saved task records")
    records, rows, replay_errors = [], [], []
    original_hashes = {str(path): digest(path.read_bytes().hex()) for path in paths}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if digest(record["inputs"]) != record["input_digest"]:
            raise ValueError("Historical input integrity check failed")
        snapshot = record.get("decision_snapshot")
        if snapshot is not None and digest(snapshot) != record.get("decision_snapshot_digest"):
            raise ValueError("Historical decision snapshot integrity check failed")
        record["review_task_path"] = str(path)
        records.append(record)
        if record.get("row"):
            rows.append(record["row"])
        try:
            replayed = replay_record(record)
            if record.get("decision") and replayed["decision"]:
                for field in ("action", "final_score", "confidence", "conflict_level", "evidence_refs", "claim_refs", "fresh_evidence_count", "stale_evidence_count"):
                    if replayed["decision"].get(field) != record["decision"].get(field):
                        raise ValueError("Replay decision differs")
                if replayed["backtest"] != record["backtest"]:
                    raise ValueError("Replay backtest differs")
        except Exception as exc:
            replay_errors.append({"ticker": record["ticker"], "trade_date": record["trade_date"], "system": record["system"], "error_type": type(exc).__name__})
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if manifest.get("identity_digest") != digest(identity):
        raise ValueError("Historical manifest integrity check failed")
    df = pd.DataFrame(rows)
    valid = df.loc[(df.execution_status == "success") & df.settled.fillna(False)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    full_summary_by_system(valid, evaluation_version="legacy").to_csv(output_dir / "historical_metrics_legacy.csv")
    # Clearly labelled metric-only reanalysis of saved actions, not new research outcomes.
    paired_significance_tests(valid, evaluation_version="v2").to_csv(output_dir / "historical_pairwise_metric_reanalysis_v2.csv", index=False)
    selected = select_cases(records)
    cases = []
    for record, group in selected:
        case = build_case(record, group)
        peers = [r for r in records if r["ticker"] == record["ticker"] and r["trade_date"] == record["trade_date"]]
        case["same_date_comparison"] = [{"system": r["system"], "status": r["status"],
                                        "action": (r.get("row") or {}).get("action"),
                                        "net_return": (r.get("row") or {}).get("raw_return")} for r in peers]
        case["selection_rule"] = {"mechanism": "Maximum conflict level, then claim count, then evidence count.",
                                   "relative_best": "Maximum dynamic net return minus strongest other successful system on fully matched dates.",
                                   "relative_worst": "Minimum dynamic net return minus weakest other successful system on fully matched dates."}[group]
        cases.append(public_value(case))
    write_json(output_dir / "case_studies.json", cases)
    coverage = []
    for system, group in valid.groupby("system"):
        original = [r for r in records if r["system"] == system]
        coverage.append({"system": system, "n_successful_settled": len(group), "status_counts": dict(Counter(r["status"] for r in original)),
                         "snapshot_news_rows": sum(len((r.get("decision_snapshot") or r["inputs"].get("snapshot") or {}).get("news", [])) for r in original),
                         "snapshot_claim_rows": sum(len((r.get("decision_snapshot") or r["inputs"].get("snapshot") or {}).get("claims", [])) for r in original)})
    review = {"run_id": manifest["run_id"], "source_run": str(run_dir), "review_version": "historical_review_v1",
              "protocol_version": identity["protocol_version"], "original_evaluation_version": identity.get("evaluation_version", "legacy"),
              "tickers": identity["tickers"], "trade_dates": identity["trade_dates"], "n_tasks": len(records),
              "status_counts": dict(Counter(r["status"] for r in records)), "n_successful_settled": len(valid),
              "future_count": sum((r.get("audit") or {}).get("future_count", 0) for r in records),
              "audited_tasks": sum(r.get("audit") is not None for r in records), "replay_checked": len(records), "replay_errors": replay_errors,
              "coverage": coverage, "case_count": len(cases),
              "source_tasks_unchanged": all(digest(Path(path).read_bytes().hex()) == value for path, value in original_hashes.items()),
              "limitations": ["Original run predates graph_collaboration_v1 and semantic_grounding_v1.",
                              "Citation existence is not semantic support; no semantic score is fabricated.",
                              "Original systems differ in inputs, architecture and decision rules; not controlled causal ablations.",
                              "Historical availability is based on recorded metadata, not independently verified disclosure archives."]}
    write_json(output_dir / "historical_review.json", review)
    markdown = ["# Auditable historical cases", "", "These are saved historical decisions, not validation of the new research protocol.", ""]
    for case in cases:
        markdown.extend([f"## {case['case_group']}: {case['ticker']} {case['trade_date']}", "",
                         f"Action: {case['action']}; net return: {case['raw_return']:.6f}; input integrity: verified.",
                         f"Original task: {case['task_path']}", f"Input digest: {case['input_digest']}",
                         f"Citations matched: {case['n_matched_citations']}/{case['n_citations']}; semantics: not evaluated.", "",
                         "```json", json.dumps(json_value(case), ensure_ascii=False, indent=2), "```", ""])
    (output_dir / "case_studies.md").write_text("\n".join(markdown), encoding="utf-8")
    return review


def select_cases(records):
    successful = [r for r in records if r["status"] == "success" and r.get("row", {}).get("settled")]
    dynamic = [r for r in successful if r["system"] == "kg_dynamic"]
    if not dynamic:
        return []
    mechanism = max(dynamic, key=lambda r: (r["row"].get("conflict_level", 0), len(r["row"].get("claim_refs", [])), len(r["row"].get("evidence_refs", []))))
    selected, used = [(mechanism, "mechanism")], {(mechanism["ticker"], mechanism["trade_date"])}
    groups = {}
    for record in successful:
        groups.setdefault((record["ticker"], record["trade_date"]), {})[record["system"]] = record
    comparable = [group for group in groups.values() if set(group) >= {"kg_dynamic", "static_kg", "no_kg_no_evidence", "evidence_no_kg"}]
    for label, reverse in (("relative_best", True), ("relative_worst", False)):
        ranked = sorted(comparable, key=lambda g: g["kg_dynamic"]["row"]["raw_return"] -
                        (max if reverse else min)(r["row"]["raw_return"] for system, r in g.items() if system != "kg_dynamic"), reverse=reverse)
        candidate = next((g["kg_dynamic"] for g in ranked if (g["kg_dynamic"]["ticker"], g["kg_dynamic"]["trade_date"]) not in used), None)
        if candidate:
            selected.append((candidate, label))
            used.add((candidate["ticker"], candidate["trade_date"]))
    return selected


def build_case(record, group):
    snapshot = record.get("decision_snapshot") or record["inputs"].get("snapshot") or {}
    row = record["row"]
    index = {}
    for key in ("signals", "fundamentals", "risks", "news", "evidences", "sources"):
        for item in snapshot.get(key, []):
            eid = item.get("evidence_id") or item.get("entity_id") or item.get("source_id")
            if eid:
                allowed = ("evidence_id", "entity_id", "source_id", "source_type", "title", "name", "direction", "strength", "metric", "value", "published_at", "effective_at", "as_of_date", "valid_from", "valid_to", "extracted_text")
                public = {k: item[k] for k in allowed if k in item}
                if "extracted_text" in public:
                    public["excerpt_truncated"] = len(public["extracted_text"]) > 1200
                    public["extracted_text"] = public["extracted_text"][:1200]
                index[str(eid)] = public
    refs = row.get("evidence_refs", [])
    matched = [eid for eid in refs if eid in index]
    claims = [{k: c[k] for k in ("entity_id", "claim_id", "text", "polarity", "evidence_ids", "as_of_date", "valid_from", "valid_to") if k in c}
              for c in snapshot.get("claims", []) if (c.get("entity_id") or c.get("claim_id")) in row.get("claim_refs", [])]
    return {"case_group": group, "ticker": record["ticker"], "trade_date": record["trade_date"], "action": row["action"],
            "raw_return": row["raw_return"], "task_path": record["review_task_path"], "input_digest": record["input_digest"],
            "snapshot_digest": record.get("decision_snapshot_digest"), "n_citations": len(refs), "n_matched_citations": len(matched),
            "unmatched_citations": [eid for eid in refs if eid not in index], "evidence": [index[eid] for eid in matched], "claims": claims,
            "agent_reports": [{k: r.get(k) for k in ("role", "stance", "confidence", "score", "summary", "evidence_refs", "claim_refs")} for r in row.get("agent_reports", [])],
            "decision_trace": row.get("decision_trace"), "backtest": record["backtest"], "temporal_audit": record.get("audit"),
            "limitations": ["Citation matching does not establish relevance or semantic support.", "Evidence-to-source edges not stored in the snapshot are not inferred."]}
