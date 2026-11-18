"""Paired retrieval and optional simulated decisions on saved research snapshots."""
from pathlib import Path
from statistics import mean

import pandas as pd

from src.eval.execution import capture_execution
from src.eval.protocol import backtest_details, decision_cutoff
from src.eval.records import code_identity, digest, model_identity, write_json
from src.eval.research import ResearchConfig, research_decision
from src.eval.research_safety import public_text, public_value
from src.eval.research_sources import ResearchSnapshotSource
from src.kg.research_paths import ABLATION_RULES, RULESET, retrieve_research_evidence
from src.sim.backtest import BacktestConfig


def run_multihop(run_dir, output_dir, cases, *, top_k=4, seed_limit=4, analyst_client=None, ablate_path_rules=False):
    output_dir = Path(output_dir).resolve()
    source = ResearchSnapshotSource(run_dir, output_dir)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Supply a non-empty list of retrieval cases")
    config = ResearchConfig(response_token_budget=source.identity["config"]["response_token_budget"],
                            risk_policy=source.identity["config"]["risk_policy"])
    prepared = {}
    for case in cases:
        required = {"ticker", "trade_date", "query"}
        if not isinstance(case, dict) or not required <= case.keys() or case.keys() - required - {"key_evidence_ids"}:
            raise ValueError("Cases require ticker, trade_date and query, with optional key_evidence_ids")
        ticker, date, query = case["ticker"], case["trade_date"], case["query"]
        if (not all(isinstance(value, str) for value in (ticker, date, query))
                or len(query) > 500 or public_text(query) != query):
            raise ValueError("Use a short lexical query without source URLs")
        if (ticker, date) in prepared:
            raise ValueError("Cases must have unique ticker/date keys")
        task = source.load_task(ticker, date)
        snapshot = task["decision_snapshot"]
        evidence = {row["evidence_id"]: row for row in snapshot["evidences"]}
        keys = case.get("key_evidence_ids")
        if keys is not None and (
            not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys)
            or len(keys) != len(set(keys)) or not set(keys) <= evidence.keys()
        ):
            raise ValueError("Key annotations must be distinct IDs in the frozen source corpus")
        cutoff = decision_cutoff(date)
        results = retrieve_research_evidence(snapshot, ticker, cutoff, query,
                                             top_k=top_k, seed_limit=seed_limit)
        if ablate_path_rules:
            for rule in ABLATION_RULES:
                results[f"without_{rule}"] = retrieve_research_evidence(
                    snapshot, ticker, cutoff, query, top_k=top_k, seed_limit=seed_limit, disabled_rule=rule
                )["graph_multihop"]
        market = (task["inputs"]["ohlcv"], task["inputs"]["backtest_config"]) if analyst_client is not None else None
        prepared[ticker, date] = (evidence, keys, results, market, task["input_digest"],
                                  task["decision_snapshot_digest"])

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", {
        "ruleset": RULESET, "source_run_id": source.manifest["run_id"],
        "source_identity_digest": source.manifest["identity_digest"], "code": code_identity(),
        "top_k": top_k, "seed_limit": seed_limit, "cases": cases,
        "ablate_path_rules": ablate_path_rules,
        "analyst": model_identity(analyst_client) if analyst_client is not None else None,
        "response_token_budget": config.response_token_budget if analyst_client is not None else None,
        "risk_policy": config.risk_policy if analyst_client is not None else None,
    })
    rows = []
    for (ticker, date), (evidence, keys, results, market, input_hash, snapshot_hash) in prepared.items():
        for variant, retrieved in results.items():
            ids = retrieved["evidence_ids"]
            hits = len(set(ids) & set(keys)) if keys else None
            row = {
                "ticker": ticker, "trade_date": date, "variant": variant, **retrieved,
                "source_input_digest": input_hash, "source_snapshot_digest": snapshot_hash,
                "evidence_pool_count": len(evidence), "retrieved_count": len(ids),
                "annotated_key_count": len(keys) if keys else None, "key_hits": hits,
                "key_recall": hits / len(keys) if keys else None,
                "key_precision": hits / len(ids) if keys and ids else None,
                "decision_status": "not_requested",
            }
            if analyst_client is not None:
                _evaluate_decision(row, [evidence[i] for i in ids], market, analyst_client, config)
            write_json(output_dir / (digest([ticker, date, variant]) + ".json"), row)
            rows.append(row)
    write_json(output_dir / "retrieval_results.json", rows)
    write_json(output_dir / "retrieval_summary.json", _paired_summary(rows))
    return rows


def _evaluate_decision(row, selected_evidence, market, client, config):
    ticker, date = row["ticker"], row["trade_date"]
    # All retrieval arms supply only selected source evidence in the same format.
    snapshot = {"company": [{"ticker": ticker}], "evidences": selected_evidence,
                "claims": [], "relations": [], "conflicts": []}
    row["decision_input_digest"] = digest(snapshot)
    stage = "analysis"
    with capture_execution() as capture:
        try:
            decision, conditions, calls = research_decision(
                ticker, decision_cutoff(date), snapshot, "text_same_evidence", client, config
            )
            row.update(action=decision["action"], ungated_action=conditions["ungated_action"],
                       confidence=decision["confidence"], cited_evidence_ids=decision["evidence_refs"],
                       prompt_characters=sum(call["prompt_characters"] for call in calls))
            stage = "backtest"
            prices, settings = market
            outcome = backtest_details(pd.DataFrame(prices), date, decision["action"],
                                       BacktestConfig(**settings), protocol_version="pit_v1")
            row.update(decision_status=capture.status, settled=outcome["settled"],
                       net_return=outcome["raw_return"])
        except Exception as exc:
            row.update(decision_status="failed", error_stage=stage, error_type=type(exc).__name__)
    row.update(calls=public_value(capture.calls), issues=capture.issues)


def _paired_summary(rows):
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], {})[row["ticker"], row["trade_date"]] = row
    full = by_variant["graph_multihop"]
    comparisons = []
    for variant in ("graph_one_hop", "text_lexical", *(f"without_{rule}" for rule in ABLATION_RULES)):
        if variant not in by_variant:
            continue
        pairs = [(row, by_variant[variant][key]) for key, row in full.items()]
        recall = [a["key_recall"] - b["key_recall"] for a, b in pairs if a["key_recall"] is not None]
        returns = [a["net_return"] - b["net_return"] for a, b in pairs if all(
            row["decision_status"] == "success" and row["settled"] and row["net_return"] is not None
            for row in (a, b)
        )]
        comparisons.append({
            "baseline": variant, "n_retrieval_pairs": len(pairs), "n_annotated_pairs": len(recall),
            "mean_key_recall_delta": mean(recall) if recall else None,
            "n_successful_settled_pairs": len(returns),
            "mean_net_return_delta": mean(returns) if returns else None,
            "n_equal_context_size_pairs": sum(a["retrieved_count"] == b["retrieved_count"] for a, b in pairs),
        })
    variant_summary = {}
    for variant, tasks in by_variant.items():
        variant_rows = list(tasks.values())
        recall = [row["key_recall"] for row in variant_rows if row["key_recall"] is not None]
        precision = [row["key_precision"] for row in variant_rows if row["key_precision"] is not None]
        variant_summary[variant] = {
            "n_cases": len(tasks), "n_empty": sum(not row["retrieved_count"] for row in variant_rows),
            "mean_retrieved_count": mean(row["retrieved_count"] for row in variant_rows),
            "n_recall_cases": len(recall), "mean_key_recall": mean(recall) if recall else None,
            "n_precision_cases": len(precision), "mean_key_precision": mean(precision) if precision else None,
            "decision_status_counts": {
                status: sum(row["decision_status"] == status for row in variant_rows)
                for status in ("not_requested", "success", "degraded", "failed")
            },
        }
    return {
        "delta_direction": "graph_multihop_minus_baseline",
        "n_cases": len(full),
        "n_cases_with_multihop_only_evidence": sum(bool(row["new_beyond_direct_support"]) for row in full.values()),
        "variants": variant_summary,
        "comparisons": comparisons,
    }
