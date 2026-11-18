"""Dissertation experiment with explicit local inputs and replayable portfolios."""
from dataclasses import replace
from datetime import date
import json
import os

from src.agents.models import AgentReport
from src.agents.pm_deterministic import ROLE_WEIGHTS, portfolio_manager_decide, strong_disagreement
from src.agents.schemas import AgentReportSchema
from src.eval.execution import capture_execution
from src.eval.protocol import decision_cutoff
from src.eval.records import RunRecorder, code_identity, digest, model_identity, write_json
from src.eval.research import ANALYSIS_PROMPT, GROUNDED, ROLES, ROLE_TASKS, budgeted_call, risk_conditions
from src.eval.research_safety import public_value
from src.eval.thesis_sources import LocalEvidence, read_json_input
from src.kg.collaboration import CollaborativeUpdater, GraphLedger
from src.kg.thesis_paths import (
    BEAM_WIDTH, MAX_HOPS, MIN_CONFIDENCE, MIN_SUPPORT, MINING_SAMPLE_WIDTH, MINING_SEED,
    PATH_LIMIT, freeze_graph, mine_rules, retrieve_paths,
)
from src.llm.client import LLMConfig, OpenAICompatibleClient
from src.llm.thesis_scoring import CLASSIFICATION_PROMPT, request_class_logprobs, score_logprobs
from src.sim.thesis_portfolio import COST_BP, HOLDING_SESSIONS, TAIL_PERCENT, MarketTape, settle_portfolios

VERSION = "thesis_v1"
REPORT_BUDGET = 511


def _task_analysis(ticker, cutoff, context, analyst, classifier):
    results = {}
    allowed_evidence = {row["evidence_id"] for row in context["evidences"]}
    allowed_claims = {row["entity_id"] for row in context["claims"]}
    for role in ROLES:
        shared = {"ticker": ticker, "cutoff": cutoff.isoformat(), "role": role, "input": context}
        report_prompt = "\n".join((ANALYSIS_PROMPT, GROUNDED, ROLE_TASKS[role]))
        user = json.dumps({**shared, "report_schema": AgentReportSchema.model_json_schema()}, ensure_ascii=False)
        response = public_value(budgeted_call(analyst, report_prompt, user, REPORT_BUDGET))
        report = AgentReportSchema.model_validate(response).model_dump()
        if not set(report["evidence_refs"]) <= allowed_evidence or not set(report["claim_refs"]) <= allowed_claims:
            raise ValueError("Analyst referenced evidence outside the retrieved paths")
        logprobs = classifier(CLASSIFICATION_PROMPT + "\n" + ROLE_TASKS[role],
                              json.dumps(shared, ensure_ascii=False))
        score = score_logprobs(logprobs)
        if (score != 0 or report["stance"] in {"bullish", "bearish"}) and not report["evidence_refs"]:
            raise ValueError("Directional analyst output requires a supporting evidence reference")
        results[role] = {"report": report, "class_logprobs": logprobs}
    return results


def _decision(ticker, day, snapshot, context, analyses):
    cutoff = decision_cutoff(day)
    conditions = risk_conditions(snapshot, cutoff)
    if conditions["weak_evidence"]:
        gate = "weak_evidence"
    elif conditions["conflicting_evidence"]:
        gate = "conflicting_evidence"
    elif not context["paths"]:
        gate = "no_rule_paths"
    else:
        gate = None
    if not context["paths"]:
        if analyses:
            raise ValueError("Unexpected model output without paths")
        return {"status": "success", "score": None, "gate": gate, "risk_conditions": conditions}
    if set(analyses) != set(ROLES):
        raise ValueError("Decision requires all four analyst classifications")
    reports = []
    for role in ROLES:
        item = analyses[role]
        report = AgentReportSchema.model_validate(item["report"]).model_dump()
        reports.append(AgentReport(role=role, **{**report, "score": score_logprobs(item["class_logprobs"])}))
    score = sum(report.score * report.confidence * ROLE_WEIGHTS[report.role] for report in reports)
    pm_trace = portfolio_manager_decide(ticker, cutoff, reports, snapshot, evaluation_version="v2")
    if not gate and strong_disagreement(pm_trace["conflict_level"], score):
        gate = "strong_analyst_disagreement"
    return {"status": "success", "score": score, "gate": gate, "risk_conditions": conditions,
            "pm_diagnostics": pm_trace}


def _schedule(market, train_range, eval_range, mode, decision_dates):
    training_start, training_end, eval_start, eval_end = [
        date.fromisoformat(day).isoformat() for day in (*train_range, *eval_range)
    ]
    if not training_start <= training_end < eval_start <= eval_end:
        raise ValueError("Training and evaluation must be chronological, non-overlapping intervals")
    training = [day for day in market.sessions if training_start <= day <= training_end]
    evaluation = [day for day in market.sessions if eval_start <= day <= eval_end]
    if decision_dates is not None:
        days = [date.fromisoformat(day).isoformat() for day in decision_dates]
        if len(set(days)) != len(days) or not set(days) <= set(evaluation):
            raise ValueError("Decision dates must be unique sessions inside the evaluation interval")
        evaluation = sorted(days)
    elif mode == "independent":
        first_by_month = {}
        for day in evaluation:
            first_by_month.setdefault(day[:7], day)
        evaluation = list(first_by_month.values())
    if not training or not evaluation:
        raise ValueError("Training and evaluation each require exchange sessions")
    return training, evaluation


def run_thesis(evidence_payload, market_payload, *, tickers, train_range, eval_range, output_dir, analyst,
               classifier, proposer, verifier, source_hashes, portfolio_mode="independent", decision_dates=None,
               classification_identity=None):
    if portfolio_mode not in {"independent", "rolling"}:
        raise ValueError("Invalid portfolio mode")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Specify a unique, non-empty ticker universe")
    tickers = sorted(tickers)
    evidence = LocalEvidence(evidence_payload)
    market = MarketTape(market_payload)
    training, evaluation = _schedule(market, train_range, eval_range, portfolio_mode, decision_dates)
    policy = {"portfolio_mode": portfolio_mode, "cutoff": "exclusive_hk_end_of_day", "evidence_limit": 40,
              "min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE, "min_hops": 2, "max_hops": MAX_HOPS,
              "mining_sample_width": MINING_SAMPLE_WIDTH, "mining_seed": MINING_SEED,
              "beam_width": BEAM_WIDTH, "path_limit": PATH_LIMIT, "holding_sessions": HOLDING_SESSIONS,
              "cost_bp_per_side": COST_BP, "tail_percent": TAIL_PERCENT, "report_budget_per_role": REPORT_BUDGET,
              "class_budget_per_role": 1, "class_temperature": 1, "class_top_logprobs": 20,
              "role_weights": ROLE_WEIGHTS, "training_label": "next_open_to_fifth_close_prior_eval_session"}
    normalized_market = {"sessions": market.sessions, "prices": [
        {"ticker": ticker, "date": day, **prices} for (ticker, day), prices in sorted(market.prices.items())
        if ticker in tickers
    ]}
    identity = {"research_version": "research_v1", "thesis_version": VERSION, "protocol_version": "pit_v1",
                "evaluation_version": "v2", "policy": policy,
                "tickers": tickers, "train_range": train_range, "evaluation_range": eval_range,
                "train_dates": training, "decision_dates": evaluation, "source_hashes": source_hashes,
                "market_digest": digest(normalized_market), "code": code_identity(),
                "analysis_prompt_digest": digest([ANALYSIS_PROMPT, GROUNDED, ROLE_TASKS, CLASSIFICATION_PROMPT]),
                "analyst": model_identity(analyst), "classifier": classification_identity,
                "constructor": model_identity(proposer), "verifier": model_identity(verifier)}
    recorder = RunRecorder(output_dir, identity)
    write_json(recorder.path / "market.json", normalized_market)
    write_json(recorder.path / "source_rejections.json", evidence.rejected)
    updater = CollaborativeUpdater(proposer, verifier)
    ledgers = {ticker: GraphLedger(ticker, strict_cutoff=True) for ticker in tickers}
    training_examples, records, tasks = [], [], {}
    rules = []
    for phase, dates in (("train", training), ("evaluate", evaluation)):
        if phase == "evaluate":
            rules = mine_rules(training_examples)
            write_json(recorder.path / "rules.json", {"digest": digest(rules), "rules": rules})
        for day in dates:
            cutoff = decision_cutoff(day)
            window = market.window(day) if phase == "train" else []
            for ticker in tickers:
                inputs = {"cutoff": cutoff.isoformat(), "phase": phase}
                if phase == "evaluate":
                    inputs["rules_digest"] = digest(rules)
                status, issues, row = "success", [], None
                stage = "source"
                with capture_execution() as capture:
                    try:
                        if ticker not in market.tickers:
                            raise ValueError("Missing OHLCV input for ticker")
                        raw = evidence.snapshot(ticker, cutoff)
                        inputs["source_snapshot_digest"] = digest(raw)
                        stage = "graph"
                        ledger = GraphLedger(ticker, ledgers[ticker].state, strict_cutoff=True)
                        update = updater.update(ledger, raw, cutoff, limit=policy["evidence_limit"])
                        snapshot = freeze_graph(ledger, raw, cutoff, limit=policy["evidence_limit"])
                        inputs.update(snapshot=snapshot, graph_update=public_value(update))
                        ledgers[ticker] = ledger
                        if phase == "train":
                            label, reason = None, "label_unavailable_before_evaluation"
                            if len(window) == HOLDING_SESSIONS and window[-1] < evaluation[0]:
                                entry, exit_price = market.marks(ticker, window)
                                reason = "missing_price"
                                if entry is not None and exit_price is not None:
                                    reason = "flat_return"
                                    if entry != exit_price:
                                        label, reason = ("UP" if exit_price > entry else "DOWN"), None
                            row = {"label": label, "label_end": window[-1] if len(window) == HOLDING_SESSIONS else None,
                                   "excluded_reason": reason}
                            if label:
                                training_examples.append((snapshot, cutoff, label))
                        else:
                            stage = "rule_retrieval"
                            context, retrieval = retrieve_paths(snapshot, cutoff, rules)
                            inputs.update(context=context, retrieval=retrieval)
                            stage = "analysis"
                            analyses = _task_analysis(ticker, cutoff, context, analyst, classifier) if context["paths"] else {}
                            row = {**_decision(ticker, day, snapshot, context, analyses), "analyses": analyses}
                    except Exception as exc:
                        status = "failed"
                        issues = [{"stage": stage, "error_type": type(exc).__name__}]
                calls = public_value(capture.calls)
                records.append(recorder.save(ticker, day, VERSION, inputs, status=status,
                                             row=row, issues=issues, calls=calls))
                if phase == "evaluate":
                    tasks[ticker, day] = row if status == "success" else {"status": status}
    portfolios = settle_portfolios(tasks, tickers, evaluation, market, portfolio_mode)
    training_rows = [record["row"] or {} for record in records if record["inputs"]["phase"] == "train"]
    summary = _portfolio_summary(portfolios)
    summary.update(training_labeled=len(training_examples), training_tasks=len(training_rows), rules=len(rules),
                   rejected_source_rows=len(evidence.rejected),
                   empty_retrieval=sum(record["inputs"].get("retrieval", {}).get("matched_count") == 0
                                       for record in records))
    write_json(recorder.path / "portfolios.json", portfolios)
    write_json(recorder.path / "thesis_summary.json", summary)
    recorder.finish(records)
    return recorder.path, summary


def _portfolio_summary(portfolios):
    settled = [row["net_return"] for row in portfolios if row["status"] == "settled"]
    return {"decision_dates": len(portfolios), "settled_batches": len(settled),
            "unsettled_batches": sum(row["status"] == "unsettled" for row in portfolios),
            "failed_selection_dates": sum(row["status"] == "selection_failed" for row in portfolios),
            "no_trade_dates": sum(row["status"] == "no_trade" for row in portfolios),
            "capacity_skips": sum(row["status"] == "capacity_skipped" for row in portfolios),
            "mean_settled_net_return": sum(settled) / len(settled) if settled else None}


def replay_thesis(run_dir):
    manifest, _ = read_json_input(run_dir / "manifest.json")
    identity = manifest["identity"]
    if (identity.get("thesis_version") != VERSION or digest(identity) != manifest["identity_digest"]
            or (identity["policy"]["holding_sessions"], identity["policy"]["tail_percent"],
                identity["policy"]["cost_bp_per_side"], identity["policy"]["role_weights"])
            != (HOLDING_SESSIONS, TAIL_PERCENT, COST_BP, ROLE_WEIGHTS)):
        raise ValueError("Invalid dissertation run manifest")
    market_payload, _ = read_json_input(run_dir / "market.json")
    if digest(market_payload) != identity["market_digest"]:
        raise ValueError("Saved market inputs changed")
    rules, _ = read_json_input(run_dir / "rules.json")
    if digest(rules["rules"]) != rules["digest"]:
        raise ValueError("Saved rules changed")
    tickers, dates = identity["tickers"], identity["decision_dates"]
    tasks = {}
    for day in dates:
        for ticker in tickers:
            record, _ = read_json_input(run_dir / "tasks" / (digest([ticker, day, VERSION]) + ".json"))
            inputs = record["inputs"]
            if (digest(inputs) != record["input_digest"] or record["ticker"] != ticker
                    or record["trade_date"] != day or record["system"] != VERSION
                    or record["run_id"] != manifest["run_id"] or inputs["rules_digest"] != rules["digest"]):
                raise ValueError("Saved task identity changed")
            if record["status"] == "success":
                context, retrieval = retrieve_paths(inputs["snapshot"], decision_cutoff(day), rules["rules"])
                if digest(context) != digest(inputs["context"]) or retrieval != inputs["retrieval"]:
                    raise ValueError("Saved evidence retrieval does not match replay")
                analyses = record["row"]["analyses"]
                result = _decision(ticker, day, inputs["snapshot"], inputs["context"], analyses)
                if digest({**result, "analyses": analyses}) != digest(record["row"]):
                    raise ValueError("Saved analyst aggregation does not match replay")
                tasks[ticker, day] = result
            else:
                tasks[ticker, day] = {"status": record["status"]}
    portfolios = settle_portfolios(tasks, tickers, dates, MarketTape(market_payload), identity["policy"]["portfolio_mode"])
    saved, _ = read_json_input(run_dir / "portfolios.json")
    if digest(portfolios) != digest(saved):
        raise ValueError("Saved portfolio does not match replay")
    return _portfolio_summary(portfolios)


def run_thesis_command(args):
    evidence, evidence_hash = read_json_input(args.evidence)
    market, market_hash = read_json_input(args.market)
    config = LLMConfig.from_env()
    analyst, proposer, verifier = [
        OpenAICompatibleClient(replace(config, model=os.getenv(f"RESEARCH_{role}_MODEL") or config.model))
        for role in ("ANALYST", "CONSTRUCTOR", "VERIFIER")
    ]

    def classifier(system, user):
        return request_class_logprobs(analyst.config, system, user)

    path, summary = run_thesis(
        evidence, market, tickers=args.tickers, train_range=args.train, eval_range=args.evaluate,
        output_dir=args.output_dir, analyst=analyst, classifier=classifier, proposer=proposer, verifier=verifier,
        source_hashes={"evidence_file": evidence_hash, "market_file": market_hash},
        portfolio_mode=args.portfolio_mode, decision_dates=args.decision_dates, classification_identity=model_identity(analyst),
    )
    print(json.dumps({"run_dir": str(path), **summary}, ensure_ascii=False))
