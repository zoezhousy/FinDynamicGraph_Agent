"""Four-system execution, separate from CLI and report generation."""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import logging
import time
import pandas as pd
from src.agents.kg_tools import KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator, OrchestratorConfig
from src.config import ExperimentConfig
from src.collectors.cache import load_ohlcv, resolve_ohlcv_path
from src.eval.baselines import baseline_no_kg_no_evidence, baseline_evidence_no_kg, baseline_static_kg, _load_raw_evidence
from src.eval.execution import capture_execution
from src.eval.outcomes import build_backtest_outcome
from src.eval.protocol import decision_cutoff, strict_snapshot, FrozenSnapshotClient, backtest_details
from src.eval.records import RunRecorder, code_identity, model_identity, digest, json_value, write_json
from src.eval.temporal_audit import audit_subgraph, audit_items, assert_no_future
from src.eval.text_evidence import load_text_input
from src.eval.metrics import paired_significance_tests, full_summary_by_system
from src.kg.connections import Neo4jSettings
from src.kg.query import KGQueryClient
from src.kg.store_neo4j import Neo4jKGStore
from src.sim.backtest import BacktestConfig, compute_trade_return

logger = logging.getLogger(__name__)
SYSTEMS = ("kg_dynamic", "no_kg_no_evidence", "evidence_no_kg", "static_kg")

def comparable_rows(df):
    if df.empty:
        return df
    return df.loc[(df["execution_status"] == "success") & df["settled"].fillna(False)].copy()

def run_grid(tickers, trade_dates, collection_config, experiment_start_date="2025-01-01", *, experiment_config=None, query_client=None, store=None, orchestrator=None, model_client=None, llm=None, runners=None, ohlcv_loader=load_ohlcv, query_factory=KGQueryClient, store_factory=Neo4jKGStore, orchestrator_factory=KGBasedOrchestrator, persist_outcomes=True):
    cfg = experiment_config or ExperimentConfig()
    runners = runners or {}
    bt_cfg = BacktestConfig(cfg.holding_days, cfg.transaction_cost_bp)
    dates = [datetime.fromisoformat(d).date().isoformat() if isinstance(d, str) else d.date().isoformat() for d in trade_dates]
    identity = {"protocol_version": cfg.protocol_version, "evaluation_version": cfg.evaluation_version, "tickers": list(tickers), "trade_dates": dates, "static_cutoff": experiment_start_date, "data_root": str(collection_config.output_root), "database": collection_config.neo4j_database, "persist_outcomes": persist_outcomes, "backtest_config": asdict(bt_cfg), "code": code_identity(), "model": model_identity(model_client, llm), "prompt_digest": digest((Path(__file__).parents[1] / "llm" / "prompts.py").read_text(encoding="utf-8"))}
    recorder = RunRecorder(cfg.output_dir, identity, run_id=cfg.run_id, resume=cfg.resume)
    rows, records = [], []
    with ExitStack() as resources:
        settings = None
        if query_client is None or (store is None and persist_outcomes):
            settings = Neo4jSettings.from_env(collection_config.neo4j_database)
        if query_client is None:
            query_client = settings.create(query_factory)
            resources.callback(query_client.close)
        if store is None and persist_outcomes:
            store = settings.create(store_factory)
            resources.callback(store.close)
            store.init_constraints()
        context = KGAgentContext(query_client)
        if orchestrator is None and cfg.protocol_version == "legacy":
            kwargs = {"model_client": model_client} if model_client is not None else {}
            if llm is not None:
                kwargs["llm"] = llm
            orchestrator = orchestrator_factory(context, **kwargs)
            resources.callback(orchestrator.close)
        static_snapshots = {}
        for ticker in tickers:
            ohlcv = None
            for date in dates:
                request_date = datetime.fromisoformat(date)
                decision_time = decision_cutoff(request_date) if cfg.protocol_version == "pit_v1" else request_date
                cutoff = decision_cutoff(experiment_start_date) if cfg.protocol_version == "pit_v1" else datetime.fromisoformat(experiment_start_date)
                for system in SYSTEMS:
                    inputs = {"protocol_version": cfg.protocol_version, "evaluation_version": cfg.evaluation_version, "decision_time": decision_time.isoformat(), "backtest_config": asdict(bt_cfg), "ohlcv_path": str(resolve_ohlcv_path(collection_config.output_root, ticker))}
                    started = time.perf_counter()
                    retrieval_seconds = 0.0
                    decision = row = backtest = audit = None
                    previous = recorder.existing(ticker, date, system)
                    phase = "input_loading"
                    with capture_execution() as capture:
                        try:
                            if ohlcv is None:
                                ohlcv = ohlcv_loader(collection_config.output_root, ticker)
                            inputs["ohlcv"] = ohlcv.to_dict("records")
                            phase = "retrieval"
                            frozen = None
                            if system in {"kg_dynamic", "static_kg"}:
                                effective = cutoff if system == "static_kg" else decision_time
                                if cfg.protocol_version == "pit_v1" and system == "static_kg" and ticker in static_snapshots:
                                    snapshot = static_snapshots[ticker]
                                else:
                                    snapshot = context.load_subgraph(ticker, effective)
                                    if cfg.protocol_version == "pit_v1" and system == "static_kg":
                                        static_snapshots[ticker] = snapshot
                                if cfg.protocol_version == "pit_v1":
                                    if effective > decision_time:
                                        raise ValueError("Static cutoff exceeds decision time")
                                    snapshot = strict_snapshot(snapshot, effective)
                                    audit = audit_subgraph(snapshot, decision_time)
                                    assert_no_future(audit, context=system)
                                    frozen = FrozenSnapshotClient(ticker, effective, snapshot, request_date=decision_time)
                                inputs.update(snapshot=snapshot, effective_cutoff=effective.isoformat())
                            if system == "evidence_no_kg":
                                if cfg.protocol_version == "pit_v1":
                                    evidence_text, items, source_path = load_text_input(ticker, decision_time, collection_config.output_root, ohlcv)
                                    audit = audit_items(items, decision_time)
                                    assert_no_future(audit, context=system)
                                    inputs.update(text_items=items, news_path=source_path)
                                else:
                                    evidence_text = _load_raw_evidence(ticker, request_date, collection_config.output_root)
                                inputs["evidence_text"] = evidence_text
                            retrieval_seconds = time.perf_counter() - started
                            inputs = json_value(inputs)
                            previous = recorder.previous(ticker, date, system, inputs)
                            if previous is not None and previous["status"] == "success":
                                records.append(previous)
                                if previous["row"] is not None:
                                    rows.append(previous["row"])
                                continue
                            phase = "decision"
                            if system in runners:
                                decision = runners[system](ticker, decision_time, inputs)
                            elif system == "no_kg_no_evidence":
                                decision = baseline_no_kg_no_evidence(ticker, decision_time, model_client=model_client)
                            elif system == "evidence_no_kg":
                                decision = baseline_evidence_no_kg(ticker, decision_time, data_root=collection_config.output_root, model_client=model_client, evidence_text=evidence_text)
                            elif cfg.protocol_version == "pit_v1":
                                with ExitStack() as task_resources:
                                    task_orchestrator = orchestrator_factory(KGAgentContext(frozen), OrchestratorConfig(persist_decision_trace=False, evaluation_version=cfg.evaluation_version), trace_store=None, model_client=model_client, llm=llm)
                                    task_resources.callback(task_orchestrator.close)
                                    decision = task_orchestrator.run_for_ticker(ticker, decision_time)
                                if system == "static_kg":
                                    decision["baseline"] = "static_kg"
                            elif system == "kg_dynamic":
                                decision = orchestrator.run_for_ticker(ticker, decision_time)
                            else:
                                decision = baseline_static_kg(ticker, decision_time, kg_context=context, static_cutoff=cutoff, model_client=model_client, llm=llm)
                            if system == "evidence_no_kg" and cfg.protocol_version == "pit_v1":
                                decision["retrieved_evidence_ids"] = [i["evidence_id"] for i in inputs["text_items"]]
                            phase = "backtest"
                            backtest = backtest_details(ohlcv, date, decision["action"], bt_cfg, protocol_version=cfg.protocol_version)
                            legacy_backtest = compute_trade_return(ohlcv, date, decision["action"], bt_cfg)
                            row = {**decision, **(legacy_backtest if cfg.protocol_version == "legacy" else backtest), "system": system}
                            phase = "persistence"
                            if system == "kg_dynamic":
                                outcome = build_backtest_outcome(ticker, request_date, decision["action"], decision, legacy_backtest, bt_cfg)
                                if not persist_outcomes:
                                    pass
                                elif cfg.protocol_version == "legacy":
                                    store.upsert_backtest_outcome(outcome)
                                else:
                                    # Persist isolated trace/outcome IDs; never overwrite legacy records.
                                    from src.kg.schema import DecisionTrace, AgentAssessment
                                    trace = DecisionTrace.model_validate(decision["decision_trace"])
                                    trace.decision_id = f"{trace.decision_id}:{recorder.run_id}"
                                    assessments = [AgentAssessment.model_validate(a) for a in decision["agent_assessments"]]
                                    for a in assessments:
                                        a.assessment_id = f"{a.assessment_id}:{recorder.run_id}"
                                    decision["decision_trace"] = trace.model_dump(mode="json")
                                    decision["agent_assessments"] = [a.model_dump(mode="json") for a in assessments]
                                    row.update(decision_trace=decision["decision_trace"], agent_assessments=decision["agent_assessments"])
                                    store.upsert_decision_trace(trace, assessments)
                                    outcome.decision_id = trace.decision_id
                                    outcome.outcome_id = f"{outcome.outcome_id}:{recorder.run_id}"
                                    if backtest["settled"]:
                                        store.upsert_backtest_outcome(outcome)
                                row.update(outcome_id=outcome.outcome_id, direction_outcome=outcome.direction_outcome if backtest["settled"] else "unsettled", is_profitable=outcome.is_profitable if backtest["settled"] else None)
                            if cfg.evaluation_version == "v2":
                                row["evaluation_version"] = "v2"
                            if cfg.protocol_version == "pit_v1":
                                row["execution_status"] = capture.status
                            rows.append(row)
                            status = capture.status
                        except ValueError as exc:
                            if str(exc).startswith("Task input snapshot changed"):
                                raise
                            if previous is not None and previous["status"] == "success":
                                raise ValueError("Task input snapshot changed or unavailable; saved success retained") from exc
                            capture.issues.append({"stage": phase, "error_type": type(exc).__name__})
                            status = "failed"
                            logger.error("%s failed for %s at %s: %s", system, ticker, date, type(exc).__name__)
                        except Exception as exc:
                            if previous is not None and previous["status"] == "success":
                                raise ValueError("Task input snapshot changed or unavailable; saved success retained") from exc
                            capture.issues.append({"stage": phase, "error_type": type(exc).__name__})
                            status = "failed"
                            logger.error("%s failed for %s at %s: %s", system, ticker, date, type(exc).__name__)
                    records.append(recorder.save(ticker, date, system, inputs, status=status, row=row, decision=decision, backtest=backtest, audit=audit.to_dict() if audit else None, calls=capture.calls, issues=capture.issues, previous=previous, decision_snapshot=capture.decision_snapshot, timings={"retrieval_latency": retrieval_seconds, "total_latency": time.perf_counter() - started}))
    recorder.finish(records)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "trade_date", "action", "system", "trade_executed", "raw_return", "execution_status", "settled"])
    df.attrs.update(run_id=recorder.run_id, run_dir=str(recorder.path), protocol_version=cfg.protocol_version, evaluation_version=cfg.evaluation_version)
    from src.eval.reporting import export_results
    if cfg.protocol_version == "legacy":
        export_results(df, cfg.output_dir)
    else:
        valid = comparable_rows(df)
        export_results(df, recorder.path, metrics_df=valid, evaluation_version=cfg.evaluation_version)
        if not valid.empty:
            paired_significance_tests(valid, evaluation_version=cfg.evaluation_version).to_csv(recorder.path / "significance_successful_settled.csv", index=False)
            full_summary_by_system(valid, evaluation_version=cfg.evaluation_version).to_csv(recorder.path / "summary_successful_settled.csv")
        from itertools import combinations
        pairs = []
        for a, b in combinations(SYSTEMS, 2):
            keys_a = set(zip(valid.loc[valid.system == a, "ticker"], valid.loc[valid.system == a, "trade_date"]))
            keys_b = set(zip(valid.loc[valid.system == b, "ticker"], valid.loc[valid.system == b, "trade_date"]))
            pairs.append({"system_a": a, "system_b": b, "n_pairs": len(keys_a & keys_b)})
        write_json(recorder.path / "comparison_coverage.json", {"pairs": pairs, "expected_tasks": len(tickers) * len(dates) * 4, "successful_settled_rows": len(valid), "technical_failures": sum(r["status"] != "success" for r in records)})
    return df
