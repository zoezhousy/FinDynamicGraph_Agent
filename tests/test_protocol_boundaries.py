from datetime import datetime, timezone
from types import SimpleNamespace
import asyncio
import pandas as pd
import pytest
from src.eval.protocol import decision_cutoff, strict_snapshot, FrozenSnapshotClient, backtest_details
from src.eval.execution import capture_execution
from src.agents.kg_tools import build_kg_tools, DynamicDatePolicy, KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator, OrchestratorConfig
from src.sim.backtest import BacktestConfig

def test_hong_kong_date_has_exact_utc_cutoff():
    cutoff = decision_cutoff("2025-06-01")
    assert cutoff == datetime(2025, 6, 1, 15, 59, 59, 999999, tzinfo=timezone.utc)
    assert decision_cutoff("2025-06-01T10:00:00+08:00").hour == 2

def test_unknown_future_expired_and_revised_records_are_excluded():
    graph = {"evidences": [{"evidence_id": "before", "published_at": "2025-06-01T23:59:59+08:00"}, {"evidence_id": "after", "published_at": "2025-06-02T00:00:00+08:00"}, {"evidence_id": "unknown"}, {"evidence_id": "invalid", "published_at": "not-a-time"}, {"evidence_id": "revision", "published_at": "2025-05-01", "effective_at": "2025-06-03"}, {"evidence_id": "expired", "published_at": "2025-05-01", "valid_to": "2025-05-31"}]}
    result = strict_snapshot(graph, decision_cutoff("2025-06-01"))
    assert [r["evidence_id"] for r in result["evidences"]] == ["before"]
    assert len(graph["evidences"]) == 6

def test_bound_tools_share_input_and_reject_out_of_scope_requests():
    cutoff = decision_cutoff("2025-06-01")
    graph = {"signals": [{"entity_id": "s1"}]}
    frozen = FrozenSnapshotClient("0700.HK", cutoff, graph)
    tools = {t.name: t for t in build_kg_tools(frozen, DynamicDatePolicy(), bound_ticker="0700.HK", bound_date=cutoff, snapshot=frozen.snapshot)}
    good = {"ticker": "0700.HK", "as_of_date": "2025-06-01"}
    result = tools["query_signals"].invoke(good)
    result[0]["entity_id"] = "mutated"
    assert tools["query_subgraph"].invoke(good)["signals"][0]["entity_id"] == "s1"
    with capture_execution() as capture:
        with pytest.raises(ValueError):
            tools["query_signals"].invoke({**good, "ticker": "0005.HK"})
        with pytest.raises(ValueError):
            tools["query_signals"].invoke({**good, "as_of_date": "2025-06-02"})
    assert capture.status == "degraded"
    assert len(capture.issues) == 2

def test_legacy_tool_cache_is_scoped_to_one_builder():
    calls = []
    query = SimpleNamespace(load_subgraph=lambda *a: calls.append(a) or {"signals": []})
    for _ in range(2):
        tools = build_kg_tools(query, DynamicDatePolicy())
        tools[0].invoke({"ticker": "0700.HK", "as_of_date": "2025-06-01"})
        tools[1].invoke({"ticker": "0700.HK", "as_of_date": "2025-06-01"})
    assert len(calls) == 2

def test_incomplete_window_is_unsettled_only_in_pit_protocol():
    bars = pd.DataFrame({"date": pd.date_range("2025-06-02", periods=2), "open": [100., 101.], "close": [101., 102.]})
    legacy = backtest_details(bars, "2025-06-01", "buy", BacktestConfig())
    strict = backtest_details(bars, "2025-06-01", "buy", BacktestConfig(), protocol_version="pit_v1")
    assert legacy["raw_return"] == pytest.approx(.019)
    assert strict["raw_return"] is None
    assert not strict["settled"]
    assert not strict["window_complete"]
    assert not backtest_details(bars, "2025-07-01", "buy", BacktestConfig(), protocol_version="pit_v1")["settled"]

def test_agent_failure_has_neutral_payload_and_separate_diagnostic():
    orchestrator = object.__new__(KGBasedOrchestrator)
    orchestrator.semaphore = asyncio.Semaphore(2)
    async def fail(data):
        raise TimeoutError("synthetic")
    with capture_execution() as capture:
        result = asyncio.run(orchestrator._run_agent_safely(SimpleNamespace(ainvoke=fail), {}, "news", "0700.HK"))
    assert result.stance == "uncertain"
    assert result.confidence == 0
    assert capture.status == "degraded"
    assert capture.issues[0]["error_type"] == "TimeoutError"

def test_orchestrator_does_not_own_injected_trace_store():
    calls = []
    store = SimpleNamespace(close=lambda: calls.append("close"))
    instance = KGBasedOrchestrator(KGAgentContext(SimpleNamespace()), OrchestratorConfig(persist_decision_trace=False), backend="legacy", trace_store=store)
    instance.close()
    instance.close()
    assert not calls


def test_langgraph_analysts_pm_and_audit_use_same_frozen_snapshot(monkeypatch):
    import json
    import src.agents.langgraph_agents as factories
    from src.agents.models import AgentReport
    seen = []
    cutoff = decision_cutoff("2025-06-01")
    graph = {"signals": [{"entity_id": "s1", "direction": "bullish", "as_of_date": "2025-05-31"}], "evidences": [], "claims": []}
    frozen = FrozenSnapshotClient("0700.HK", cutoff, graph)
    def factory(llm, tools):
        async def invoke(data):
            value = tools[1].invoke({"ticker": "0700.HK", "as_of_date": "2025-06-01"})
            seen.append(value)
            payload = {"stance": "bullish", "confidence": .8, "score": .8, "summary": "fixed", "evidence_refs": ["s1"], "claim_refs": [], "factors": []}
            return {"messages": [SimpleNamespace(content=json.dumps(payload))]}
        return SimpleNamespace(ainvoke=invoke)
    for name in ("create_news_agent", "create_technical_agent", "create_fundamental_agent", "create_risk_agent"):
        monkeypatch.setattr(factories, name, factory)
    instance = KGBasedOrchestrator(KGAgentContext(frozen), OrchestratorConfig(persist_decision_trace=False), backend="langgraph", llm=object(), trace_store=None)
    with capture_execution() as capture:
        result = instance.run_for_ticker("0700.HK", cutoff)
    assert len(seen) == 4
    assert all(value == graph["signals"] for value in seen)
    assert result["retrieved_evidence_ids"] == ["s1"]
    assert capture.status == "success"
    assert len(capture.calls) == 4

def test_orchestrator_backend_initialization_failure_closes_owned_store(monkeypatch):
    import src.agents.orchestrator as module
    closed = []
    store = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(module.KGBasedOrchestrator, "_init_trace_store", lambda self: store)
    def fail(self):
        raise ValueError("synthetic backend failure")
    monkeypatch.setattr(module.KGBasedOrchestrator, "_init_langgraph", fail)
    with pytest.raises(ValueError):
        module.KGBasedOrchestrator(KGAgentContext(SimpleNamespace()), backend="langgraph")
    assert closed == [True]


def test_missing_pandas_times_and_conflict_endpoints():
    from src.eval.protocol import available_time, eligible_record
    cutoff = decision_cutoff("2025-06-01")
    assert available_time(pd.NA) is None
    assert available_time(pd.NaT) is None
    assert not eligible_record({"published_at": pd.NA}, cutoff)
    graph = {"claims": [{"claim_id": "a", "published_at": "2025-05-01"},
                        {"claim_id": "b", "published_at": "2025-05-02"},
                        {"claim_id": "future", "published_at": "2025-06-02"}],
             "conflicts": [{"claim_a": "a", "claim_b": "b"},
                           {"claim_a": "a", "claim_b": "future"}]}
    result = strict_snapshot(graph, cutoff)
    assert result["conflicts"] == [{"claim_a": "a", "claim_b": "b"}]
