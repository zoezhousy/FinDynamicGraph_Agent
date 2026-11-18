import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from types import SimpleNamespace
import pandas as pd
import pytest
from src.eval.records import RunRecorder, digest, write_json
from src.eval.replay import replay_record
from src.eval.execution import record_call
from src.eval.runner import run_grid
from src.config import CollectionConfig, ExperimentConfig

class Query:
    def __init__(self):
        self.calls = []
        self.closed = 0
        self.version = 1
    def load_subgraph(self, ticker, when):
        self.calls.append((ticker, when))
        return {"company": [{"ticker": ticker}], "signals": [{"entity_id": f"s{self.version}", "direction": "bullish", "as_of_date": "2025-01-01"}], "evidences": [{"evidence_id": "ev1", "published_at": "2025-01-01"}], "claims": []}
    def close(self):
        self.closed += 1

class Store:
    def __init__(self):
        self.closed = 0
        self.outcomes = []
        self.traces = []
    def init_constraints(self):
        pass
    def upsert_backtest_outcome(self, outcome):
        self.outcomes.append(outcome)
    def upsert_decision_trace(self, trace, assessments):
        self.traces.append(trace)
    def close(self):
        self.closed += 1

class Model:
    def __init__(self):
        self.calls = 0
    def generate_json(self, system, user):
        self.calls += 1
        payload = {"action": "buy", "stance": "bullish", "confidence": .8, "score": .8, "summary": "fixed", "reason": "fixed", "evidence_refs": ["ev1"], "factors": []}
        record_call({"system_prompt": system, "user_prompt": user, "parsed_response": payload, "raw_response": json.dumps(payload), "retry_count": 0})
        return payload

def bars(count=7):
    return pd.DataFrame({"date": pd.bdate_range("2025-06-02", periods=count), "open": [100.] * count, "close": [102.] * count})

def run(tmp_path, monkeypatch, *, run_id="sample", resume=False, query=None, model=None, data=None, protocol="pit_v1", runners=None):
    monkeypatch.setenv("AGENT_BACKEND", "legacy")
    query, model, store = query or Query(), model or Model(), Store()
    df = run_grid(["0700.HK"], ["2025-06-01"], CollectionConfig(output_root=tmp_path / "inputs"), experiment_config=ExperimentConfig(output_dir=tmp_path, protocol_version=protocol, run_id=run_id, resume=resume), query_client=query, store=store, model_client=model, runners=runners, ohlcv_loader=lambda *a: bars() if data is None else data)
    return df, query, store, model

def task_records(tmp_path, run_id="sample"):
    return [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "runs" / run_id / "tasks").glob("*.json")]

def test_run_has_four_tasks_shared_snapshots_and_injected_ownership(tmp_path, monkeypatch):
    df, query, store, model = run(tmp_path, monkeypatch)
    assert len(df) == 4
    assert set(df.system) == {"kg_dynamic", "static_kg", "no_kg_no_evidence", "evidence_no_kg"}
    assert set(df.execution_status) == {"success"}
    assert len(query.calls) == 2
    assert query.closed == store.closed == 0
    assert store.traces[0].decision_id.endswith(":sample")
    assert store.outcomes[0].outcome_id.endswith(":sample")
    records = task_records(tmp_path)
    assert len(records) == 4
    assert all(r["input_digest"] == digest(r["inputs"]) for r in records)
    assert all(r["calls"] for r in records)
    assert (tmp_path / "runs" / "sample" / "comparison_coverage.json").exists()
    assert not (tmp_path / "trades_latest.csv").exists()

def test_resume_skips_success_and_rejects_snapshot_change(tmp_path, monkeypatch):
    df, query, store, model = run(tmp_path, monkeypatch)
    count = model.calls
    resumed, *_ = run(tmp_path, monkeypatch, resume=True, query=query, model=model)
    assert model.calls == count
    assert len(resumed) == 4
    assert all(r["attempt"] == 1 for r in task_records(tmp_path))
    query.version = 2
    with pytest.raises(ValueError, match="snapshot changed"):
        run(tmp_path, monkeypatch, resume=True, query=query, model=model)

def test_resume_rejects_changed_model_identity(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "different-model")
    with pytest.raises(ValueError, match="identity changed"):
        run(tmp_path, monkeypatch, resume=True)

def test_failures_are_retried_without_repeating_successful_systems(tmp_path, monkeypatch):
    def failed(*args):
        raise TimeoutError("synthetic")
    df, query, store, model = run(tmp_path, monkeypatch, runners={"no_kg_no_evidence": failed})
    assert len(df) == 3
    assert [r["status"] for r in task_records(tmp_path)].count("failed") == 1
    resumed, *_ = run(tmp_path, monkeypatch, resume=True, query=query, model=model)
    assert len(resumed) == 4
    attempts = {r["system"]: r["attempt"] for r in task_records(tmp_path)}
    assert attempts["no_kg_no_evidence"] == 2
    assert attempts["kg_dynamic"] == 1

def test_baseline_fallback_is_degraded_not_a_success(tmp_path, monkeypatch):
    class FailedModel:
        def generate_json(self, *args):
            raise TimeoutError("synthetic")
    df, *_ = run(tmp_path, monkeypatch, model=FailedModel())
    records = {r["system"]: r for r in task_records(tmp_path)}
    assert records["no_kg_no_evidence"]["status"] == "degraded"
    assert records["no_kg_no_evidence"]["decision"]["action"] == "abstain"
    assert records["evidence_no_kg"]["status"] == "degraded"
    coverage = json.loads((tmp_path / "runs" / "sample" / "comparison_coverage.json").read_text())
    assert coverage["successful_settled_rows"] == 0

def test_replay_needs_no_clients_and_preserves_deterministic_scores(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    for record in task_records(tmp_path):
        result = replay_record(record)
        for field in ("action", "final_score", "confidence", "conflict_level", "evidence_refs"):
            assert result["decision"][field] == record["decision"][field]
        assert result["backtest"] == record["backtest"]
    corrupted = task_records(tmp_path)[0]
    corrupted["inputs"]["ohlcv"][0]["open"] = 999.
    with pytest.raises(ValueError, match="digest"):
        replay_record(corrupted)

def test_legacy_output_does_not_gain_protocol_columns(tmp_path, monkeypatch):
    df, *_ = run(tmp_path, monkeypatch, protocol="legacy")
    assert len(df) == 4
    assert "execution_status" not in df.columns
    assert "settled" not in df.columns
    assert (tmp_path / "trades_latest.csv").exists()
    assert len(task_records(tmp_path)) == 4

def test_unsettled_window_is_excluded_from_comparisons(tmp_path, monkeypatch):
    df, *_ = run(tmp_path, monkeypatch, data=bars(2))
    assert df.raw_return.isna().all()
    assert not df.settled.any()
    coverage = json.loads((tmp_path / "runs" / "sample" / "comparison_coverage.json").read_text())
    assert coverage["successful_settled_rows"] == 0
    assert all(p["n_pairs"] == 0 for p in coverage["pairs"])

def test_invalid_run_path_and_configuration_changes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        RunRecorder(tmp_path, {}, run_id="../escape")
    RunRecorder(tmp_path, {"protocol": "legacy"}, run_id="fixed")
    with pytest.raises(FileExistsError):
        RunRecorder(tmp_path, {}, run_id="fixed")
    with pytest.raises(ValueError, match="identity changed"):
        RunRecorder(tmp_path, {"protocol": "pit_v1"}, run_id="fixed", resume=True)


def test_input_loading_failures_record_all_tasks_and_can_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "legacy")
    query, model = Query(), Model()
    cfg = CollectionConfig(output_root=tmp_path / "inputs")
    exp = ExperimentConfig(output_dir=tmp_path, protocol_version="pit_v1", run_id="sample")
    def fail(*args):
        raise FileNotFoundError("synthetic missing input")
    df = run_grid(["0700.HK"], ["2025-06-01"], cfg, experiment_config=exp,
                  query_client=query, store=Store(), model_client=model, ohlcv_loader=fail)
    assert df.empty
    assert len(task_records(tmp_path)) == 4
    assert all(r["status"] == "failed" for r in task_records(tmp_path))
    resumed, *_ = run(tmp_path, monkeypatch, resume=True, query=query, model=model)
    assert len(resumed) == 4
    assert all(r["attempt"] == 2 for r in task_records(tmp_path))
    assert len(list((tmp_path / "runs" / "sample" / "attempts").glob("*.json"))) == 4


def test_preliminary_pit_keeps_schema_and_fixed_static_snapshot(tmp_path, monkeypatch):
    from src.eval.run_preliminary import run_preliminary, RESULT_COLUMNS
    monkeypatch.setenv("AGENT_BACKEND", "legacy")
    class ChangingQuery(Query):
        def load_subgraph(self, ticker, when):
            graph = super().load_subgraph(ticker, when)
            graph["signals"][0]["entity_id"] = when.date().isoformat()
            return graph
    root = tmp_path / "inputs"
    (root / "0700.HK").mkdir(parents=True)
    bars(10).to_parquet(root / "0700.HK" / "ohlcv_2021_now.parquet", index=False)
    query = ChangingQuery()
    df = run_preliminary(ticker="0700.HK", trade_dates=[datetime(2025, 6, 1), datetime(2025, 6, 3)],
        data_root=root, output_dir=tmp_path / "out", static_cutoff=datetime(2025, 1, 1),
        query_client=query, model_client=Model(), experiment_config=ExperimentConfig(protocol_version="pit_v1", run_id="focused"))
    assert list(df.columns) == list(RESULT_COLUMNS)
    assert len(df) == 8
    assert df.future_count.eq(0).all()
    assert df.loc[df.system == "static_kg", "snapshot_hash"].nunique() == 1
    assert df.loc[df.system == "kg_dynamic", "snapshot_hash"].nunique() == 2
    assert len(query.calls) == 3
    assert query.closed == 0


def test_replay_validates_actual_aggregation_snapshot(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    record = next(r for r in task_records(tmp_path) if r["system"] == "kg_dynamic")
    assert record["decision_snapshot"] == record["inputs"]["snapshot"]
    record["decision_snapshot"]["signals"][0]["entity_id"] = "corrupted"
    with pytest.raises(ValueError, match="snapshot digest"):
        replay_record(record)


def test_direct_client_captures_raw_response_retries_without_credentials(monkeypatch):
    from src.llm.client import LLMConfig, OpenAICompatibleClient
    from src.eval.execution import capture_execution
    import requests
    calls = []
    raw = '```json\n{"action": "buy"}\n```'
    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("synthetic")
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": raw}}]})
    monkeypatch.setattr("src.llm.client.requests.post", post)
    monkeypatch.setattr("src.llm.client.time.sleep", lambda *args: None)
    client = OpenAICompatibleClient(LLMConfig("https://example.invalid", "synthetic-test-key", "fake-model"))
    with capture_execution() as capture:
        assert client.generate_json("system", "user") == {"action": "buy"}
    assert capture.calls[0]["raw_response"] == raw
    assert capture.calls[0]["retry_count"] == 1
    assert "synthetic-test-key" not in json.dumps(capture.calls)


def test_repeated_input_failure_keeps_attempt_history_and_failure_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BACKEND", "legacy")
    def fail(*args):
        raise FileNotFoundError("synthetic")
    for resume in (False, True):
        run_grid(["0700.HK"], ["2025-06-01"], CollectionConfig(output_root=tmp_path / "inputs"),
            experiment_config=ExperimentConfig(output_dir=tmp_path, protocol_version="pit_v1", run_id="sample", resume=resume),
            query_client=Query(), store=Store(), model_client=Model(), ohlcv_loader=fail)
    records = task_records(tmp_path)
    assert all(r["attempt"] == 2 for r in records)
    assert all(r["issues"][0]["stage"] == "input_loading" for r in records)
    summary = json.loads((tmp_path / "runs" / "sample" / "run_summary.json").read_text())
    assert summary["technical_failure_rate"] == 1.0
    assert len(list((tmp_path / "runs" / "sample" / "attempts").glob("*.json"))) == 4


def test_resume_does_not_overwrite_success_when_input_is_unavailable(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch)
    before = task_records(tmp_path)
    def fail(*args):
        raise FileNotFoundError("synthetic")
    with pytest.raises(ValueError, match="saved success retained"):
        run_grid(["0700.HK"], ["2025-06-01"], CollectionConfig(output_root=tmp_path / "inputs"),
            experiment_config=ExperimentConfig(output_dir=tmp_path, protocol_version="pit_v1", run_id="sample", resume=True),
            query_client=Query(), store=Store(), model_client=Model(), ohlcv_loader=fail)
    assert before == task_records(tmp_path)
