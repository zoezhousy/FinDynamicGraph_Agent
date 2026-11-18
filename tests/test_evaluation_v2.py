"""Versioned evaluation regressions use synthetic evidence and temporary runs."""
import json
import math
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.agents.evidence import evidence_freshness
from src.agents.models import AgentReport
from src.agents.pm_deterministic import portfolio_manager_decide
from src.config import CollectionConfig, ExperimentConfig
from src.eval.grounding import citation_precision, stale_evidence_rate
from src.eval.metrics import directional_accuracy, full_summary_by_system, paired_significance_tests, win_rate
from src.eval.replay import replay_record, replay_run
from src.eval.runner import run_grid


def test_explicit_empty_retrieval_rejects_fake_citation():
    row = {"evidence_refs": ["fake"], "retrieved_evidence_ids": [], "claim_refs": [], "retrieved_claim_ids": []}
    assert citation_precision([row], evaluation_version="v2") == 0
    assert citation_precision([row]) == 1  # historical compatibility


@pytest.mark.parametrize("row", [{"evidence_refs": ["e1"]}, {"evidence_refs": []}, {"evidence_refs": ["e1"], "retrieved_evidence_ids": None}])
def test_unverifiable_citations_are_not_reported_as_perfect(row):
    assert math.isnan(citation_precision([row], evaluation_version="v2"))


def test_citations_must_match_their_reference_type():
    row = {"evidence_refs": ["e1", "c1"], "claim_refs": ["c1"], "retrieved_evidence_ids": '["e1"]', "retrieved_claim_ids": ["c1"]}
    assert citation_precision([row], evaluation_version="v2") == pytest.approx(2 / 3)


@pytest.mark.parametrize(("item", "expected"), [
    ({"published_at": "2025-06-01T08:10:00+00:00"}, "fresh"),
    ({"published_at": "2025-05-02T15:59:59+00:00"}, "fresh"),
    ({"published_at": "2025-05-02T15:59:58+00:00"}, "stale"),
    ({"published_at": "2025-06-01T08:10:00+00:00", "valid_to": "2025-06-01T15:59:59+00:00"}, "stale"),
    ({"published_at": "2025-06-02T00:00:00+00:00"}, "unknown"),
    ({"published_at": "invalid"}, "unknown"),
    ({}, "unknown"),
    ({"published_at": "2025-06-01T16:10:00"}, "fresh"),
])
def test_freshness_boundary_expiry_and_unknown(item, expected):
    graph = {"evidences": [{"evidence_id": "e1", **item}]}
    when = datetime(2025, 6, 1, 15, 59, 59, tzinfo=timezone.utc)
    assert evidence_freshness("e1", graph, when) == expected


def test_unknown_evidence_is_not_counted_as_fresh():
    graph = {"evidences": [{"evidence_id": "e1"}], "claims": []}
    reports = [AgentReport(role="technical", stance="bullish", confidence=.8, score=.8, summary="fixed", evidence_refs=["e1"], claim_refs=[], factors=[])]
    decision = portfolio_manager_decide("TEST", datetime(2025, 6, 1), reports, graph, evaluation_version="v2")
    assert decision["unknown_evidence_count"] == 1
    assert decision["fresh_evidence_count"] == decision["stale_evidence_count"] == 0
    assert math.isnan(stale_evidence_rate([decision], evaluation_version="v2"))
    assert "unknown_evidence_count" not in portfolio_manager_decide("TEST", datetime(2025, 6, 1), reports, graph)


def test_direction_is_separate_from_net_profit():
    trades = pd.DataFrame([{"action": "buy", "trade_executed": True, "gross_return": .0005, "raw_return": -.0005}])
    assert directional_accuracy(trades, evaluation_version="v2") == 1
    assert win_rate(trades) == directional_accuracy(trades) == 0
    with pytest.raises(ValueError, match="gross_return"):
        directional_accuracy(trades.drop(columns="gross_return"), evaluation_version="v2")


def pair_rows():
    rows = []
    for day in range(1, 7):
        for system, value in (("a", .02), ("b", -.01)):
            rows.append({"ticker": "TEST", "trade_date": f"2025-06-{day:02d}", "system": system, "action": "buy", "trade_executed": True, "raw_return": value, "gross_return": value + .001})
    rows[-1].update(action="hold", trade_executed=False, raw_return=0, gross_return=0)
    return pd.DataFrame(rows)


def test_each_test_reports_its_actual_sample_and_adjusted_pvalue():
    result = paired_significance_tests(pair_rows(), evaluation_version="v2").set_index("test")
    assert result.loc["wilcoxon_raw_return", "n_pairs"] == 6
    assert result.loc["mcnemar_directional", "n_pairs"] == 5
    assert result.loc["mcnemar_directional", "n_discordant"] == 5
    assert result.loc["wilcoxon_raw_return", "effect"] == pytest.approx(.17 / 6)
    assert result.loc["mcnemar_directional", "effect"] == 1
    ordered = result.sort_values("p_value")
    expected_first = min(1, 2 * ordered.iloc[0].p_value)
    assert ordered.iloc[0].p_value_adjusted == pytest.approx(expected_first)
    assert ordered.iloc[1].p_value_adjusted == pytest.approx(max(expected_first, ordered.iloc[1].p_value))


def test_duplicate_pairs_and_mixed_versions_are_rejected():
    rows = pair_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        paired_significance_tests(pd.concat([rows, rows.iloc[:1]]), evaluation_version="v2")
    rows["evaluation_version"] = ["v2", "legacy"] * 6
    with pytest.raises(ValueError, match="mix evaluation"):
        full_summary_by_system(rows, evaluation_version="v2")


def test_no_trades_has_no_directional_accuracy_or_mcnemar_sample():
    rows = pair_rows().assign(trade_executed=False, raw_return=0, gross_return=0)
    assert math.isnan(directional_accuracy(rows, evaluation_version="v2"))
    result = paired_significance_tests(rows, evaluation_version="v2")
    assert list(result.test) == ["wilcoxon_raw_return"]
    assert result.iloc[0].n_pairs == 6
    assert math.isnan(result.iloc[0].p_value_adjusted)


def test_v2_run_records_reports_replay_and_resume_version(tmp_path):
    class Query:
        def load_subgraph(self, ticker, when):
            return {"signals": [], "claims": [], "evidences": [{"evidence_id": "e1", "published_at": "2025-01-01T08:00:00+00:00"}]}
    class Orchestrator:
        def __init__(self, context, config, **kwargs):
            self.context, self.config = context, config
        def close(self):
            pass
        def run_for_ticker(self, ticker, when):
            reports = [AgentReport(role="technical", stance="bullish", confidence=.8, score=.8, summary="fixed", evidence_refs=["e1"], claim_refs=[], factors=[])]
            return portfolio_manager_decide(ticker, when, reports, self.context.load_subgraph(ticker, when), evaluation_version=self.config.evaluation_version)
    class Model:
        def generate_json(self, *args):
            return {"action": "buy", "stance": "bullish", "score": .8, "confidence": .8, "summary": "fixed", "factors": []}
    ohlcv = pd.DataFrame({"date": pd.bdate_range("2025-06-02", periods=7), "open": 100., "close": 100.05})
    kwargs = dict(query_client=Query(), persist_outcomes=False, model_client=Model(), orchestrator_factory=Orchestrator, ohlcv_loader=lambda *args: ohlcv)
    cfg = CollectionConfig(output_root=tmp_path / "inputs")
    exp = ExperimentConfig(protocol_version="pit_v1", evaluation_version="v2", output_dir=tmp_path, run_id="new")
    df = run_grid(["TEST"], ["2025-06-01"], cfg, experiment_config=exp, **kwargs)
    assert set(df.evaluation_version) == {"v2"}
    run_dir = tmp_path / "runs" / "new"
    summary = pd.read_csv(run_dir / "results_summary_latest.csv").set_index("system")
    assert summary.loc["kg_dynamic", "directional_accuracy"] == 1
    assert summary.loc["kg_dynamic", "win_rate"] == 0
    assert summary.loc["kg_dynamic", "stale_evidence_rate"] == 1
    assert math.isnan(summary.loc["evidence_no_kg", "citation_precision"])
    for path in (run_dir / "tasks").glob("*.json"):
        record = json.loads(path.read_text())
        assert record["inputs"]["evaluation_version"] == "v2"
        replayed = replay_record(record)
        assert replayed["backtest"] == record["backtest"]
        if record["system"] in {"kg_dynamic", "static_kg"}:
            assert replayed["decision"]["stale_evidence_count"] == record["decision"]["stale_evidence_count"]
    assert replay_run(run_dir).attrs["evaluation_version"] == "v2"
    with pytest.raises(ValueError, match="identity changed"):
        old = ExperimentConfig(protocol_version="pit_v1", output_dir=tmp_path, run_id="new", resume=True)
        run_grid(["TEST"], ["2025-06-01"], cfg, experiment_config=old, **kwargs)


def test_v2_requires_explicit_pit_protocol():
    with pytest.raises(ValueError, match="requires pit_v1"):
        ExperimentConfig(evaluation_version="v2")


def test_export_rejects_mixed_versions_before_writing(tmp_path):
    from src.eval.reporting import export_results
    rows = pair_rows()
    rows["evaluation_version"] = ["legacy", "v2"] * 6
    target = tmp_path / "report"
    with pytest.raises(ValueError, match="mix evaluation"):
        export_results(rows, target, evaluation_version="v2")
    assert not target.exists()


def test_record_without_evaluation_version_replays_legacy_freshness():
    from src.eval.records import digest
    when = datetime(2025, 6, 1, 15, 59, 59, tzinfo=timezone.utc)
    graph = {"evidences": [{"evidence_id": "e1", "published_at": "2025-06-01T08:10:00+00:00"}], "claims": []}
    reports = [AgentReport(role="technical", stance="bullish", confidence=.8, score=.8, summary="fixed", evidence_refs=["e1"], claim_refs=[], factors=[])]
    decision = portfolio_manager_decide("TEST", when, reports, graph)
    inputs = {"protocol_version": "pit_v1", "decision_time": when.isoformat(), "snapshot": graph, "backtest_config": {"holding_days": 5, "transaction_cost_bp": 5.}, "ohlcv": [{"date": f"2025-06-{day:02d}", "open": 100., "close": 102.} for day in range(2, 7)]}
    record = {"inputs": inputs, "input_digest": digest(inputs), "status": "success", "system": "kg_dynamic", "ticker": "TEST", "trade_date": "2025-06-01", "decision": decision, "run_id": "old"}
    replayed = replay_record(record)["decision"]
    assert replayed["stale_evidence_count"] == decision["stale_evidence_count"] == 1
    assert "evaluation_version" not in replayed
    assert portfolio_manager_decide("TEST", when, reports, graph, evaluation_version="v2")["stale_evidence_count"] == 0
