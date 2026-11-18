"""Synthetic regression coverage for frozen forecasts and outcome isolation."""
from copy import deepcopy
import json

import pytest

from src.agents.pm_deterministic import ROLE_WEIGHTS
from src.eval.prediction_review import (
    FrozenForecast, assess_review, labels_for, replay_prediction_review,
    run_prediction_review,
    freeze_research_predictions,
)
from src.eval.records import RunRecorder


def forecast():
    report = {"stance": "bullish", "confidence": 1.0, "score": 0.5,
              "summary": "Revenue grew. The outlook supports UP.", "evidence_refs": ["e1"]}
    return {"ticker": "0700.HK", "date": "2025-01-03",
            "reports": {role: deepcopy(report) for role in ROLE_WEIGHTS},
            "snapshot": {"evidences": [{"ticker": "0700.HK", "evidence_id": "e1",
                          "text": "Revenue grew.", "published_at": "2025-01-03T12:00:00+08:00"}],
                         "claims": []}}


def market():
    return {"sessions": ["2025-01-03", "2025-01-06"], "prices": [
        {"ticker": "0700.HK", "date": "2025-01-03", "close": 100},
        {"ticker": "0700.HK", "date": "2025-01-06", "close": 101}]}


def response():
    return {"units": [{"text": "Revenue grew.", "kind": "factual", "verdict": "supported",
                       "quotes": {"e1": "Revenue grew."}}],
            "citations": [{"evidence_id": "e1", "relevant": True, "quote": "Revenue grew."}],
            "reasoning_direction": "UP", "reasoning_quotes": {"news": "The outlook supports UP."}}


class Judge:
    def __init__(self):
        self.prompts = []
        self.fail = False

    def generate_json(self, system, user):
        self.prompts.append(json.loads(user))
        result = response()
        if self.fail:
            result["units"][0]["quotes"]["e1"] = "Fabricated quote"
        return result


def test_review_end_to_end_replay_and_outcome_isolation(tmp_path):
    judge = Judge()
    payload = forecast()
    original = deepcopy(payload)
    result = run_prediction_review([payload], market(), client=judge, output_dir=tmp_path, run_id="test")
    assert payload == original
    assert result["all_predictions"]["accuracy"] == 1
    assert result["supported_retention"] == 1
    assert replay_prediction_review(result["run_dir"])["rows"] == result["rows"]
    prompt = judge.prompts[0]
    assert set(prompt) == {"ticker", "cutoff", "prediction", "reports", "evidence", "cited_ids", "schema"}
    assert "label" not in prompt and "prices" not in prompt
    assert prompt["prediction"] == {"score": 0.5, "direction": "UP"}
    run_prediction_review([payload], market(), client=judge, output_dir=tmp_path, run_id="test", resume=True)
    assert len(judge.prompts) == 1


def test_failed_review_preserves_direction_and_attempt(tmp_path):
    judge = Judge()
    judge.fail = True
    result = run_prediction_review([forecast()], market(), client=judge, output_dir=tmp_path, run_id="test")
    assert result["n_review_failed"] == 1
    assert result["rows"][0]["direction"] == "UP"
    assert result["all_predictions"]["n"] == 1
    assert result["reviewed_predictions"]["n"] == 0
    replay_prediction_review(result["run_dir"])
    judge.fail = False
    result = run_prediction_review([forecast()], market(), client=judge, output_dir=tmp_path, run_id="test", resume=True)
    assert result["n_review_success"] == 1
    assert len(list((tmp_path / "runs/test/attempts").glob("*.json"))) == 1
    replay_prediction_review(result["run_dir"])


@pytest.mark.parametrize("change", ["future", "scope", "quote", "citation", "empty", "reasoning_quote"])
def test_reject_unavailable_inputs_and_invalid_judgments(change):
    payload, raw = forecast(), response()
    if change in {"future", "scope"}:
        key, value = ("published_at", "2025-01-04T12:00:00+08:00") if change == "future" else ("ticker", "0005.HK")
        payload["snapshot"]["evidences"][0][key] = value
        with pytest.raises(ValueError):
            FrozenForecast.model_validate(payload)
        return
    if change == "quote":
        raw["units"][0]["quotes"] = {"e1": "not present"}
    elif change == "citation":
        raw["citations"] = []
    elif change == "empty":
        raw["units"] = []
    else:
        raw["reasoning_quotes"] = {"news": "not in analyst summary"}
    with pytest.raises(ValueError):
        assess_review(FrozenForecast.model_validate(payload), raw)


def test_conflict_and_insufficient_keep_original_forecast():
    frozen = FrozenForecast.model_validate(forecast())
    raw = response()
    raw["reasoning_direction"] = "DOWN"
    assert assess_review(frozen, raw)["category"] == "conflicting"
    raw["reasoning_direction"] = "mixed"
    assert assess_review(frozen, raw)["category"] == "insufficient"
    assert frozen.prediction()["direction"] == "UP"


def test_calendar_labels_missing_flat_and_year_boundary():
    frozen = FrozenForecast.model_validate(forecast())
    tape = market()
    assert labels_for([frozen], tape)[(frozen.ticker, frozen.date)]["next_session"] == "2025-01-06"
    tape["prices"][1]["close"] = 100
    assert labels_for([frozen], tape)[(frozen.ticker, frozen.date)]["label"] == "DOWN"
    tape["prices"][1]["close"] = None
    assert labels_for([frozen], tape)[(frozen.ticker, frozen.date)]["label"] is None
    payload = forecast()
    payload["date"] = "2025-12-31"
    tape = {"sessions": ["2025-12-31", "2026-01-02"], "prices": [
        {"ticker": frozen.ticker, "date": "2025-12-31", "close": 100},
        {"ticker": frozen.ticker, "date": "2026-01-02", "close": 110}]}
    assert labels_for([FrozenForecast.model_validate(payload)], tape)[(frozen.ticker, "2025-12-31")]["label"] is None


def test_resume_drift_and_replay_tampering(tmp_path):
    run_prediction_review([forecast()], market(), client=Judge(), output_dir=tmp_path, run_id="test")
    changed = market()
    changed["prices"][1]["close"] = 99
    with pytest.raises(ValueError, match="identity changed"):
        run_prediction_review([forecast()], changed, client=Judge(), output_dir=tmp_path, run_id="test", resume=True)
    path = tmp_path / "runs/test/frozen_forecasts.json"
    value = json.loads(path.read_text())
    value[0]["forecast"]["reports"]["news"]["score"] = -1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        replay_prediction_review(path.parent)
    with pytest.raises(ValueError, match="Saved review inputs changed"):
        run_prediction_review([forecast()], market(), client=Judge(), output_dir=tmp_path, run_id="test", resume=True)


def test_missing_evidence_and_duplicate_forecast(tmp_path):
    payload = forecast()
    payload["snapshot"]["evidences"] = []
    for report in payload["reports"].values():
        report["evidence_refs"] = []
    judge = Judge()
    result = run_prediction_review([payload], market(), client=judge, output_dir=tmp_path, run_id="test")
    assert result["n_review_failed"] == 1 and not judge.prompts
    with pytest.raises(ValueError, match="Duplicate forecast"):
        run_prediction_review([payload, payload], market(), client=judge, output_dir=tmp_path, run_id="duplicate")


def test_freeze_saved_analysts_includes_semantic_failures_and_preserves_source(tmp_path):
    item = forecast()
    recorder = RunRecorder(tmp_path, {"research_version": "research_v1", "variants": {"full_graph": {}}}, run_id="source")
    reports = [{"role": role, **report} for role, report in item["reports"].items()]
    recorder.save(item["ticker"], item["date"], "full_graph", {"fixed": True},
                  status="failed", decision={"agent_reports": reports}, decision_snapshot=item["snapshot"],
                  issues=[{"stage": "semantic_grounding", "error_type": "ValueError"}])
    path = recorder.task_path(item["ticker"], item["date"], "full_graph")
    original = path.read_bytes()
    output = tmp_path / "predictions.json"
    assert freeze_research_predictions(recorder.path, output)["exported"] == 1
    saved = json.loads(output.read_text())
    assert FrozenForecast.model_validate(saved[0]).prediction()["direction"] == "UP"
    assert path.read_bytes() == original
    assert json.loads(output.with_suffix(".json.provenance.json").read_text())["tasks"][0]["source_status"] == "failed"
    with pytest.raises(ValueError, match="new file outside"):
        freeze_research_predictions(recorder.path, output)
    damaged = json.loads(path.read_text())
    damaged["inputs"] = {"fixed": False}
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="integrity"):
        freeze_research_predictions(recorder.path, tmp_path / "other.json")
