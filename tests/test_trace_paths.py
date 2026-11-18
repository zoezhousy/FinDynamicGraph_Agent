"""Rule-only API input and stable shared-prefix regressions."""
from copy import deepcopy
import json

import pytest

from src.eval.hsi_daily import analyze, snapshot_for, run_hsi, replay_hsi
from src.eval.protocol import decision_cutoff
from src.eval.prediction_review import FrozenForecast
from src.kg.trace_paths import mine_trace_rules, explore_trace, validate_context
from src.sim.thesis_portfolio import MarketTape
from test_hsi_daily import inputs, Analyst


def examples(label="UP"):
    _, market = inputs()
    tape = MarketTape(market)
    return [(snapshot_for("0700.HK", day, tape, None)[0], decision_cutoff(day), label)
            for day in tape.sessions[:5]]


def test_mine_both_heads_and_duplicate_key_rejected():
    rules, audit = mine_trace_rules(examples())
    assert len(rules) == 1 and rules[0]["head"] == "UP"
    assert rules[0]["support"] == 5 and rules[0]["confidence"] == 1
    assert rules[0]["body_atoms"][0]["end_var"] == rules[0]["body_atoms"][1]["start_var"]
    assert audit["n_examples"] == 5
    down, _ = mine_trace_rules(examples("DOWN"))
    assert down[0]["head"] == "DOWN"
    with pytest.raises(ValueError, match="unique"):
        mine_trace_rules(examples() + examples()[:1])


def test_only_matched_source_quotes_reach_api_and_review():
    rules, _ = mine_trace_rules(examples())
    snapshot, cutoff, _ = examples()[-1]
    unrelated = deepcopy(snapshot["evidences"][0])
    unrelated.update(evidence_id="unrelated", text="SECRET_UNRELATED_TO_SELECTED_PATH")
    snapshot["evidences"].append(unrelated)
    context, audit = explore_trace(snapshot, cutoff, rules)
    assert audit["selected_count"] == 1
    assert "SECRET_UNRELATED" not in json.dumps(context)
    assert all("source_fingerprint" not in item for item in context["evidences"])
    validate_context(snapshot, cutoff, context)
    client = Analyst()
    forecast = analyze("0700.HK", "2024-12-30", snapshot, context,
                       {"status": "no_news", "score": None, "articles": []}, client)
    assert "SECRET_UNRELATED" not in json.dumps(client.inputs)
    assert "SECRET_UNRELATED" not in json.dumps(forecast.review_input())
    damaged = deepcopy(context)
    damaged["evidences"].append(unrelated)
    with pytest.raises(ValueError, match="outside selected paths"):
        validate_context(snapshot, cutoff, damaged)


def test_empty_rules_future_sources_and_quote_failure_have_no_paths():
    snapshot, cutoff, _ = examples()[-1]
    context, audit = explore_trace(snapshot, cutoff, [])
    assert not context["paths"] and audit["status"] == "no_eligible_rule_paths"
    rules, _ = mine_trace_rules(examples())
    future = deepcopy(snapshot)
    future["evidences"][0]["published_at"] = "2099-01-01T12:00:00+08:00"
    assert not explore_trace(future, cutoff, rules)[0]["paths"]
    malformed = deepcopy(snapshot)
    malformed["claims"][0]["quotes"] = {snapshot["evidences"][0]["evidence_id"]: "not a source substring"}
    assert not explore_trace(malformed, cutoff, rules)[0]["paths"]


def test_prompt_system_and_stock_prefix_identical_across_roles():
    class RecordingAnalyst(Analyst):
        def __init__(self):
            super().__init__()
            self.requests = []

        def generate_json_with_budget(self, system, user, *, max_tokens):
            self.requests.append((system, user))
            return super().generate_json_with_budget(system, user, max_tokens=max_tokens)

    rules, _ = mine_trace_rules(examples())
    snapshot, cutoff, _ = examples()[-1]
    context, _ = explore_trace(snapshot, cutoff, rules)
    client = RecordingAnalyst()
    analyze("0700.HK", "2024-12-30", snapshot, context, {"status": "no_news", "score": None}, client)
    assert len({system for system, user in client.requests}) == 1
    common = client.requests[0][1].split(',"role":')[0]
    assert all(user.startswith(common) for _, user in client.requests)
    assert all('"schema"' not in json.loads(user) for _, user in client.requests)
    # Reordered dictionary keys cannot fragment the otherwise identical prefix.
    reordered = {key: context[key] for key in reversed(context)}
    second = RecordingAnalyst()
    analyze("0700.HK", "2024-12-30", snapshot, reordered, {"status": "no_news", "score": None}, second)
    assert second.requests == client.requests
    for _, user in client.requests:
        payload = json.loads(user)
        assert list(payload)[:3] == ["ticker", "cutoff", "input"]
        if payload["role"] == "news":
            assert list(payload)[-1] == "news_sentiment"


def test_trace_daily_generation_and_replay(tmp_path):
    universe, market = inputs()
    result = run_hsi(universe, market, [], client=Analyst(), output_dir=tmp_path, run_id="trace",
                     tickers=["0700.HK"], dates=["2025-01-02"], variants=["full"], retrieval="trace")
    assert result["summary"]["full"]["n_success"] == 1
    assert replay_hsi(result["run_dir"]) == result["summary"]


def test_no_rules_never_calls_analyst(tmp_path):
    universe, market = inputs()
    # Four training examples cannot satisfy the fixed minimum support of five.
    market["sessions"] = market["sessions"][2:]
    market["prices"] = market["prices"][2:]
    client = Analyst()
    result = run_hsi(universe, market, [], client=client, output_dir=tmp_path, run_id="empty",
                     tickers=["0700.HK"], dates=["2025-01-02"], variants=["full"], retrieval="trace")
    assert result["summary"]["full"]["n_success"] == 0
    assert client.inputs == []
    replay_hsi(result["run_dir"])
