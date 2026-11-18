"""Synthetic daily-calendar, frozen-source and live-entry regression tests."""
from copy import deepcopy
import json

import pytest

from src.eval.hsi_data import CONSTITUENTS, technical_source
from src.eval.hsi_daily import run_hsi, replay_hsi, validate_inputs, sentiment_for, snapshot_for
from src.eval.prediction_review import run_prediction_review, replay_prediction_review
from src.eval.thesis_sources import LocalEvidence
from src.sim.thesis_portfolio import MarketTape


def inputs():
    universe = {"source": "official fixture", "published_at": "2024-07-09", "selection": "official_top10_before_test",
                "constituents": [{"ticker": t, "name": n, "weight": w} for t, n, w in CONSTITUENTS]}
    sessions = ["2024-12-20", "2024-12-23", "2024-12-24", "2024-12-27", "2024-12-30", "2024-12-31",
                "2025-01-02", "2025-01-03", "2025-01-06"]
    market = {"sessions": sessions, "prices": [{"ticker": "0700.HK", "date": day, "close": 100 + i,
               "open": 99 + i} for i, day in enumerate(sessions)]}
    return universe, market


class Analyst:
    def __init__(self):
        self.inputs = []
        self.fail = False

    def generate_json(self, system, user):
        payload = json.loads(user)
        self.inputs.append(payload)
        if self.fail:
            raise RuntimeError("fixture provider failure")
        if "reasoning_direction" in str(payload.get("schema")):
            evidence = payload["evidence"][0]
            return {"units": [{"text": "Price observation.", "kind": "factual", "verdict": "supported",
                               "quotes": {evidence["evidence_id"]: evidence["text"]}}],
                    "citations": [{"evidence_id": eid, "relevant": True, "quote": evidence["text"]} for eid in payload["cited_ids"]],
                    "reasoning_direction": "UP", "reasoning_quotes": {"technical": "Source indicates UP."}}
        context = payload["input"]
        report = {"stance": "bullish", "confidence": 0.6, "score": 0.2,
                  "summary": "Source indicates UP.", "evidence_refs": [context["evidences"][0]["evidence_id"]]}
        if "role" not in payload:
            return {"reports": {role: report for role in ("technical", "news", "fundamental", "risk")}}
        return report

    def generate_json_with_budget(self, system, user, *, max_tokens):
        assert max_tokens in (2048, 8192)
        return self.generate_json(system, user)


def test_daily_generation_review_replay_and_resume(tmp_path):
    universe, market = inputs()
    client = Analyst()
    result = run_hsi(universe, market, [], client=client, output_dir=tmp_path, run_id="daily",
                     tickers=["0700.HK"], dates=["2025-01-02", "2025-01-03"])
    assert all(row["n_success"] == 2 for row in result["summary"].values())
    assert replay_hsi(result["run_dir"]) == result["summary"]
    original_calls = len(client.inputs)
    run_hsi(universe, market, [], client=client, output_dir=tmp_path, run_id="daily", resume=True,
            tickers=["0700.HK"], dates=["2025-01-02", "2025-01-03"])
    assert len(client.inputs) == original_calls
    root = tmp_path / "runs/daily"
    paired = json.loads((root / "hsi_paired.json").read_text())
    assert all(row["n_pairs"] == 2 for row in paired.values())
    forecasts = json.loads((root / "full_forecasts.json").read_text())
    assert all(json.loads((path).read_text())["row"]["sentiment"]["status"] in {"no_news", "disabled"}
               for path in (root / "tasks").glob("*.json"))
    review = run_prediction_review(forecasts, market, client=client, output_dir=tmp_path / "reviews", run_id="review")
    assert review["n_review_success"] == 2
    replay_prediction_review(review["run_dir"])
    for payload in client.inputs:
        assert "market" not in payload and "label" not in payload and "prices" not in payload


def test_technical_has_no_future_price_and_missing_session(tmp_path):
    universe, market = inputs()
    source = technical_source("0700.HK", "2025-01-02", MarketTape(market))
    changed = deepcopy(market)
    changed["prices"][-1]["close"] = 999999
    assert source == technical_source("0700.HK", "2025-01-02", MarketTape(changed))
    changed["prices"][6]["close"] = None
    client = Analyst()
    result = run_hsi(universe, changed, [], client=client, output_dir=tmp_path, run_id="missing",
                     tickers=["0700.HK"], dates=["2025-01-02"], variants=["full"])
    assert result["summary"]["full"]["n_success"] == 0
    assert not client.inputs
    replay_hsi(result["run_dir"])


def test_failed_analysts_retry_preserves_attempts_and_drift_refused(tmp_path):
    universe, market = inputs()
    client = Analyst()
    client.fail = True
    kwargs = dict(client=client, output_dir=tmp_path, run_id="retry", tickers=["0700.HK"],
                  dates=["2025-01-02"], variants=["full"])
    run_hsi(universe, market, [], **kwargs)
    client.fail = False
    result = run_hsi(universe, market, [], **kwargs, resume=True)
    assert result["summary"]["full"]["n_success"] == 1
    assert list((tmp_path / "runs/retry/attempts").glob("*.json"))
    changed = deepcopy(market)
    changed["prices"][-1]["close"] = 300
    with pytest.raises(ValueError, match="identity changed"):
        run_hsi(universe, changed, [], **kwargs, resume=True)


def test_scope_and_future_universe_rejected():
    universe, market = inputs()
    universe["published_at"] = "2025-01-02"
    with pytest.raises(ValueError, match="precede"):
        validate_inputs(universe, market, [])


def test_no_news_distinct_from_no_company_background():
    _, market = inputs()
    source = {"ticker": "0700.HK", "source_type": "local_news", "source": "fixture",
              "document_id": "news", "version": "1", "language": "en",
              "published_at": "2025-01-02T12:00:00+08:00", "text": "Tencent announced a new product."}
    sources = LocalEvidence([source])
    from src.eval.protocol import decision_cutoff
    cutoff = decision_cutoff("2025-01-02")
    snapshot, _ = snapshot_for("0700.HK", "2025-01-02", MarketTape(market), sources)
    result = sentiment_for("0700.HK", decision_cutoff("2024-12-31"), cutoff, snapshot, sources, Analyst(), "graph")
    assert result["status"] == "no_background" and result["score"] is None


def test_parallel_batch_replay_retains_missing_stock_tasks(tmp_path):
    from src.eval.hsi_batch import run_batch, replay_batch
    universe, market = inputs()
    result = run_batch(universe, market, [], client_factory=Analyst, output_dir=tmp_path,
                       run_id="batch", workers=2)
    assert result["n_expected"] == 30
    assert result["n_success"] == 3
    assert result["n_failed"] == 27
    assert result["n_unlabeled"] == 1
    assert replay_batch(result["run_dir"])["n_success"] == 3
    retried = run_batch(universe, market, [], client_factory=Analyst, output_dir=tmp_path,
                        run_id="batch", workers=2, resume=True)
    assert retried["n_success"] == 3
