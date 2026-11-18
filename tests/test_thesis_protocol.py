from datetime import date, datetime, timedelta, timezone
from math import log

import pytest

from src.eval.protocol import decision_cutoff, eligible_record
from src.eval.records import digest
from src.eval.thesis import replay_thesis, run_thesis
from src.eval.thesis_sources import LocalEvidence
from src.kg.collaboration import GraphLedger, CollaborativeUpdater
from src.kg.thesis_paths import freeze_graph, mine_rules, retrieve_paths
from src.llm.thesis_scoring import parse_classification, score_logprobs
from src.sim.thesis_portfolio import MarketTape, settle_portfolios

TICKERS = ["0005.HK", "0700.HK", "1299.HK"]


def source(ticker="0700.HK", kind="southbound_holding", **overrides):
    data = {"ticker": ticker, "source_type": kind, "source": "local source",
            "document_id": "holding-2025-01", "version": "1", "language": "zh-Hant",
            "published_at": "2025-01-01T12:00:00+08:00", "text": "Holdings were 100 shares.",
            "period": "2025-01", "value": 100, "unit": "shares"}
    data.update(overrides)
    return data


class Proposer:
    def generate_json(self, system, user):
        return {"proposals": []}


class Analyst:
    def generate_json_with_budget(self, system, user, max_tokens):
        import json
        evidence = json.loads(user)["input"]["evidences"][0]["evidence_id"]
        return {"stance": "neutral", "confidence": 0.8, "score": 0, "summary": "Synthetic",
                "evidence_refs": [evidence]}


def snapshot(source_rows, cutoff):
    raw = LocalEvidence(source_rows).snapshot("0700.HK", cutoff)
    ledger = GraphLedger("0700.HK", strict_cutoff=True)
    CollaborativeUpdater(Proposer(), None).update(ledger, raw, cutoff)
    return freeze_graph(ledger, raw, cutoff, limit=40)


def trading_days(n=24):
    start = date(2025, 1, 2)
    return [(start + timedelta(days=i)).isoformat() for i in range(n * 2)
            if (start + timedelta(days=i)).weekday() < 5][:n]


def market_data(days):
    return {"sessions": days, "prices": [
        {"ticker": ticker, "date": day, "open": 100, "close": 103 - 2 * i}
        for day in days for i, ticker in enumerate(TICKERS)
    ]}


def test_import_revisions_expiry_and_strict_cutoff():
    instant = datetime(2025, 1, 2, 4, tzinfo=timezone.utc)
    rows = [source(), source(version="2", published_at=instant.isoformat(), text="Holdings were 101 shares.",
                            value=101, valid_to="2025-01-03T12:00:00+08:00"),
            source(document_id="unknown", published_at=None)]
    imported = LocalEvidence(rows)
    assert len(imported.rejected) == 1
    before = imported.snapshot("0700.HK", instant)["evidences"]
    after = imported.snapshot("0700.HK", instant + timedelta(seconds=1))["evidences"]
    assert before[0]["version"] == "1" and after[0]["version"] == "2"
    assert before[0]["evidence_id"] != after[0]["evidence_id"]
    assert imported.snapshot("0700.HK", decision_cutoff("2025-01-03"))["evidences"] == []
    assert eligible_record(after[0], instant)
    assert not eligible_record(after[0], instant, strict=True)


def test_hk_paths_have_exact_quotes_and_training_support():
    cutoff = decision_cutoff("2025-01-04")
    graph = snapshot([source(), source(kind="ah_premium", document_id="premium", a_share="600001.SS",
                                      unit="percent", value=12, text="Premium was 12 percent.")], cutoff)
    assert {x["type"] for x in graph["relations"]} >= {"AH_PREMIUM", "SOUTHBOUND_HOLDING"}
    training = [(graph, cutoff, label) for label in ("UP", "UP", "UP", "DOWN", "DOWN")]
    assert not mine_rules(training[:4])
    rules = mine_rules(training)
    assert {len(rule["body"]) for rule in rules} == {2, 3}
    assert all(rule["support"] == 5 and rule["confidence"] == .6 for rule in rules)
    ctx, audit = retrieve_paths(graph, cutoff, rules)
    assert audit["matched_count"] == 2
    assert {p["edges"][-1]["quote"] for p in ctx["paths"]} == {
        "Holdings were 100 shares.", "Premium was 12 percent.",
    }
    # Any edge/content at or after the effective cutoff is excluded at traversal time.
    earlier = decision_cutoff("2024-12-31")
    assert retrieve_paths(graph, earlier, rules)[0]["paths"] == []


def test_class_probabilities_require_three_single_token_candidates():
    items = [{"token": token, "bytes": [ord(token)], "logprob": log(p)}
             for token, p in zip("ABC", (.6, .3, .1))]
    response = {"choices": [{"finish_reason": "length", "message": {"content": "A"},
                             "logprobs": {"content": [{**items[0], "top_logprobs": items}]}}]}
    assert score_logprobs(parse_classification(response)) == pytest.approx(.5)
    items.pop()
    with pytest.raises(ValueError, match="All three"):
        parse_classification(response)


def test_portfolio_five_sessions_long_short_and_missing_entry():
    days = trading_days(12)
    tasks = {(t, day): {"status": "success", "score": 1, "gate": None}
             for day in days[:2] for t in TICKERS}
    data = market_data(days)
    independent = settle_portfolios(tasks, TICKERS, days[:2], MarketTape(data))
    row = independent[0]
    assert row["entry_date"] == days[1] and row["exit_date"] == days[5]
    assert [(p["ticker"], p["target_weight"]) for p in row["positions"]] == [(TICKERS[0], .5), (TICKERS[2], -.5)]
    assert row["net_return"] == pytest.approx(.019)
    rolling = settle_portfolios(tasks, TICKERS, days[:2], MarketTape(data), "rolling")
    assert rolling[0]["net_return"] == pytest.approx(.019 / 5)
    assert rolling[1]["entry_date"] < rolling[0]["exit_date"]
    next(row for row in data["prices"] if row["ticker"] == TICKERS[2] and row["date"] == days[1])["open"] = None
    missing = settle_portfolios(tasks, TICKERS, days[:2], MarketTape(data))[0]
    assert missing["status"] == "unsettled" and missing["unsettled_reason"] == "missing_entry"
    assert all(p["net_pnl"] is None for p in missing["positions"])
    del tasks[(TICKERS[2], days[1])]
    assert settle_portfolios(tasks, TICKERS, days[:2], MarketTape(data))[1]["status"] == "selection_failed"


def test_thesis_end_to_end_and_offline_replay(tmp_path, monkeypatch, capsys):
    import json
    days = trading_days()
    evidence = [source(ticker=t) for t in TICKERS]
    market = market_data(days)

    def classify(system, user):
        ticker = json.loads(user)["ticker"]
        p = {TICKERS[0]: .7, TICKERS[1]: .5, TICKERS[2]: .2}[ticker]
        return {"positive": log(p), "neutral": log(.1), "negative": log(.9 - p)}

    path, summary = run_thesis(
        evidence, market, tickers=TICKERS, train_range=(days[0], days[9]),
        eval_range=(days[11], days[15]), output_dir=tmp_path, analyst=Analyst(),
        classifier=classify, proposer=Proposer(), verifier=None,
        source_hashes={"evidence_file": digest(evidence), "market_file": digest(market)},
    )
    # Only train days 0 through 5 have a fifth-session close strictly before day 11.
    assert summary["training_labeled"] == 18
    assert summary["rules"] == 1 and summary["settled_batches"] == 1
    assert summary["mean_settled_net_return"] == pytest.approx(.019)
    from src.eval.run_research import main
    monkeypatch.setattr("sys.argv", ["run_research", "thesis-replay", str(path)])
    main()
    assert json.loads(capsys.readouterr().out)["settled_batches"] == 1

    task = path / "tasks" / (digest([TICKERS[0], days[11], "thesis_v1"]) + ".json")
    record = json.loads(task.read_text(encoding="utf-8"))
    record["row"]["analyses"]["technical"]["class_logprobs"]["positive"] = -3
    task.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match replay"):
        replay_thesis(path)

def test_strict_claim_revision_can_be_replayed_at_exact_boundary():
    from src.kg.collaboration import ClaimProposal, Verification, evidence_corpus

    publication = datetime(2025, 1, 1, 4, tzinfo=timezone.utc)
    revision = datetime(2025, 1, 2, 4, tzinfo=timezone.utc)
    initial = source()
    corrected = source(version="2", published_at=revision.isoformat(), value=101, text="Corrected: 101 shares.")
    ledger = GraphLedger("0700.HK", strict_cutoff=True)

    def commit(item, effective, predecessor=None):
        cutoff = effective + timedelta(seconds=1)
        raw = LocalEvidence([item]).snapshot("0700.HK", cutoff)
        corpus, _ = evidence_corpus(raw, "0700.HK", cutoff, strict=True)
        evidence_id = corpus[0]["evidence_id"]
        claim = ClaimProposal(ticker="0700.HK", claim_type="fundamental", subject="Tencent",
                              predicate="held", metric="southbound holding", period="2025-01",
                              value=str(item["value"]), text=item["text"], polarity="neutral",
                              evidence_ids=[evidence_id], quotes={evidence_id: item["text"]},
                              available_at=effective, revision_of=predecessor)
        review = Verification(proposal_index=0, supported=True, evidence_ids=[evidence_id], reason="Supported")
        result = ledger.commit([claim], [review], corpus, cutoff)
        assert not result["rejected"]
        return result["accepted"][0]

    old = commit(initial, publication)
    new = commit(corrected, revision, predecessor=old)
    historical = ledger.freeze(revision)["claims"]
    assert [row["entity_id"] for row in historical] == [old]
    assert historical[0]["valid_to"] is None and "superseded_by" not in historical[0]
    assert [row["entity_id"] for row in ledger.freeze(revision + timedelta(seconds=1))["claims"]] == [new]


def test_rule_mining_samples_paths_and_generalizes_nodes(monkeypatch):
    cutoff = decision_cutoff("2025-01-04")
    graph = snapshot([
        source(),
        source(kind="ah_premium", document_id="premium", a_share="600001.SS",
               unit="percent", value=3, text="Premium was 3 percent."),
    ], cutoff)
    examples = [(graph, cutoff, "UP")] * 5
    all_rules = mine_rules(examples)
    assert len(all_rules) == 2

    monkeypatch.setattr("src.kg.thesis_paths.MINING_SAMPLE_WIDTH", 1)
    sampled = mine_rules(examples)
    assert len(sampled) == 1
    assert sampled == mine_rules(examples)
    assert sampled[0] in all_rules
    atoms = sampled[0]["body_atoms"]
    assert [atom["relation"] for atom in atoms] == sampled[0]["body"]
    assert atoms[0]["start_var"] == "X0"
    assert all(left["end_var"] == right["start_var"] for left, right in zip(atoms, atoms[1:]))


def test_beam_and_output_limits_preserve_ah_observation_identity():
    cutoff = decision_cutoff("2025-01-04")
    graph = snapshot([source(kind="ah_premium", document_id=f"premium-{i}", a_share="600001.SS",
                             period=f"period-{i}", unit="percent", value=i, text=f"Premium was {i} percent.")
                      for i in range(12)], cutoff)
    rule, = mine_rules([(graph, cutoff, "UP")] * 5)
    ctx, audit = retrieve_paths(graph, cutoff, [rule])
    assert audit["beam_pruned"] == 4 and audit["output_truncated"] == 4
    assert len(ctx["paths"]) == 4
    assert all(path["edges"][0]["observation_id"] == path["node_ids"][2] for path in ctx["paths"])
