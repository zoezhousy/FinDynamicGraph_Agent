from src.decision.graph_decision import GraphDecisionEngine


def test_buy_decision_when_bullish_score_is_stronger():
    engine = GraphDecisionEngine()

    decision = engine.decide(
        "0005.HK",
        [
            {
                "direction": "bullish",
                "strength": 0.8,
                "evidence_id": "ev_1",
            }
        ],
    )

    assert decision.action == "buy"
    assert decision.evidence_ids == ["ev_1"]


def test_abstain_when_no_signals():
    engine = GraphDecisionEngine()

    decision = engine.decide("0005.HK", [])

    assert decision.action == "abstain"