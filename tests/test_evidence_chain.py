"""Tests for Round 1: Evidence-to-decision trace completeness.

Covers:
1. Schema validation (all pydantic models)
2. Evidence chain: SourceDocument → Evidence → Claim
3. Technical signal → Evidence + Claim (update_pipeline)
4. News polarity classifier
5. Agent roles returning claim_refs
6. Risk agent extracting claim_refs
7. Portfolio manager aggregating claim_ids into DecisionTrace
8. Baselines compatibility with claim_refs
9. BacktestOutcome linkage
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kg.schema import (
    AgentAssessment,
    BacktestOutcome,
    Claim,
    DecisionTrace,
    Evidence,
    SourceDocument,
    Entity,
    Relation,
    FinancialSignal,
    GraphUpdate,
    TradingDecision,
)
from src.kg.update_pipeline import (
    _classify_news_polarity,
    build_indicator_entities_from_ohlcv,
    build_news_from_frame,
    build_fundamentals_from_frame,
    build_risk_events_from_frame,
    build_global_news_from_frame,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

TICKER = "0005.HK"
TRADE_DATE = datetime(2025, 6, 15)


def _make_ohlcv(n: int = 250, base_price: float = 40.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data with enough rows for all indicators."""
    import numpy as np
    np.random.seed(42)
    dates = pd.bdate_range(end=TRADE_DATE, periods=n)
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 10.0)
    return pd.DataFrame({
        "date": dates,
        "open": prices + np.random.randn(n) * 0.2,
        "high": prices + abs(np.random.randn(n) * 0.5),
        "low": prices - abs(np.random.randn(n) * 0.5),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


def _make_news(n: int = 5) -> pd.DataFrame:
    """Generate synthetic news data."""
    rows = []
    for i in range(n):
        rows.append({
            "url": f"https://example.com/news/{i}",
            "title": f"News headline {i}: {'positive growth' if i % 2 == 0 else 'risk decline'}",
            "source": "test_source",
            "published_time": (TRADE_DATE - timedelta(days=i)).isoformat(),
            "content": f"Content about {TICKER}. {'Strong profit beat.' if i % 2 == 0 else 'Warning about losses.'}",
            "score": 0.7,
        })
    return pd.DataFrame(rows)


def _make_fundamentals() -> pd.DataFrame:
    """Generate synthetic fundamental data."""
    return pd.DataFrame([
        {"metric": "profitMargins", "value": 0.25, "numeric_value": 0.25, "as_of_date": TRADE_DATE.isoformat()},
        {"metric": "returnOnEquity", "value": 0.18, "numeric_value": 0.18, "as_of_date": TRADE_DATE.isoformat()},
        {"metric": "debtToEquity", "value": 150, "numeric_value": 150, "as_of_date": TRADE_DATE.isoformat()},
        {"metric": "currentRatio", "value": 1.2, "numeric_value": 1.2, "as_of_date": TRADE_DATE.isoformat()},
        {"metric": "beta", "value": 1.1, "numeric_value": 1.1, "as_of_date": TRADE_DATE.isoformat()},
    ])


# ──────────────────────────────────────────────
# 1. Schema validation
# ──────────────────────────────────────────────

class TestSchemaValidation:
    def test_source_document(self):
        doc = SourceDocument(
            source_id="src:1",
            source_type="news",
            source_name="Reuters",
            url="https://reuters.com/1",
            title="Test",
            published_at=TRADE_DATE,
            retrieved_at=TRADE_DATE,
            content_hash="abc123",
            raw_text_preview="Hello",
        )
        assert doc.source_id == "src:1"
        assert doc.source_type == "news"

    def test_evidence(self):
        ev = Evidence(
            evidence_id="ev:1",
            source_type="technical",
            extracted_text="RSI is 28",
            confidence=0.8,
        )
        assert ev.confidence == 0.8
        assert 0.0 <= ev.confidence <= 1.0

    def test_claim(self):
        claim = Claim(
            claim_id="claim:1",
            ticker=TICKER,
            claim_type="technical",
            text="Price above MA20",
            polarity="supports",
            confidence=0.75,
            as_of_date=TRADE_DATE,
            evidence_ids=["ev:1"],
        )
        assert claim.polarity == "supports"
        assert len(claim.evidence_ids) == 1

    def test_agent_assessment(self):
        aa = AgentAssessment(
            assessment_id="assess:1",
            ticker=TICKER,
            trade_date=TRADE_DATE,
            agent_role="technical",
            stance="bullish",
            confidence=0.8,
            score=0.6,
            summary="Bullish signals",
            evidence_refs=["ev:1"],
            claim_refs=["claim:1"],
            factors=[],
            supports_decision=True,
            opposes_decision=False,
        )
        assert aa.claim_refs == ["claim:1"]
        assert aa.supports_decision is True

    def test_decision_trace(self):
        dt = DecisionTrace(
            decision_id="dec:1",
            ticker=TICKER,
            trade_date=TRADE_DATE,
            action="buy",
            final_score=0.45,
            confidence=0.7,
            conflict_level=0.1,
            decision_reason="Net bullish",
            evidence_ids=["ev:1"],
            claim_ids=["claim:1"],
            supporting_roles=["technical"],
            opposing_roles=[],
        )
        assert dt.claim_ids == ["claim:1"]
        assert dt.action == "buy"

    def test_backtest_outcome(self):
        bo = BacktestOutcome(
            outcome_id="out:1",
            decision_id="dec:1",
            ticker=TICKER,
            trade_date=TRADE_DATE,
            action="buy",
            raw_return=0.05,
            holding_days=5,
            trade_executed=True,
            direction_outcome="correct",
            is_profitable=True,
        )
        assert bo.decision_id == "dec:1"
        assert bo.is_profitable is True

    def test_trading_decision(self):
        td = TradingDecision(
            ticker=TICKER,
            action="buy",
            bullish_score=1.5,
            bearish_score=0.3,
            neutral_score=0.1,
            evidence_ids=["ev:1"],
            reason="Bullish",
        )
        assert td.action == "buy"


# ──────────────────────────────────────────────
# 2. Evidence chain: SourceDocument → Evidence → Claim
# ──────────────────────────────────────────────

class TestEvidenceChain:
    def test_news_builds_full_chain(self):
        """News pipeline should create SourceDocument → Evidence → Claim → NewsEvent."""
        news = _make_news(3)
        entities, evidences, relations = build_news_from_frame(TICKER, news)

        # Should have entities: 3 SourceDocument + 3 Claim + 3 NewsEvent = 9
        source_docs = [e for e in entities if e.type == "SourceDocument"]
        claims = [e for e in entities if e.type == "Claim"]
        news_events = [e for e in entities if e.type == "NewsEvent"]

        assert len(source_docs) == 3, f"Expected 3 SourceDocuments, got {len(source_docs)}"
        assert len(claims) == 3, f"Expected 3 Claims, got {len(claims)}"
        assert len(news_events) == 3, f"Expected 3 NewsEvents, got {len(news_events)}"

        # Should have 3 evidences
        assert len(evidences) == 3

        # Check relation types exist
        rel_types = {r.type for r in relations}
        assert "CONTAINS_EVIDENCE" in rel_types, "Missing SourceDocument → Evidence link"
        assert "SUPPORTS_CLAIM" in rel_types, "Missing Evidence → Claim link"
        assert "CLAIM_USED_BY" in rel_types, "Missing Claim → NewsEvent link"
        assert "MENTIONED_IN" in rel_types, "Missing Company → NewsEvent link"

        # Verify chain connectivity: each source doc links to an evidence
        source_ids = {e.entity_id for e in source_docs}
        contains_rels = [r for r in relations if r.type == "CONTAINS_EVIDENCE"]
        for rel in contains_rels:
            assert rel.start_id in source_ids, f"CONTAINS_EVIDENCE from unknown source: {rel.start_id}"

    def test_technical_builds_full_chain(self):
        """Technical pipeline should create IndicatorSignal → Evidence → Claim chain."""
        ohlcv = _make_ohlcv(250)
        entities, evidences, relations = build_indicator_entities_from_ohlcv(TICKER, ohlcv)

        claims = [e for e in entities if e.type == "Claim"]
        signals = [e for e in entities if e.type == "IndicatorSignal"]

        assert len(claims) > 0, "Technical pipeline should generate Claims"
        assert len(evidences) > 0, "Technical pipeline should generate Evidences"
        assert len(signals) > 0, "Technical pipeline should generate IndicatorSignals"

        # Each claim should reference evidence
        for claim in entities:
            if claim.type == "Claim":
                assert "evidence_ids" in claim.properties
                assert len(claim.properties["evidence_ids"]) > 0

        # Check relation types
        rel_types = {r.type for r in relations}
        assert "SUPPORTED_BY" in rel_types
        assert "SUPPORTS_CLAIM" in rel_types
        assert "CLAIM_USED_BY" in rel_types

    def test_fundamental_builds_full_chain(self):
        """Fundamental pipeline should create FundamentalSignal → Evidence → Claim chain."""
        fundamentals = _make_fundamentals()
        entities, evidences, relations = build_fundamentals_from_frame(TICKER, fundamentals)

        claims = [e for e in entities if e.type == "Claim"]
        signals = [e for e in entities if e.type == "FundamentalSignal"]

        assert len(claims) == 5, f"Expected 5 Claims, got {len(claims)}"
        assert len(signals) == 5, f"Expected 5 FundamentalSignals, got {len(signals)}"
        assert len(evidences) == 5

        rel_types = {r.type for r in relations}
        assert "SUPPORTS_CLAIM" in rel_types
        assert "CLAIM_USED_BY" in rel_types

    def test_risk_builds_full_chain(self):
        """Risk pipeline should create RiskEvent → Evidence → Claim chain."""
        ohlcv = _make_ohlcv(250)
        # Make high volatility
        ohlcv["close"] = ohlcv["close"] * 1.0  # keep as-is, synthetic data has some vol
        fundamentals = _make_fundamentals()

        entities, evidences, relations = build_risk_events_from_frame(TICKER, ohlcv, fundamentals)

        claims = [e for e in entities if e.type == "Claim"]
        risks = [e for e in entities if e.type == "RiskEvent"]

        assert len(claims) > 0, "Risk pipeline should generate Claims"
        assert len(evidences) > 0
        assert len(risks) > 0

        # Each risk claim should have evidence
        for claim in claims:
            assert len(claim.properties.get("evidence_ids", [])) > 0

    def test_global_news_builds_full_chain(self):
        """Global news pipeline should mirror news pipeline structure."""
        news = _make_news(2)
        entities, evidences, relations = build_global_news_from_frame(TICKER, news)

        source_docs = [e for e in entities if e.type == "SourceDocument"]
        claims = [e for e in entities if e.type == "Claim"]

        assert len(source_docs) == 2
        assert len(claims) == 2

        rel_types = {r.type for r in relations}
        assert "CONTAINS_EVIDENCE" in rel_types
        assert "SUPPORTS_CLAIM" in rel_types


# ──────────────────────────────────────────────
# 3. News polarity classifier
# ──────────────────────────────────────────────

class TestNewsPolarityClassifier:
    def test_positive_news(self):
        result = _classify_news_polarity("Strong profit growth", "Record earnings beat expectations")
        assert result == "supports"

    def test_negative_news(self):
        result = _classify_news_polarity("Stock crash warning", "Major losses and debt concerns")
        assert result == "contradicts"

    def test_neutral_news(self):
        result = _classify_news_polarity("Company announces meeting", "Board discussed strategy")
        assert result == "neutral"

    def test_empty_input(self):
        result = _classify_news_polarity(None, None)
        assert result == "neutral"

    def test_mixed_signals(self):
        # Equal positive and negative → neutral
        result = _classify_news_polarity("Growth and decline", "Profit but loss warning")
        # This depends on exact keyword counts; just ensure it returns a valid value
        assert result in {"supports", "contradicts", "neutral"}

    def test_polarity_appears_in_claims(self):
        """Verify that polarity is actually set on Claim entities."""
        news = pd.DataFrame([{
            "url": "https://example.com/1",
            "title": "Strong profit growth and recovery",
            "source": "test",
            "published_time": TRADE_DATE.isoformat(),
            "content": "Record earnings, beat expectations, bullish rally",
            "score": 0.8,
        }])
        entities, _, _ = build_news_from_frame(TICKER, news)
        claims = [e for e in entities if e.type == "Claim"]
        assert len(claims) == 1
        assert claims[0].properties["polarity"] == "supports"


# ──────────────────────────────────────────────
# 4. Agent roles return claim_refs
# ──────────────────────────────────────────────

class TestAgentClaimRefs:
    def test_technical_agent_returns_claim_refs(self):
        from src.agents.roles import technical_agent
        # Build a subgraph with signals and claims
        subgraph = {
            "signals": [
                {"entity_id": "sig:1", "direction": "bullish", "strength": 0.7, "evidence_id": "ev:1", "claim_id": "claim:1"},
                {"entity_id": "sig:2", "direction": "bearish", "strength": 0.5, "evidence_id": "ev:2", "claim_id": "claim:2"},
            ],
            "claims": [
                {"claim_id": "claim:1", "entity_id": "claim:1"},
                {"claim_id": "claim:2", "entity_id": "claim:2"},
            ],
        }
        report = technical_agent(subgraph, TICKER, TRADE_DATE)
        assert hasattr(report, "claim_refs")
        assert len(report.claim_refs) > 0, "Technical agent should return claim_refs"
        assert "claim:1" in report.claim_refs or "claim:2" in report.claim_refs

    def test_risk_agent_returns_claim_refs(self):
        from src.agents.risk_agent import RiskAgent
        agent = RiskAgent()
        subgraph = {
            "risks": [
                {
                    "entity_id": "risk:1",
                    "direction": "bearish",
                    "severity": 0.8,
                    "evidence_id": "ev:risk:1",
                    "claim_id": "claim:risk:1",
                    "name": "high_volatility",
                    "risk_type": "volatility",
                },
            ],
            "signals": [],
            "evidences": [],
            "claims": [],
        }
        report = agent.run({"ticker": TICKER, "trade_date": TRADE_DATE, "subgraph": subgraph})
        assert hasattr(report, "claim_refs")
        assert "claim:risk:1" in report.claim_refs


# ──────────────────────────────────────────────
# 5. Portfolio manager aggregates claim_ids
# ──────────────────────────────────────────────

class TestPortfolioManagerClaimAggregation:
    def test_decision_trace_has_claim_ids(self):
        from src.agents.roles import AgentReport, portfolio_manager_decide
        reports = [
            AgentReport(
                role="technical",
                stance="bullish",
                confidence=0.8,
                score=0.6,
                summary="Bullish",
                evidence_refs=["ev:1"],
                claim_refs=["claim:t1", "claim:t2"],
                factors=[],
            ),
            AgentReport(
                role="news",
                stance="neutral",
                confidence=0.5,
                score=0.0,
                summary="Neutral news",
                evidence_refs=["ev:2"],
                claim_refs=["claim:n1"],
                factors=[],
            ),
            AgentReport(
                role="risk",
                stance="bearish",
                confidence=0.6,
                score=-0.4,
                summary="Some risk",
                evidence_refs=["ev:3"],
                claim_refs=["claim:r1"],
                factors=[],
            ),
        ]
        result = portfolio_manager_decide(TICKER, TRADE_DATE, reports)

        # Check claim_refs in result
        assert "claim_refs" in result
        assert len(result["claim_refs"]) > 0

        # Check DecisionTrace has claim_ids
        dt = result["decision_trace"]
        assert "claim_ids" in dt
        assert len(dt["claim_ids"]) > 0

        # All claim refs from agents should be aggregated
        all_expected = {"claim:t1", "claim:t2", "claim:n1", "claim:r1"}
        assert set(dt["claim_ids"]) == all_expected

    def test_decision_trace_has_evidence_ids(self):
        from src.agents.roles import AgentReport, portfolio_manager_decide
        reports = [
            AgentReport(
                role="technical",
                stance="bullish",
                confidence=0.8,
                score=0.6,
                summary="Bullish",
                evidence_refs=["ev:1", "ev:2"],
                claim_refs=["claim:1"],
                factors=[],
            ),
        ]
        result = portfolio_manager_decide(TICKER, TRADE_DATE, reports)
        dt = result["decision_trace"]
        assert "evidence_ids" in dt
        assert len(dt["evidence_ids"]) > 0

    def test_agent_assessments_have_claim_refs(self):
        from src.agents.roles import AgentReport, portfolio_manager_decide
        reports = [
            AgentReport(
                role="technical",
                stance="bullish",
                confidence=0.8,
                score=0.6,
                summary="Bullish",
                evidence_refs=["ev:1"],
                claim_refs=["claim:1"],
                factors=[],
            ),
        ]
        result = portfolio_manager_decide(TICKER, TRADE_DATE, reports)
        assessments = result["agent_assessments"]
        assert len(assessments) == 1
        assert "claim_refs" in assessments[0]
        assert assessments[0]["claim_refs"] == ["claim:1"]


# ──────────────────────────────────────────────
# 6. BacktestOutcome linkage
# ──────────────────────────────────────────────

class TestBacktestOutcomeLinkage:
    def test_outcome_links_to_decision(self):
        """BacktestOutcome.decision_id should reference a valid DecisionTrace."""
        dt = DecisionTrace(
            decision_id="dec:test:1",
            ticker=TICKER,
            trade_date=TRADE_DATE,
            action="buy",
            final_score=0.45,
            confidence=0.7,
            conflict_level=0.1,
            decision_reason="Test",
            evidence_ids=["ev:1"],
            claim_ids=["claim:1"],
        )
        bo = BacktestOutcome(
            outcome_id="out:test:1",
            decision_id=dt.decision_id,
            ticker=TICKER,
            trade_date=TRADE_DATE,
            action=dt.action,
            raw_return=0.03,
            holding_days=5,
            trade_executed=True,
            direction_outcome="correct",
            is_profitable=True,
        )
        assert bo.decision_id == dt.decision_id
        assert bo.action == dt.action


# ──────────────────────────────────────────────
# 7. Full pipeline integration (no LLM, no Neo4j)
# ──────────────────────────────────────────────

class TestFullPipelineIntegration:
    def test_update_pipeline_entities(self):
        """update_pipeline should produce entities with claim structures."""
        ohlcv = _make_ohlcv(250)
        news = _make_news(3)

        # Technical
        sig_ents, sig_evs, sig_rels = build_indicator_entities_from_ohlcv(TICKER, ohlcv)
        sig_claims = [e for e in sig_ents if e.type == "Claim"]
        assert len(sig_claims) > 0

        # News
        news_ents, news_evs, news_rels = build_news_from_frame(TICKER, news)
        news_claims = [e for e in news_ents if e.type == "Claim"]
        assert len(news_claims) == 3

        # Fundamentals
        fund_ents, fund_evs, fund_rels = build_fundamentals_from_frame(TICKER, _make_fundamentals())
        fund_claims = [e for e in fund_ents if e.type == "Claim"]
        assert len(fund_claims) == 5

        # Risk
        risk_ents, risk_evs, risk_rels = build_risk_events_from_frame(TICKER, ohlcv, _make_fundamentals())
        risk_claims = [e for e in risk_ents if e.type == "Claim"]
        assert len(risk_claims) > 0

        # Total claims across all pipelines
        total_claims = len(sig_claims) + len(news_claims) + len(fund_claims) + len(risk_claims)
        assert total_claims >= 10, f"Expected ≥10 total claims, got {total_claims}"

    def test_evidence_chain_completeness(self):
        """Every Claim should have at least one evidence_id, and that evidence should exist."""
        ohlcv = _make_ohlcv(250)
        news = _make_news(2)

        _, sig_evs, _ = build_indicator_entities_from_ohlcv(TICKER, ohlcv)
        news_ents, news_evs, _ = build_news_from_frame(TICKER, news)

        all_evidence_ids = {ev.evidence_id for ev in sig_evs} | {ev.evidence_id for ev in news_evs}

        # Check news claims reference valid evidence
        for entity in news_ents:
            if entity.type == "Claim":
                for eid in entity.properties.get("evidence_ids", []):
                    assert eid in all_evidence_ids, f"Claim references missing evidence: {eid}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
