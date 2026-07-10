"""Tests for DecisionTrace, AgentAssessment, and BacktestOutcome schema."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.kg.schema import AgentAssessment, BacktestOutcome, DecisionTrace


# ── DecisionTrace ──────────────────────────────────────────────────────


class TestDecisionTrace:
    def test_has_evidence_ids(self):
        dt = DecisionTrace(
            decision_id="decision:0700.HK:2025-03-01",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="buy",
            final_score=0.35,
            confidence=0.65,
            conflict_level=0.2,
            decision_reason="Test reason",
            evidence_ids=["ev1", "ev2"],
            claim_ids=["claim1"],
        )
        assert dt.evidence_ids == ["ev1", "ev2"]
        assert len(dt.evidence_ids) == 2

    def test_has_claim_ids(self):
        dt = DecisionTrace(
            decision_id="decision:0700.HK:2025-03-01",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="buy",
            final_score=0.35,
            confidence=0.65,
            conflict_level=0.2,
            decision_reason="Test",
            claim_ids=["c1", "c2", "c3"],
        )
        assert dt.claim_ids == ["c1", "c2", "c3"]
        assert len(dt.claim_ids) == 3

    def test_defaults(self):
        dt = DecisionTrace(
            decision_id="d1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="hold",
            final_score=0.0,
            confidence=0.5,
            conflict_level=0.0,
            decision_reason="Default test",
        )
        assert dt.evidence_ids == []
        assert dt.claim_ids == []
        assert dt.bullish_support_count == 0
        assert dt.bearish_support_count == 0
        assert dt.stale_evidence_count == 0
        assert dt.fresh_evidence_count == 0
        assert dt.evidence_alignment == "unknown"

    def test_model_dump_serializable(self):
        dt = DecisionTrace(
            decision_id="d1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="buy",
            final_score=0.3,
            confidence=0.6,
            conflict_level=0.1,
            decision_reason="test",
            evidence_ids=["ev1"],
            claim_ids=["c1"],
            trace={"technical_score": 0.2},
        )
        data = dt.model_dump(mode="json")
        assert isinstance(data["evidence_ids"], list)
        assert isinstance(data["claim_ids"], list)
        assert isinstance(data["trace"], dict)


# ── AgentAssessment ────────────────────────────────────────────────────


class TestAgentAssessment:
    def test_has_claim_refs(self):
        aa = AgentAssessment(
            assessment_id="a1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            agent_role="technical",
            stance="bullish",
            confidence=0.7,
            score=0.4,
            summary="Bullish test",
            claim_refs=["c1", "c2"],
        )
        assert aa.claim_refs == ["c1", "c2"]

    def test_has_evidence_refs(self):
        aa = AgentAssessment(
            assessment_id="a1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            agent_role="news",
            stance="neutral",
            confidence=0.5,
            score=0.0,
            summary="Neutral test",
            evidence_refs=["ev1", "ev2", "ev3"],
        )
        assert len(aa.evidence_refs) == 3

    def test_supports_opposes_decision(self):
        aa = AgentAssessment(
            assessment_id="a1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            agent_role="risk",
            stance="bearish",
            confidence=0.6,
            score=-0.3,
            summary="Risk test",
            supports_decision=False,
            opposes_decision=True,
        )
        assert aa.supports_decision is False
        assert aa.opposes_decision is True

    def test_defaults(self):
        aa = AgentAssessment(
            assessment_id="a1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            agent_role="fundamental",
            stance="bullish",
            confidence=0.7,
            score=0.3,
            summary="test",
        )
        assert aa.evidence_refs == []
        assert aa.claim_refs == []
        assert aa.factors == []
        assert aa.supports_decision is None


# ── BacktestOutcome ────────────────────────────────────────────────────


class TestBacktestOutcome:
    def test_links_to_decision(self):
        bo = BacktestOutcome(
            outcome_id="outcome:0700.HK:2025-03-01:kg_dynamic",
            decision_id="decision:0700.HK:2025-03-01",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="buy",
            system="kg_dynamic",
            raw_return=0.025,
            holding_days=5,
            trade_executed=True,
            direction_outcome="correct",
            is_profitable=True,
        )
        assert bo.decision_id == "decision:0700.HK:2025-03-01"
        assert bo.direction_outcome == "correct"
        assert bo.is_profitable is True

    def test_not_executed(self):
        bo = BacktestOutcome(
            outcome_id="outcome:0700.HK:2025-03-01:kg_dynamic",
            decision_id="decision:0700.HK:2025-03-01",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="abstain",
            system="kg_dynamic",
            raw_return=0.0,
            holding_days=0,
            trade_executed=False,
            direction_outcome="not_executed",
            is_profitable=None,
        )
        assert bo.trade_executed is False
        assert bo.is_profitable is None

    def test_model_dump_serializable(self):
        bo = BacktestOutcome(
            outcome_id="o1",
            decision_id="d1",
            ticker="0700.HK",
            trade_date=datetime(2025, 3, 1),
            action="buy",
            system="kg_dynamic",
            metadata={"key": "value"},
        )
        data = bo.model_dump(mode="json")
        assert isinstance(data["trade_date"], str)
        assert isinstance(data["metadata"], dict)
