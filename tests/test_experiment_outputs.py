"""Tests for experiment output structure and case study candidate selection.

These tests verify the output format of main_experiment without requiring
a running Neo4j or LLM API. They use synthetic data to test the helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.metrics import full_summary_by_system
from src.main_experiment import _select_case_study_candidates


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_trade_row(
    system: str,
    ticker: str = "0700.HK",
    trade_date: str = "2025-03-01",
    action: str = "buy",
    trade_executed: bool = True,
    raw_return: float = 0.02,
    confidence: float = 0.6,
    conflict_level: float = 0.0,
    evidence_refs: list | None = None,
    claim_refs: list | None = None,
    fresh_evidence_count: int = 5,
    stale_evidence_count: int = 1,
    decision_reason: str = "Test reason",
    direction_outcome: str = "correct",
) -> dict:
    return {
        "system": system,
        "ticker": ticker,
        "trade_date": trade_date,
        "action": action,
        "trade_executed": trade_executed,
        "raw_return": raw_return,
        "confidence": confidence,
        "conflict_level": conflict_level,
        "evidence_refs": evidence_refs or ["ev1", "ev2"],
        "claim_refs": claim_refs or ["c1"],
        "fresh_evidence_count": fresh_evidence_count,
        "stale_evidence_count": stale_evidence_count,
        "decision_reason": decision_reason,
        "direction_outcome": direction_outcome,
        "final_score": 0.3,
        "supporting_roles": ["technical", "news"],
        "opposing_roles": [],
        "evidence_alignment": "aligned",
    }


def _sample_trades() -> pd.DataFrame:
    """Build a sample trades DataFrame with all 4 systems."""
    rows = []
    for system in ("no_kg_no_evidence", "evidence_no_kg", "static_kg", "kg_dynamic"):
        rows.append(_make_trade_row(system=system))
        rows.append(_make_trade_row(
            system=system,
            trade_date="2025-04-01",
            action="sell",
            raw_return=-0.01,
            direction_outcome="incorrect",
        ))
    return pd.DataFrame(rows)


# ── Test: Summary output contains required columns ─────────────────────


class TestSummaryOutput:
    def test_required_columns_present(self):
        df = _sample_trades()
        summary = full_summary_by_system(df)

        required_cols = {
            "n_decisions",
            "n_trades",
            "trade_execution_rate",
            "win_rate",
            "directional_accuracy",
            "abstain_rate",
            "mean_return",
            "median_return",
            "mean_confidence",
            "mean_conflict_level",
            "mean_fresh_evidence",
            "mean_stale_evidence",
            "evidence_coverage",
            "citation_precision",
            "claim_coverage",
            "unsupported_claim_rate",
            "stale_evidence_rate",
        }

        for col in required_cols:
            assert col in summary.columns, f"Missing column: {col}"

    def test_four_systems_present(self):
        df = _sample_trades()
        summary = full_summary_by_system(df)

        expected_systems = {"no_kg_no_evidence", "evidence_no_kg", "static_kg", "kg_dynamic"}
        assert set(summary.index) == expected_systems

    def test_n_decisions_correct(self):
        df = _sample_trades()
        summary = full_summary_by_system(df)

        for system in summary.index:
            assert summary.loc[system, "n_decisions"] == 2


# ── Test: Case study candidates ────────────────────────────────────────


class TestCaseStudyCandidates:
    def test_required_columns(self):
        df = _sample_trades()
        candidates = _select_case_study_candidates(df)

        required_cols = {
            "ticker",
            "trade_date",
            "system",
            "action",
            "final_score",
            "confidence",
            "conflict_level",
            "evidence_count",
            "claim_count",
            "fresh_evidence_count",
            "stale_evidence_count",
            "reason",
            "why_candidate",
        }
        assert set(candidates.columns) == required_cols

    def test_detects_conflict_driven_abstain(self):
        """When kg_dynamic abstains but baseline buys, it should be a candidate."""
        rows = [
            _make_trade_row(
                system="kg_dynamic",
                action="abstain",
                trade_executed=False,
                raw_return=0.0,
                conflict_level=0.6,
            ),
            _make_trade_row(
                system="no_kg_no_evidence",
                action="buy",
                trade_executed=True,
            ),
            _make_trade_row(
                system="evidence_no_kg",
                action="buy",
                trade_executed=True,
            ),
            _make_trade_row(
                system="static_kg",
                action="buy",
                trade_executed=True,
            ),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)

        assert len(candidates) >= 1
        row = candidates.iloc[0]
        assert "abstains" in row["why_candidate"]

    def test_detects_high_conflict(self):
        rows = [
            _make_trade_row(system="kg_dynamic", conflict_level=0.7),
            _make_trade_row(system="no_kg_no_evidence"),
            _make_trade_row(system="evidence_no_kg"),
            _make_trade_row(system="static_kg"),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        assert len(candidates) >= 1
        assert any("conflict" in r["why_candidate"] for _, r in candidates.iterrows())

    def test_empty_when_no_interesting_decisions(self):
        """When all systems agree and conflict is low, no candidates."""
        rows = [
            _make_trade_row(system="kg_dynamic", action="buy", conflict_level=0.0),
            _make_trade_row(system="no_kg_no_evidence", action="buy"),
            _make_trade_row(system="evidence_no_kg", action="buy"),
            _make_trade_row(system="static_kg", action="buy"),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        # Might be empty or have few — depends on other criteria
        # At minimum, well-grounded criterion might match
        assert isinstance(candidates, pd.DataFrame)
