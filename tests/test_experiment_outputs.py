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
            "case_group",
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

    def test_case_group_values_valid(self):
        df = _sample_trades()
        candidates = _select_case_study_candidates(df)
        valid_groups = {"mechanism", "outcome_best", "outcome_worst"}
        assert set(candidates["case_group"].unique()).issubset(valid_groups)

    def test_detects_conflict_driven_abstain(self):
        """When kg_dynamic abstains but baseline buys, it should be a mechanism candidate."""
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

        mechanism = candidates[candidates["case_group"] == "mechanism"]
        assert len(mechanism) >= 1
        assert any("abstains" in r["why_candidate"] for _, r in mechanism.iterrows())

    def test_detects_high_conflict(self):
        rows = [
            _make_trade_row(system="kg_dynamic", conflict_level=0.7),
            _make_trade_row(system="no_kg_no_evidence"),
            _make_trade_row(system="evidence_no_kg"),
            _make_trade_row(system="static_kg"),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        mechanism = candidates[candidates["case_group"] == "mechanism"]
        assert len(mechanism) >= 1
        assert any("conflict" in r["why_candidate"] for _, r in mechanism.iterrows())

    def test_empty_when_no_interesting_decisions(self):
        """When all systems agree and conflict is low, only outcome groups may appear."""
        rows = [
            _make_trade_row(system="kg_dynamic", action="buy", conflict_level=0.0),
            _make_trade_row(system="no_kg_no_evidence", action="buy"),
            _make_trade_row(system="evidence_no_kg", action="buy"),
            _make_trade_row(system="static_kg", action="buy"),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        assert isinstance(candidates, pd.DataFrame)
        # outcome groups always present (at least best + worst)
        assert set(candidates["case_group"].unique()).issubset(
            {"mechanism", "outcome_best", "outcome_worst"}
        )

    def test_mechanism_group_ignores_outcome(self):
        """Mechanism candidates must be selectable without direction_outcome."""
        rows = [
            # kg_dynamic: incorrect outcome, but high conflict → mechanism
            _make_trade_row(
                system="kg_dynamic",
                conflict_level=0.8,
                raw_return=-0.03,
                direction_outcome="incorrect",
            ),
            _make_trade_row(system="no_kg_no_evidence"),
            _make_trade_row(system="evidence_no_kg"),
            _make_trade_row(system="static_kg"),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        mechanism = candidates[candidates["case_group"] == "mechanism"]
        assert len(mechanism) >= 1
        # The incorrect-outcome row should still be in mechanism group
        assert any(r["conflict_level"] >= 0.5 for _, r in mechanism.iterrows())

    def test_outcome_best_and_worst_present(self):
        """At least one best-case and one worst-case in outcome groups."""
        rows = [
            _make_trade_row(
                system="kg_dynamic",
                trade_date="2025-03-01",
                raw_return=0.05,
                direction_outcome="correct",
            ),
            _make_trade_row(
                system="no_kg_no_evidence",
                trade_date="2025-03-01",
                raw_return=-0.02,
                direction_outcome="incorrect",
            ),
            _make_trade_row(
                system="evidence_no_kg",
                trade_date="2025-03-01",
                raw_return=-0.01,
                direction_outcome="incorrect",
            ),
            _make_trade_row(
                system="static_kg",
                trade_date="2025-03-01",
                raw_return=0.01,
                direction_outcome="correct",
            ),
            _make_trade_row(
                system="kg_dynamic",
                trade_date="2025-04-01",
                raw_return=-0.05,
                direction_outcome="incorrect",
            ),
            _make_trade_row(
                system="no_kg_no_evidence",
                trade_date="2025-04-01",
                raw_return=0.01,
                direction_outcome="correct",
            ),
            _make_trade_row(
                system="evidence_no_kg",
                trade_date="2025-04-01",
                raw_return=0.02,
                direction_outcome="correct",
            ),
            _make_trade_row(
                system="static_kg",
                trade_date="2025-04-01",
                raw_return=0.01,
                direction_outcome="correct",
            ),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        groups = set(candidates["case_group"].unique())
        assert "outcome_best" in groups
        assert "outcome_worst" in groups

    def test_no_cherry_picking_in_mechanism_group(self):
        """Mechanism group must not contain outcome-correctness as a reason."""
        rows = [
            _make_trade_row(system="kg_dynamic", conflict_level=0.7, direction_outcome="correct"),
            _make_trade_row(system="no_kg_no_evidence", direction_outcome="incorrect"),
            _make_trade_row(system="evidence_no_kg", direction_outcome="incorrect"),
            _make_trade_row(system="static_kg", direction_outcome="incorrect"),
        ]
        df = pd.DataFrame(rows)
        candidates = _select_case_study_candidates(df)
        mechanism = candidates[candidates["case_group"] == "mechanism"]
        for _, row in mechanism.iterrows():
            assert "correct" not in row["why_candidate"].lower() or "incorrect" not in row["why_candidate"].lower(), \
                f"Mechanism reason references outcome: {row['why_candidate']}"
