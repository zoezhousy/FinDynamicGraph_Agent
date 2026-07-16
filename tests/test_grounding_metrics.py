"""Tests for src/eval/grounding.py grounding metrics."""

from __future__ import annotations

import pytest
import pandas as pd

from src.eval.grounding import (
    citation_precision,
    claim_coverage,
    compute_grounding_metrics,
    evidence_coverage,
    grounding_metrics_by_system,
    stale_evidence_rate,
    unsupported_claim_rate,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _row(
    evidence_refs=None,
    claim_refs=None,
    decision_reason="Some reason",
    stale_evidence_count=None,
    fresh_evidence_count=None,
    trade_date="2025-03-01",
    system="kg_dynamic",
    retrieved_evidence_ids=None,
    retrieved_claim_ids=None,
) -> dict:
    return {
        "evidence_refs": evidence_refs or [],
        "claim_refs": claim_refs or [],
        "decision_reason": decision_reason,
        "stale_evidence_count": stale_evidence_count,
        "fresh_evidence_count": fresh_evidence_count,
        "trade_date": trade_date,
        "system": system,
        "retrieved_evidence_ids": retrieved_evidence_ids or [],
        "retrieved_claim_ids": retrieved_claim_ids or [],
    }


# ── citation_precision ─────────────────────────────────────────────────


class TestCitationPrecision:
    def test_all_have_refs(self):
        rows = [
            _row(evidence_refs=["ev1", "ev2"]),
            _row(evidence_refs=["ev3"]),
        ]
        assert citation_precision(rows) == 1.0

    def test_empty_refs(self):
        rows = [
            _row(evidence_refs=[]),
            _row(evidence_refs=[]),
        ]
        # 0 total refs → 0.0
        assert citation_precision(rows) == 0.0

    def test_mixed(self):
        rows = [
            _row(evidence_refs=["ev1"]),
            _row(evidence_refs=[]),
        ]
        # 1 ref total, 1 existing → 1.0 (all refs that exist are counted)
        assert citation_precision(rows) == 1.0

    def test_dataframe_input(self):
        df = pd.DataFrame([
            _row(evidence_refs=["ev1", "ev2"]),
            _row(evidence_refs=["ev3"]),
        ])
        assert citation_precision(df) == 1.0

    def test_empty_input(self):
        assert citation_precision([]) == 0.0

    def test_hallucinated_refs_below_one(self):
        """Refs NOT in the retrieved set should drag precision below 1.0."""
        rows = [
            _row(
                evidence_refs=["ev_real", "ev_fake"],
                claim_refs=["c_real"],
                retrieved_evidence_ids=["ev_real"],
                retrieved_claim_ids=["c_real"],
            ),
        ]
        # 3 total refs, 2 exist in retrieved set → 2/3
        assert citation_precision(rows) == pytest.approx(2 / 3)

    def test_all_refs_exist_in_retrieved(self):
        """When every ref is in the retrieved set, precision stays 1.0."""
        rows = [
            _row(
                evidence_refs=["ev1", "ev2"],
                claim_refs=["c1"],
                retrieved_evidence_ids=["ev1", "ev2", "ev3"],
                retrieved_claim_ids=["c1"],
            ),
        ]
        assert citation_precision(rows) == 1.0

    def test_fallback_when_no_retrieved_ids(self):
        """Old heuristic kicks in when retrieved-ID columns are absent."""
        rows = [
            {"evidence_refs": ["ev1"], "claim_refs": []},
        ]
        # No retrieved columns → fallback: non-empty refs count as valid
        assert citation_precision(rows) == 1.0


# ── claim_coverage ─────────────────────────────────────────────────────


class TestClaimCoverage:
    def test_all_have_claims(self):
        rows = [
            _row(claim_refs=["c1"]),
            _row(claim_refs=["c2", "c3"]),
        ]
        assert claim_coverage(rows) == 1.0

    def test_none_have_claims(self):
        rows = [
            _row(claim_refs=[]),
            _row(claim_refs=[]),
        ]
        assert claim_coverage(rows) == 0.0

    def test_partial(self):
        rows = [
            _row(claim_refs=["c1"]),
            _row(claim_refs=[]),
        ]
        assert claim_coverage(rows) == 0.5

    def test_empty_input(self):
        assert claim_coverage([]) == 0.0


# ── unsupported_claim_rate ─────────────────────────────────────────────


class TestUnsupportedClaimRate:
    def test_all_supported(self):
        rows = [
            _row(claim_refs=["c1"], decision_reason="reason"),
            _row(claim_refs=["c2"], decision_reason="reason"),
        ]
        assert unsupported_claim_rate(rows) == 0.0

    def test_all_unsupported(self):
        rows = [
            _row(claim_refs=[], decision_reason="some reason"),
            _row(claim_refs=[], decision_reason="another reason"),
        ]
        assert unsupported_claim_rate(rows) == 1.0

    def test_mixed(self):
        rows = [
            _row(claim_refs=["c1"], decision_reason="reason"),
            _row(claim_refs=[], decision_reason="reason"),
        ]
        assert unsupported_claim_rate(rows) == 0.5

    def test_no_reason_skipped(self):
        """Decisions without a reason are not counted."""
        rows = [
            _row(claim_refs=[], decision_reason=""),
            _row(claim_refs=["c1"], decision_reason="reason"),
        ]
        # Only 1 has reason, and it's supported → 0.0
        assert unsupported_claim_rate(rows) == 0.0

    def test_empty_input(self):
        assert unsupported_claim_rate([]) == 0.0


# ── stale_evidence_rate ────────────────────────────────────────────────


class TestStaleEvidenceRate:
    def test_uses_precomputed_counts(self):
        rows = [
            _row(stale_evidence_count=3, fresh_evidence_count=7),
            _row(stale_evidence_count=1, fresh_evidence_count=9),
        ]
        rate = stale_evidence_rate(rows)
        assert rate == pytest.approx(4 / 20)

    def test_no_stale(self):
        rows = [
            _row(stale_evidence_count=0, fresh_evidence_count=10),
        ]
        assert stale_evidence_rate(rows) == 0.0

    def test_all_stale(self):
        rows = [
            _row(stale_evidence_count=5, fresh_evidence_count=0),
        ]
        assert stale_evidence_rate(rows) == 1.0

    def test_fallback_when_counts_missing(self):
        """When counts are None, falls back to counting evidence_refs."""
        rows = [
            _row(evidence_refs=["ev1", "ev2"], stale_evidence_count=None, fresh_evidence_count=None),
        ]
        # No per-evidence date info → assumes all fresh → 0.0
        assert stale_evidence_rate(rows) == 0.0

    def test_empty_input(self):
        assert stale_evidence_rate([]) == 0.0


# ── evidence_coverage ──────────────────────────────────────────────────


class TestEvidenceCoverage:
    def test_all_have_evidence(self):
        rows = [
            _row(evidence_refs=["ev1"]),
            _row(evidence_refs=["ev2", "ev3"]),
        ]
        assert evidence_coverage(rows) == 1.0

    def test_none_have_evidence(self):
        rows = [
            _row(evidence_refs=[]),
            _row(evidence_refs=[]),
        ]
        assert evidence_coverage(rows) == 0.0

    def test_partial(self):
        rows = [
            _row(evidence_refs=["ev1"]),
            _row(evidence_refs=[]),
            _row(evidence_refs=["ev2"]),
        ]
        assert evidence_coverage(rows) == pytest.approx(2 / 3)

    def test_empty_input(self):
        assert evidence_coverage([]) == 0.0


# ── compute_grounding_metrics ──────────────────────────────────────────


class TestComputeGroundingMetrics:
    def test_returns_expected_keys(self):
        rows = [_row(evidence_refs=["ev1"], claim_refs=["c1"])]
        result = compute_grounding_metrics(rows)
        expected = {
            "citation_precision",
            "claim_coverage",
            "unsupported_claim_rate",
            "stale_evidence_rate",
            "evidence_coverage",
        }
        assert set(result.keys()) == expected

    def test_empty_input(self):
        result = compute_grounding_metrics([])
        assert all(v == 0.0 for v in result.values())


# ── grounding_metrics_by_system ────────────────────────────────────────


class TestGroundingMetricsBySystem:
    def test_groups_by_system(self):
        df = pd.DataFrame([
            _row(system="kg_dynamic", evidence_refs=["ev1"], claim_refs=["c1"]),
            _row(system="kg_dynamic", evidence_refs=["ev2"]),
            _row(system="static_kg", evidence_refs=[]),
            _row(system="static_kg", evidence_refs=["ev3"]),
        ])
        result = grounding_metrics_by_system(df)
        assert "kg_dynamic" in result.index
        assert "static_kg" in result.index
        assert result.loc["kg_dynamic", "evidence_coverage"] == 1.0
        assert result.loc["static_kg", "evidence_coverage"] == 0.5


# ── Missing fields safety ─────────────────────────────────────────────


class TestMissingFieldsSafety:
    """Verify functions don't crash when fields are missing or malformed."""

    def test_string_encoded_list(self):
        rows = [{"evidence_refs": "['ev1', 'ev2']", "claim_refs": "[]", "decision_reason": "r"}]
        # Should not crash
        result = compute_grounding_metrics(rows)
        assert isinstance(result, dict)

    def test_none_fields(self):
        rows = [{"evidence_refs": None, "claim_refs": None, "decision_reason": None}]
        result = compute_grounding_metrics(rows)
        assert isinstance(result, dict)

    def test_missing_keys(self):
        rows = [{"ticker": "0700.HK"}]
        result = compute_grounding_metrics(rows)
        assert isinstance(result, dict)
