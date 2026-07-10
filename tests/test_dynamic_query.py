"""Tests for dynamic KG query and temporal filtering logic.

These tests validate the pure-Python query helpers in src.kg.query and
the contradiction detection in src.kg.update_pipeline — no Neo4j required.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.kg.query import (
    compute_snapshot_summary_from_lists,
    filter_claims_by_date,
    filter_evidence_by_date,
    filter_signals_by_date,
    find_conflict_pairs,
)
from src.kg.update_pipeline import (
    build_conflict_relations,
    detect_contradictory_claims,
    mark_expired_entities,
)
from src.kg.schema import Entity


# ── Fixtures ───────────────────────────────────────────────────────────


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _make_claim(
    claim_id: str,
    ticker: str = "0700.HK",
    polarity: str = "supports",
    as_of: str = "2024-06-01",
    valid_from: str | None = "2024-06-01",
    valid_to: str | None = None,
    is_active: bool = True,
    confidence: float = 0.7,
    claim_type: str = "news",
) -> dict:
    return {
        "claim_id": claim_id,
        "ticker": ticker,
        "claim_type": claim_type,
        "polarity": polarity,
        "confidence": confidence,
        "as_of_date": as_of,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "is_active": is_active,
        "text": f"Claim {claim_id}",
    }


def _make_evidence(
    evidence_id: str,
    published_at: str = "2024-06-01",
    valid_from: str | None = "2024-06-01",
    valid_to: str | None = None,
    is_active: bool = True,
) -> dict:
    return {
        "evidence_id": evidence_id,
        "published_at": published_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "is_active": is_active,
        "source_name": "test_source",
    }


def _make_signal(
    signal_id: str,
    as_of: str = "2024-06-01",
    valid_from: str | None = "2024-06-01",
    valid_to: str | None = None,
    is_active: bool = True,
) -> dict:
    return {
        "signal_id": signal_id,
        "as_of_date": as_of,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "is_active": is_active,
    }


# ── Test: filter_claims_by_date ────────────────────────────────────────


class TestFilterClaimsByDate:
    """as_of_date query should not return future evidence."""

    def test_returns_claims_valid_at_as_of(self):
        claims = [_make_claim("c1", as_of="2024-06-01", valid_from="2024-06-01")]
        result = filter_claims_by_date(claims, _dt("2024-06-01"))
        assert len(result) == 1
        assert result[0]["claim_id"] == "c1"

    def test_excludes_future_claims(self):
        claims = [_make_claim("c1", as_of="2024-07-01", valid_from="2024-07-01")]
        result = filter_claims_by_date(claims, _dt("2024-06-01"))
        assert len(result) == 0

    def test_excludes_expired_claims(self):
        """Old signal with valid_to should not appear after expiry."""
        claims = [_make_claim("c1", valid_from="2024-01-01", valid_to="2024-03-01")]
        result = filter_claims_by_date(claims, _dt("2024-06-01"))
        assert len(result) == 0

    def test_includes_claims_inside_valid_window(self):
        """Active signal should appear inside valid window."""
        claims = [_make_claim("c1", valid_from="2024-01-01", valid_to="2024-12-01")]
        result = filter_claims_by_date(claims, _dt("2024-06-01"))
        assert len(result) == 1

    def test_excludes_inactive_claims(self):
        claims = [_make_claim("c1", is_active=False)]
        result = filter_claims_by_date(claims, _dt("2024-06-01"))
        assert len(result) == 0

    def test_handles_none_valid_from(self):
        claims = [_make_claim("c1", valid_from=None)]
        result = filter_claims_by_date(claims, _dt("2024-06-01"))
        assert len(result) == 1


# ── Test: filter_evidence_by_date ──────────────────────────────────────


class TestFilterEvidenceByDate:
    def test_excludes_future_evidence(self):
        evs = [_make_evidence("ev1", published_at="2024-07-01")]
        result = filter_evidence_by_date(evs, _dt("2024-06-01"))
        assert len(result) == 0

    def test_includes_past_evidence(self):
        evs = [_make_evidence("ev1", published_at="2024-05-01")]
        result = filter_evidence_by_date(evs, _dt("2024-06-01"))
        assert len(result) == 1

    def test_excludes_expired_evidence(self):
        evs = [_make_evidence("ev1", valid_from="2024-01-01", valid_to="2024-03-01")]
        result = filter_evidence_by_date(evs, _dt("2024-06-01"))
        assert len(result) == 0

    def test_excludes_inactive_evidence(self):
        evs = [_make_evidence("ev1", is_active=False)]
        result = filter_evidence_by_date(evs, _dt("2024-06-01"))
        assert len(result) == 0


# ── Test: filter_signals_by_date ───────────────────────────────────────


class TestFilterSignalsByDate:
    def test_excludes_future_signals(self):
        sigs = [_make_signal("s1", as_of="2024-07-01")]
        result = filter_signals_by_date(sigs, _dt("2024-06-01"))
        assert len(result) == 0

    def test_includes_past_signals(self):
        sigs = [_make_signal("s1", as_of="2024-05-01")]
        result = filter_signals_by_date(sigs, _dt("2024-06-01"))
        assert len(result) == 1

    def test_excludes_expired_signals(self):
        sigs = [_make_signal("s1", valid_from="2024-01-01", valid_to="2024-03-01")]
        result = filter_signals_by_date(sigs, _dt("2024-06-01"))
        assert len(result) == 0

    def test_includes_active_signals_in_window(self):
        sigs = [_make_signal("s1", valid_from="2024-01-01", valid_to="2024-12-01")]
        result = filter_signals_by_date(sigs, _dt("2024-06-01"))
        assert len(result) == 1


# ── Test: Contradiction / Conflict detection ───────────────────────────


class TestContradictionDetection:
    """Contradictory claims are preserved, not overwritten."""

    def test_bullish_bearish_detected_as_conflict(self):
        new = [Entity(
            entity_id="claim:new",
            type="Claim",
            properties={
                "ticker": "0700.HK",
                "claim_type": "news",
                "polarity": "supports",
                "as_of_date": "2024-06-01",
                "is_active": True,
            },
        )]
        existing = [Entity(
            entity_id="claim:old",
            type="Claim",
            properties={
                "ticker": "0700.HK",
                "claim_type": "news",
                "polarity": "contradicts",
                "as_of_date": "2024-05-15",
                "is_active": True,
            },
        )]
        conflicts = detect_contradictory_claims(new, existing)
        assert len(conflicts) == 1
        assert conflicts[0][0] == "claim:new"
        assert conflicts[0][1] == "claim:old"

    def test_same_polarity_no_conflict(self):
        new = [Entity(
            entity_id="claim:new",
            type="Claim",
            properties={
                "ticker": "0700.HK",
                "claim_type": "news",
                "polarity": "supports",
                "as_of_date": "2024-06-01",
                "is_active": True,
            },
        )]
        existing = [Entity(
            entity_id="claim:old",
            type="Claim",
            properties={
                "ticker": "0700.HK",
                "claim_type": "news",
                "polarity": "supports",
                "as_of_date": "2024-05-15",
                "is_active": True,
            },
        )]
        conflicts = detect_contradictory_claims(new, existing)
        assert len(conflicts) == 0

    def test_inactive_existing_no_conflict(self):
        new = [Entity(
            entity_id="claim:new",
            type="Claim",
            properties={
                "ticker": "0700.HK",
                "claim_type": "news",
                "polarity": "supports",
                "as_of_date": "2024-06-01",
                "is_active": True,
            },
        )]
        existing = [Entity(
            entity_id="claim:old",
            type="Claim",
            properties={
                "ticker": "0700.HK",
                "claim_type": "news",
                "polarity": "contradicts",
                "as_of_date": "2024-05-15",
                "is_active": False,
            },
        )]
        conflicts = detect_contradictory_claims(new, existing)
        assert len(conflicts) == 0

    def test_conflict_relation_can_be_queried(self):
        """Test that find_conflict_pairs works on in-memory claims."""
        claims = [
            _make_claim("c1", polarity="supports", claim_type="news"),
            _make_claim("c2", polarity="contradicts", claim_type="news"),
        ]
        pairs = find_conflict_pairs(claims)
        assert len(pairs) == 1
        ids = {pairs[0]["claim_a"], pairs[0]["claim_b"]}
        assert "c1" in ids
        assert "c2" in ids

    def test_build_conflict_relations(self):
        conflicts = [("claim:new", "claim:old", _dt("2024-06-01"))]
        rels = build_conflict_relations(conflicts)
        assert len(rels) == 2  # bidirectional
        types = {r.type for r in rels}
        assert "CONFLICTS_WITH" in types


# ── Test: mark_expired_entities ────────────────────────────────────────


class TestMarkExpired:
    def test_sets_valid_to_and_is_active_false(self):
        entities = [Entity(
            entity_id="e1",
            type="Claim",
            properties={"ticker": "0700.HK", "is_active": True},
        )]
        expired = mark_expired_entities(entities, _dt("2024-06-01"))
        assert len(expired) == 1
        assert expired[0].properties["is_active"] is False
        assert expired[0].properties["valid_to"] == "2024-06-01T00:00:00"


# ── Test: compute_snapshot_summary_from_lists ──────────────────────────


class TestSnapshotSummary:
    def test_returns_expected_keys(self):
        summary = compute_snapshot_summary_from_lists(
            ticker="0700.HK",
            as_of_date=_dt("2024-06-01"),
            claims=[],
            evidences=[],
            signals=[],
        )
        expected_keys = {
            "ticker", "as_of_date", "active_signal_count", "active_claim_count",
            "bullish_claim_count", "bearish_claim_count", "neutral_claim_count",
            "supporting_claim_count", "contradicting_claim_count",
            "evidence_count", "fresh_evidence_count", "stale_evidence_count",
            "latest_evidence_dates", "conflict_count", "top_claims", "top_sources",
        }
        assert set(summary.keys()) == expected_keys

    def test_counts_correct_with_data(self):
        claims = [
            _make_claim("c1", polarity="supports"),
            _make_claim("c2", polarity="contradicts"),
            _make_claim("c3", polarity="neutral"),
        ]
        evidences = [
            _make_evidence("ev1", published_at="2024-05-01"),
            _make_evidence("ev2", published_at="2024-05-15"),
        ]
        signals = [_make_signal("s1")]

        summary = compute_snapshot_summary_from_lists(
            ticker="0700.HK",
            as_of_date=_dt("2024-06-01"),
            claims=claims,
            evidences=evidences,
            signals=signals,
        )
        assert summary["active_claim_count"] == 3
        assert summary["bullish_claim_count"] == 1
        assert summary["bearish_claim_count"] == 1
        assert summary["neutral_claim_count"] == 1
        assert summary["active_signal_count"] == 1
        assert summary["evidence_count"] == 2
        assert summary["stale_evidence_count"] == 2  # both before as_of

    def test_future_evidence_excluded(self):
        evidences = [_make_evidence("ev1", published_at="2024-07-01")]
        summary = compute_snapshot_summary_from_lists(
            ticker="0700.HK",
            as_of_date=_dt("2024-06-01"),
            claims=[],
            evidences=evidences,
            signals=[],
        )
        assert summary["evidence_count"] == 0

    def test_conflict_count_with_contradictory_claims(self):
        claims = [
            _make_claim("c1", polarity="supports", claim_type="news"),
            _make_claim("c2", polarity="contradicts", claim_type="news"),
        ]
        summary = compute_snapshot_summary_from_lists(
            ticker="0700.HK",
            as_of_date=_dt("2024-06-01"),
            claims=claims,
            evidences=[],
            signals=[],
        )
        assert summary["conflict_count"] == 1

    def test_no_conflict_when_same_polarity(self):
        claims = [
            _make_claim("c1", polarity="supports", claim_type="news"),
            _make_claim("c2", polarity="supports", claim_type="news"),
        ]
        summary = compute_snapshot_summary_from_lists(
            ticker="0700.HK",
            as_of_date=_dt("2024-06-01"),
            claims=claims,
            evidences=[],
            signals=[],
        )
        assert summary["conflict_count"] == 0
