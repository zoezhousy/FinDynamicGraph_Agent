from __future__ import annotations

from datetime import datetime

import pytest

from src.eval.temporal_audit import (
    TemporalLeakageError,
    assert_no_future,
    audit_items,
    audit_subgraph,
    snapshot_fingerprint,
)


def test_audit_counts_valid_stale_and_future() -> None:
    audit = audit_items(
        [
            {"evidence_id": "fresh", "published_at": "2025-05-25"},
            {"evidence_id": "old", "published_at": "2025-01-01"},
            {"evidence_id": "expired", "valid_from": "2025-01-01", "valid_to": "2025-05-01"},
            {"evidence_id": "future", "effective_at": "2025-07-01"},
        ],
        datetime(2025, 6, 1),
        staleness_days=30,
    )
    assert audit.retrieved_count == 4
    assert audit.valid_count == 2
    assert audit.stale_count == 2
    assert audit.future_count == 1
    assert audit.temporal_precision == 0.5
    assert audit.future_evidence_rate == 0.25


def test_future_assertion_fails_with_ids() -> None:
    audit = audit_items(
        [{"evidence_id": "future", "valid_from": "2025-06-02"}],
        datetime(2025, 6, 1),
    )
    with pytest.raises(TemporalLeakageError, match="future"):
        assert_no_future(audit, context="kg_dynamic")


def test_subgraph_flattening_and_deduplication() -> None:
    subgraph = {
        "evidences": [
            {"evidence_id": "ev1", "published_at": "2025-05-01"},
            {"evidence_id": "ev1", "published_at": "2025-05-01"},
        ],
        "claims": [{"claim_id": "cl1", "valid_from": "2025-05-01"}],
    }
    audit = audit_subgraph(subgraph, datetime(2025, 6, 1))
    assert audit.retrieved_count == 2
    assert audit.future_count == 0


def test_snapshot_fingerprint_is_order_independent_and_changes_with_content() -> None:
    first = {
        "evidences": [
            {"evidence_id": "a", "published_at": "2025-01-01"},
            {"evidence_id": "b", "published_at": "2025-02-01"},
        ]
    }
    reordered = {"evidences": list(reversed(first["evidences"]))}
    changed = {
        "evidences": first["evidences"]
        + [{"evidence_id": "c", "published_at": "2025-03-01"}]
    }
    assert snapshot_fingerprint(first) == snapshot_fingerprint(reordered)
    assert snapshot_fingerprint(first) != snapshot_fingerprint(changed)


def test_current_is_active_does_not_change_historical_validity() -> None:
    audit = audit_items(
        [{
            "evidence_id": "historical",
            "valid_from": "2025-01-01",
            "valid_to": "2025-07-01",
            "is_active": False,
        }],
        datetime(2025, 6, 1),
    )
    assert audit.valid_count == 1
    assert audit.future_count == 0
