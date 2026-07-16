"""Tests for retrieved evidence/claim ID persistence in decision records.

Verifies that each system correctly populates retrieved_evidence_ids and
retrieved_claim_ids from the subgraph query, and that these fields survive
serialization to DataFrame and parquet/csv.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.agents.roles import AgentReport, extract_retrieved_ids, portfolio_manager_decide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subgraph(
    evidence_ids: list[str] | None = None,
    claim_ids: list[str] | None = None,
    signal_ids: list[str] | None = None,
    news_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
) -> dict:
    """Build a minimal subgraph dict with specified IDs."""
    sg: dict = {
        "company": [{"ticker": "TEST.HK"}],
        "signals": [],
        "fundamentals": [],
        "risks": [],
        "news": [],
        "evidences": [],
        "sources": [],
        "claims": [],
        "conflicts": [],
    }
    for eid in (evidence_ids or []):
        sg["evidences"].append({"evidence_id": eid, "title": f"Evidence {eid}"})
    for cid in (claim_ids or []):
        sg["claims"].append({"entity_id": cid, "claim_text": f"Claim {cid}"})
    for sid in (signal_ids or []):
        sg["signals"].append({"entity_id": sid, "indicator": "rsi"})
    for nid in (news_ids or []):
        sg["news"].append({"evidence_id": nid, "title": f"News {nid}"})
    for src_id in (source_ids or []):
        sg["sources"].append({"source_id": src_id, "source_type": "news"})
    return sg


def _make_report(role: str = "news", evidence_refs=None, claim_refs=None) -> AgentReport:
    return AgentReport(
        role=role,
        stance="bullish",
        confidence=0.7,
        score=0.3,
        summary="test",
        evidence_refs=evidence_refs or [],
        claim_refs=claim_refs or [],
        factors=[],
    )


# ---------------------------------------------------------------------------
# Tests for extract_retrieved_ids
# ---------------------------------------------------------------------------


class TestExtractRetrievedIds:
    """Unit tests for the extract_retrieved_ids helper."""

    def test_multiple_evidence(self):
        sg = _make_subgraph(evidence_ids=["ev1", "ev2", "ev3"])
        ev_ids, cl_ids = extract_retrieved_ids(sg)
        assert ev_ids == ["ev1", "ev2", "ev3"]
        assert cl_ids == []

    def test_multiple_claims(self):
        sg = _make_subgraph(claim_ids=["cl1", "cl2"])
        ev_ids, cl_ids = extract_retrieved_ids(sg)
        assert ev_ids == []
        assert cl_ids == ["cl1", "cl2"]

    def test_mixed_evidence_and_claims(self):
        sg = _make_subgraph(
            evidence_ids=["ev_a", "ev_b"],
            claim_ids=["cl_x"],
        )
        ev_ids, cl_ids = extract_retrieved_ids(sg)
        assert ev_ids == ["ev_a", "ev_b"]
        assert cl_ids == ["cl_x"]

    def test_deduplicated(self):
        sg = _make_subgraph(evidence_ids=["ev1", "ev1", "ev2"])
        ev_ids, _ = extract_retrieved_ids(sg)
        assert ev_ids == ["ev1", "ev2"]

    def test_none_subgraph(self):
        ev_ids, cl_ids = extract_retrieved_ids(None)
        assert ev_ids == []
        assert cl_ids == []

    def test_empty_subgraph(self):
        sg = _make_subgraph()
        ev_ids, cl_ids = extract_retrieved_ids(sg)
        assert ev_ids == []
        assert cl_ids == []

    def test_stable_sorted_order(self):
        sg = _make_subgraph(evidence_ids=["ev_c", "ev_a", "ev_b"])
        ev_ids, _ = extract_retrieved_ids(sg)
        assert ev_ids == sorted(ev_ids)

    def test_signals_included(self):
        sg = _make_subgraph(signal_ids=["sig:rsi:2025-01-01", "sig:ma50:2025-01-01"])
        ev_ids, _ = extract_retrieved_ids(sg)
        assert "sig:rsi:2025-01-01" in ev_ids
        assert "sig:ma50:2025-01-01" in ev_ids

    def test_news_evidence_included(self):
        sg = _make_subgraph(news_ids=["news:TEST.HK:abc123"])
        ev_ids, _ = extract_retrieved_ids(sg)
        assert "news:TEST.HK:abc123" in ev_ids

    def test_sources_included(self):
        sg = _make_subgraph(source_ids=["source:news:TEST.HK:def456"])
        ev_ids, _ = extract_retrieved_ids(sg)
        assert "source:news:TEST.HK:def456" in ev_ids

    def test_whitespace_stripped(self):
        sg = _make_subgraph(evidence_ids=[" ev1 ", "  ev2  "])
        ev_ids, _ = extract_retrieved_ids(sg)
        assert ev_ids == ["ev1", "ev2"]


# ---------------------------------------------------------------------------
# Tests for portfolio_manager_decide integration
# ---------------------------------------------------------------------------


class TestPortfolioManagerDecideRetrievedIds:
    """Verify that portfolio_manager_decide includes retrieved IDs."""

    def test_retrieved_ids_in_decision(self):
        sg = _make_subgraph(
            evidence_ids=["ev1", "ev2"],
            claim_ids=["cl1"],
        )
        reports = [_make_report(evidence_refs=["ev1"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        assert "retrieved_evidence_ids" in decision
        assert "retrieved_claim_ids" in decision
        assert "ev1" in decision["retrieved_evidence_ids"]
        assert "ev2" in decision["retrieved_evidence_ids"]
        assert decision["retrieved_claim_ids"] == ["cl1"]

    def test_agent_refs_subset_of_retrieved(self):
        """Agent only cites ev1, but retrieved set contains ev1 and ev2."""
        sg = _make_subgraph(evidence_ids=["ev1", "ev2"])
        reports = [_make_report(evidence_refs=["ev1"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        assert decision["evidence_refs"] == ["ev1"]
        assert set(decision["retrieved_evidence_ids"]) == {"ev1", "ev2"}

    def test_no_subgraph_retrieved_ids_empty(self):
        reports = [_make_report()]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=None)
        assert decision["retrieved_evidence_ids"] == []
        assert decision["retrieved_claim_ids"] == []

    def test_empty_subgraph_retrieved_ids_empty(self):
        sg = _make_subgraph()
        reports = [_make_report()]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        assert decision["retrieved_evidence_ids"] == []
        assert decision["retrieved_claim_ids"] == []


# ---------------------------------------------------------------------------
# Tests for system-level behavior
# ---------------------------------------------------------------------------


class TestSystemRetrievedIds:
    """Verify correct retrieved set behavior per system type."""

    def test_kg_dynamic_has_retrieved_ids(self):
        """KG dynamic system should have non-empty retrieved IDs when subgraph has data."""
        sg = _make_subgraph(
            evidence_ids=["ev1", "ev2", "ev3"],
            claim_ids=["cl1"],
        )
        reports = [_make_report(evidence_refs=["ev1", "ev2"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        assert len(decision["retrieved_evidence_ids"]) == 3
        assert len(decision["retrieved_claim_ids"]) == 1

    def test_static_kg_has_own_retrieved_ids(self):
        """Static_kg uses its own frozen subgraph — different from kg_dynamic."""
        # Simulating: static subgraph has ev1, ev2; dynamic would have ev1, ev2, ev3
        sg_static = _make_subgraph(evidence_ids=["ev1", "ev2"])
        reports = [_make_report(evidence_refs=["ev1"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg_static)
        assert decision["retrieved_evidence_ids"] == ["ev1", "ev2"]

    def test_no_kg_system_has_empty_retrieved(self):
        """no_kg_no_evidence system should have empty retrieved lists."""
        from src.eval.baselines import baseline_no_kg_no_evidence
        # We can't easily call this without LLM, but _build_baseline_result defaults
        # to empty lists when not provided
        from src.eval.baselines import _build_baseline_result
        result = _build_baseline_result(
            ticker="TEST.HK",
            trade_date=datetime(2025, 6, 1),
            action="abstain",
            final_score=0.0,
            confidence=0.3,
            conflict_level=0.0,
            reason="test",
            evidence_refs=[],
            agent_reports=[],
            baseline="no_kg_no_evidence",
        )
        assert result["retrieved_evidence_ids"] == []
        assert result["retrieved_claim_ids"] == []


# ---------------------------------------------------------------------------
# Tests for DataFrame / serialization persistence
# ---------------------------------------------------------------------------


class TestRetrievedIdsSerialization:
    """Verify fields survive DataFrame and parquet/csv round-trip."""

    def test_in_dataframe(self):
        sg = _make_subgraph(evidence_ids=["ev1", "ev2"], claim_ids=["cl1"])
        reports = [_make_report(evidence_refs=["ev1"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        # Simulate what main_experiment does
        row = {**decision, "system": "kg_dynamic"}
        df = pd.DataFrame([row])
        assert "retrieved_evidence_ids" in df.columns
        assert "retrieved_claim_ids" in df.columns

    def test_parquet_roundtrip(self, tmp_path):
        sg = _make_subgraph(evidence_ids=["ev1", "ev2"], claim_ids=["cl1"])
        reports = [_make_report(evidence_refs=["ev1"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        row = {**decision, "system": "kg_dynamic"}
        df = pd.DataFrame([row])

        # Lists become strings in parquet via _make_serializable
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)

        assert "retrieved_evidence_ids" in loaded.columns
        assert "retrieved_claim_ids" in loaded.columns
        # Values are string-encoded lists — verify they're recoverable
        val = loaded.iloc[0]["retrieved_evidence_ids"]
        assert isinstance(val, str)
        assert "ev1" in val

    def test_csv_roundtrip(self, tmp_path):
        sg = _make_subgraph(evidence_ids=["ev1"], claim_ids=["cl1"])
        reports = [_make_report(evidence_refs=["ev1"])]
        decision = portfolio_manager_decide("TEST.HK", datetime(2025, 6, 1), reports, subgraph=sg)
        row = {**decision, "system": "kg_dynamic"}
        df = pd.DataFrame([row])

        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)

        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)

        assert "retrieved_evidence_ids" in loaded.columns
        assert "retrieved_claim_ids" in loaded.columns

    def test_different_dates_no_cross_contamination(self):
        """Each trade_date should have its own retrieved set."""
        sg1 = _make_subgraph(evidence_ids=["ev_jan1", "ev_jan2"])
        sg2 = _make_subgraph(evidence_ids=["ev_feb1"])

        reports1 = [_make_report(evidence_refs=["ev_jan1"])]
        reports2 = [_make_report(evidence_refs=["ev_feb1"])]

        d1 = portfolio_manager_decide("TEST.HK", datetime(2025, 1, 1), reports1, subgraph=sg1)
        d2 = portfolio_manager_decide("TEST.HK", datetime(2025, 2, 1), reports2, subgraph=sg2)

        assert "ev_jan1" in d1["retrieved_evidence_ids"]
        assert "ev_feb1" not in d1["retrieved_evidence_ids"]
        assert "ev_feb1" in d2["retrieved_evidence_ids"]
        assert "ev_jan1" not in d2["retrieved_evidence_ids"]

    def test_different_tickers_no_cross_contamination(self):
        """Each ticker should have its own retrieved set."""
        sg_a = _make_subgraph(evidence_ids=["ev_A1"])
        sg_b = _make_subgraph(evidence_ids=["ev_B1"])

        reports_a = [_make_report(evidence_refs=["ev_A1"])]
        reports_b = [_make_report(evidence_refs=["ev_B1"])]

        da = portfolio_manager_decide("AAA.HK", datetime(2025, 6, 1), reports_a, subgraph=sg_a)
        db = portfolio_manager_decide("BBB.HK", datetime(2025, 6, 1), reports_b, subgraph=sg_b)

        assert da["retrieved_evidence_ids"] == ["ev_A1"]
        assert db["retrieved_evidence_ids"] == ["ev_B1"]
