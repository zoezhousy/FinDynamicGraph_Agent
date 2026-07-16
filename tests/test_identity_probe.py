"""Tests for src/probes/identity_probe.py.

Tests the anonymization and diff logic in isolation (no Neo4j/LLM).
"""

from __future__ import annotations

import pytest

from src.probes.identity_probe import (
    ANON_MAP,
    anonymize_subgraph,
    diff_decisions,
    _replace_name,
    _approx_differs,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _mini_subgraph(ticker: str = "0700.HK", name: str = "Tencent Holdings") -> dict:
    """Build a minimal but realistic subgraph for testing."""
    return {
        "company": [{"ticker": ticker, "name": name, "sector": "Technology"}],
        "signals": [
            {
                "entity_id": f"sig:{ticker}:ma20:2025-03-01",
                "ticker": ticker,
                "name": "price_below_ma20",
                "direction": "bearish",
                "strength": 0.7,
                "as_of_date": "2025-03-01",
            },
        ],
        "news": [
            {
                "entity_id": f"news:{ticker}:001",
                "ticker": ticker,
                "headline": f"{name} reports strong Q4 earnings, beats estimates",
                "published_at": "2025-02-28",
                "evidence_id": f"ev:{ticker}:news:001",
            },
        ],
        "fundamentals": [
            {
                "entity_id": f"fund:{ticker}:roe",
                "ticker": ticker,
                "metric": "returnonequity",
                "numeric_value": 0.22,
                "direction": "bullish",
                "strength": 0.65,
            },
        ],
        "risks": [
            {
                "entity_id": f"risk:{ticker}:reg",
                "ticker": ticker,
                "risk_type": "regulatory",
                "direction": "bearish",
                "severity": 0.5,
            },
        ],
        "evidences": [
            {"evidence_id": f"ev:{ticker}:news:001", "published_at": "2025-02-28"},
        ],
        "sources": [],
        "claims": [
            {
                "entity_id": f"claim:{ticker}:001",
                "claim_id": f"claim:{ticker}:001",
                "polarity": "supports",
                "claim_type": "sentiment",
                "confidence": 0.8,
            },
        ],
        "conflicts": [],
    }


def _mini_decision(
    action: str = "buy",
    score: float = 0.25,
    confidence: float = 0.65,
    agents: list | None = None,
) -> dict:
    """Build a minimal decision dict."""
    return {
        "action": action,
        "final_score": score,
        "confidence": confidence,
        "conflict_level": 0.0,
        "_agent_reports": agents or [
            {"role": "technical", "stance": "bullish", "confidence": 0.7, "score": 0.4, "summary": "MA20 crossover"},
            {"role": "news", "stance": "neutral", "confidence": 0.5, "score": 0.0, "summary": "Mixed signals"},
            {"role": "fundamental", "stance": "bullish", "confidence": 0.6, "score": 0.3, "summary": "Strong ROE"},
            {"role": "risk", "stance": "neutral", "confidence": 0.5, "score": -0.1, "summary": "Moderate risk"},
        ],
    }


# ── Anonymization ──────────────────────────────────────────────────────


class TestAnonymizeSubgraph:
    def test_ticker_replaced_in_company(self):
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        assert anon["company"][0]["ticker"] == "BETA.HK"

    def test_company_name_replaced(self):
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        assert anon["company"][0]["name"] == "BetaTech Holdings"

    def test_ticker_replaced_in_all_nodes(self):
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        for node in anon["signals"]:
            assert node["ticker"] == "BETA.HK"
        for node in anon["news"]:
            assert node["ticker"] == "BETA.HK"
        for node in anon["fundamentals"]:
            assert node["ticker"] == "BETA.HK"

    def test_headline_text_replaced(self):
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        assert "Tencent" not in anon["news"][0]["headline"]
        assert "BetaTech" in anon["news"][0]["headline"]

    def test_numeric_values_preserved(self):
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        assert anon["signals"][0]["strength"] == 0.7
        assert anon["fundamentals"][0]["numeric_value"] == 0.22
        assert anon["risks"][0]["severity"] == 0.5

    def test_entity_ids_preserved(self):
        """Entity IDs are opaque refs — should not be anonymized."""
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        assert anon["signals"][0]["entity_id"] == "sig:0700.HK:ma20:2025-03-01"

    def test_dates_preserved(self):
        sg = _mini_subgraph()
        anon = anonymize_subgraph(sg, "0700.HK")
        assert anon["signals"][0]["as_of_date"] == "2025-03-01"
        assert anon["news"][0]["published_at"] == "2025-02-28"

    def test_does_not_mutate_original(self):
        sg = _mini_subgraph()
        original_ticker = sg["company"][0]["ticker"]
        anonymize_subgraph(sg, "0700.HK")
        assert sg["company"][0]["ticker"] == original_ticker

    def test_hsbc_mapping(self):
        sg = _mini_subgraph(ticker="0005.HK", name="HSBC Holdings")
        anon = anonymize_subgraph(sg, "0005.HK")
        assert anon["company"][0]["ticker"] == "ALPHA.HK"
        assert "HSBC" not in anon["company"][0]["name"]
        assert "AlphaCorp" in anon["company"][0]["name"]

    def test_aia_mapping(self):
        sg = _mini_subgraph(ticker="1299.HK", name="AIA Group")
        anon = anonymize_subgraph(sg, "1299.HK")
        assert anon["company"][0]["ticker"] == "GAMMA.HK"
        assert "AIA" not in anon["company"][0]["name"]
        assert "GammaHoldings" in anon["company"][0]["name"]

    def test_unknown_ticker_raises(self):
        sg = _mini_subgraph()
        with pytest.raises(ValueError, match="No anonymization mapping"):
            anonymize_subgraph(sg, "9999.HK")


# ── Name replacement ───────────────────────────────────────────────────


class TestReplaceName:
    def test_case_insensitive(self):
        assert "BetaTech" in _replace_name("TENCENT reports earnings", "0700.HK")

    def test_partial_match_respects_word_boundary(self):
        # "Tencentia" should NOT be replaced
        result = _replace_name("Tencentia is not Tencent", "0700.HK")
        assert "Tencentia" in result
        assert "BetaTech" in result

    def test_no_match_returns_original(self):
        text = "Some unrelated text"
        assert _replace_name(text, "0700.HK") == text


# ── Diff logic ─────────────────────────────────────────────────────────


class TestDiffDecisions:
    def test_identical_decisions(self):
        d = _mini_decision()
        diff = diff_decisions(d, d)
        assert diff["identical"] is True
        assert diff["top_level_diffs"] == []
        assert diff["agent_diffs"] == []

    def test_different_action(self):
        real = _mini_decision(action="buy")
        anon = _mini_decision(action="hold")
        diff = diff_decisions(real, anon)
        assert diff["identical"] is False
        assert any(d["field"] == "action" for d in diff["top_level_diffs"])

    def test_different_confidence(self):
        real = _mini_decision(confidence=0.80)
        anon = _mini_decision(confidence=0.55)
        diff = diff_decisions(real, anon)
        assert diff["identical"] is False
        assert any(d["field"] == "confidence" for d in diff["top_level_diffs"])

    def test_different_agent_stance(self):
        real_agents = [
            {"role": "technical", "stance": "bullish", "confidence": 0.7, "score": 0.4, "summary": "x"},
            {"role": "news", "stance": "neutral", "confidence": 0.5, "score": 0.0, "summary": "y"},
            {"role": "fundamental", "stance": "bullish", "confidence": 0.6, "score": 0.3, "summary": "z"},
            {"role": "risk", "stance": "neutral", "confidence": 0.5, "score": -0.1, "summary": "w"},
        ]
        anon_agents = list(real_agents)
        anon_agents[0] = {**real_agents[0], "stance": "bearish"}  # technical flips

        real = _mini_decision(agents=real_agents)
        anon = _mini_decision(agents=anon_agents)
        diff = diff_decisions(real, anon)

        assert diff["identical"] is False
        tech_diffs = [a for a in diff["agent_diffs"] if a["role"] == "technical"]
        assert len(tech_diffs) == 1
        assert tech_diffs[0]["stance_real"] == "bullish"
        assert tech_diffs[0]["stance_anon"] == "bearish"

    def test_small_score_difference_within_tolerance(self):
        real = _mini_decision(score=0.250)
        anon = _mini_decision(score=0.253)
        diff = diff_decisions(real, anon)
        # Within 0.005 tolerance
        assert not any(d["field"] == "final_score" for d in diff["top_level_diffs"])


# ── Approx differs ─────────────────────────────────────────────────────


class TestApproxDiffers:
    def test_same_value(self):
        assert _approx_differs(0.5, 0.5) is False

    def test_within_tolerance(self):
        assert _approx_differs(0.5, 0.504) is False

    def test_outside_tolerance(self):
        assert _approx_differs(0.5, 0.51) is True

    def test_string_comparison(self):
        assert _approx_differs("buy", "hold") is True
        assert _approx_differs("buy", "buy") is False

    def test_none_values(self):
        assert _approx_differs(None, None) is False
        assert _approx_differs(None, 0.5) is True
