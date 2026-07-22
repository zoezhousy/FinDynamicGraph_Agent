"""KG query tools for LangGraph agents.

Provides:
- ``KGAgentContext`` — thin wrapper around ``KGQueryClient`` (legacy, used by orchestrator/baselines)
- ``DynamicDatePolicy`` / ``FixedCutoffPolicy`` — date resolution for dynamic vs static KG
- ``build_kg_tools`` — creates LangChain ``@tool`` functions with request-scoped caching
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.tools import tool

from src.kg.query import KGQueryClient

logger = logging.getLogger(__name__)


# ── Legacy context wrapper (kept for backward compatibility) ────────────


class KGAgentContext:
    def __init__(self, query_client: KGQueryClient) -> None:
        self.query_client = query_client

    def load_subgraph(self, ticker: str, trade_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        return self.query_client.get_ticker_subgraph(ticker, trade_date)

    def load_snapshot_summary(self, ticker: str, trade_date: datetime) -> Dict[str, Any]:
        return self.query_client.get_snapshot_summary(ticker, trade_date)


# ── Date policies ──────────────────────────────────────────────────────


@dataclass
class DynamicDatePolicy:
    """Use trade_date as-is (for kg_dynamic)."""

    def resolve(self, as_of_date: str) -> str:
        return as_of_date


@dataclass
class FixedCutoffPolicy:
    """Always query with a fixed cutoff date (for static_kg baseline)."""

    cutoff: str  # e.g. "2025-01-01"

    def resolve(self, as_of_date: str) -> str:
        return self.cutoff


# ── Tool builder ───────────────────────────────────────────────────────


def build_kg_tools(
    query_client: KGQueryClient,
    date_policy: DynamicDatePolicy | FixedCutoffPolicy,
) -> list:
    """Build LangChain tool set with request-scoped cache.

    Each (ticker, effective_date) pair executes at most one full KG query.
    Subsequent tool calls for the same pair return cached results.

    The cache is scoped to this ``build_kg_tools()`` call — create a fresh
    set per ``run_for_ticker_async()`` invocation so different trade dates
    don't share stale data.
    """
    _cache: dict[tuple[str, str], dict] = {}

    def _get_snapshot(ticker: str, as_of_date: str) -> dict:
        effective_str = date_policy.resolve(as_of_date)
        key = (ticker, effective_str)
        if key not in _cache:
            logger.debug("KG cache miss: %s as_of=%s (effective=%s)", ticker, as_of_date, effective_str)
            effective_dt = datetime.fromisoformat(effective_str)
            _cache[key] = query_client.get_ticker_subgraph(ticker, effective_dt)
        return _cache[key]

    @tool
    def query_subgraph(ticker: str, as_of_date: str) -> dict:
        """Load the full temporal KG subgraph for a ticker as of a date.

        Returns a dict with keys: company, signals, fundamentals, risks,
        news, evidences, sources, claims, conflicts.
        All data is temporally filtered to be valid on or before as_of_date.
        """
        return _get_snapshot(ticker, as_of_date)

    @tool
    def query_signals(ticker: str, as_of_date: str) -> list[dict]:
        """Get technical indicator signals (MA, RSI, MACD, etc.) from the KG."""
        return _get_snapshot(ticker, as_of_date).get("signals", [])

    @tool
    def query_fundamentals(ticker: str, as_of_date: str) -> list[dict]:
        """Get fundamental metrics (PE, margins, growth, etc.) from the KG."""
        return _get_snapshot(ticker, as_of_date).get("fundamentals", [])

    @tool
    def query_risks(ticker: str, as_of_date: str) -> list[dict]:
        """Get risk events from the KG."""
        return _get_snapshot(ticker, as_of_date).get("risks", [])

    @tool
    def query_news(ticker: str, as_of_date: str) -> list[dict]:
        """Get news events from the KG."""
        return _get_snapshot(ticker, as_of_date).get("news", [])

    @tool
    def query_claims(ticker: str, as_of_date: str) -> dict:
        """Get claims, evidence chains, and source documents from the KG.

        Returns a dict with keys: claims, evidences, sources, conflicts.
        """
        snap = _get_snapshot(ticker, as_of_date)
        return {
            "claims": snap.get("claims", []),
            "evidences": snap.get("evidences", []),
            "sources": snap.get("sources", []),
            "conflicts": snap.get("conflicts", []),
        }

    return [query_subgraph, query_signals, query_fundamentals,
            query_risks, query_news, query_claims]
