"""Baseline systems for comparison experiments.

Each baseline returns a dict compatible with the kg_dynamic decision format,
including: ticker, trade_date, action, final_score, confidence, conflict_level,
decision_reason, evidence_refs, supporting_roles, opposing_roles,
stale_evidence_count, fresh_evidence_count, evidence_alignment, trace,
agent_reports, baseline.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.agents.roles import (
    AgentReport,
    _normalize_stance,
    _safe_float,
    portfolio_manager_decide,
)
from src.llm.client import OpenAICompatibleClient
from src.llm.prompts import AGENT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Baseline 1: no_kg_no_evidence
# Pure LLM decision with only ticker + date, no data at all.
# ---------------------------------------------------------------------------

_NO_KG_SYSTEM_PROMPT = """\
You are a financial trading advisor.
You must decide whether to buy, sell, hold, or abstain from a stock.
You have NO access to any market data, news, technical indicators, or fundamental data.
You must rely ONLY on your general knowledge about the company.

Return valid JSON only. Do not wrap in markdown or backticks.
You must produce exactly these fields:
- action: buy | sell | hold | abstain
- confidence: float in [0, 1]
- score: float in [-1, 1]  (positive = bullish, negative = bearish)
- reason: short explanation of your decision
"""

_NO_KG_USER_PROMPT = """\
Ticker: {ticker}
Trade date: {trade_date}

Decide: buy, sell, hold, or abstain.
"""


def baseline_no_kg_no_evidence(ticker: str, trade_date: datetime) -> Dict[str, Any]:
    """Baseline 1: Pure LLM with no external data or KG."""
    client = OpenAICompatibleClient()
    try:
        payload = client.generate_json(
            _NO_KG_SYSTEM_PROMPT,
            _NO_KG_USER_PROMPT.format(
                ticker=ticker,
                trade_date=trade_date.date().isoformat(),
            ),
        )
    except Exception:
        # If LLM fails, abstain.
        payload = {}

    action = str(payload.get("action") or "abstain").lower()
    if action not in {"buy", "sell", "hold", "abstain"}:
        action = "abstain"

    confidence = _safe_float(payload.get("confidence"), 0.3, 0.0, 1.0)
    score = _safe_float(payload.get("score"), 0.0, -1.0, 1.0)
    reason = str(payload.get("reason") or "Pure LLM baseline with no data or KG.")

    return _build_baseline_result(
        ticker=ticker,
        trade_date=trade_date,
        action=action,
        final_score=score,
        confidence=confidence,
        conflict_level=0.0,
        reason=reason,
        evidence_refs=[],
        agent_reports=[],
        baseline="no_kg_no_evidence",
    )


# ---------------------------------------------------------------------------
# Baseline 2: evidence_no_kg
# LLM with raw news + OHLCV summary (unstructured text, no KG).
# ---------------------------------------------------------------------------

_EVIDENCE_SYSTEM_PROMPT = """\
You are a financial trading advisor.
You are given raw market data and recent news for a stock.
Decide whether to buy, sell, hold, or abstain.

Return valid JSON only. Do not wrap in markdown or backticks.
You must produce exactly these fields:
- stance: bullish | bearish | neutral | uncertain
- confidence: float in [0, 1]
- score: float in [-1, 1]
- summary: short explanation
- factors: array of objects with name, direction, weight
"""

_EVIDENCE_USER_PROMPT = """\
Ticker: {ticker}
Trade date: {trade_date}

Recent OHLCV summary:
{ohlcv_summary}

Recent news headlines:
{news_text}

Analyze the above information and decide: buy, sell, hold, or abstain.
Prefer conservative judgments when evidence is weak.
"""


def _filter_news_by_trade_date(
    df: pd.DataFrame,
    trade_date: datetime,
    max_items: int = 10,
) -> pd.DataFrame:
    """Filter news to only include items published on or before trade_date.

    Attempts to parse the published timestamp from the first available column
    in priority order: published_time, published_at, date.
    Rows with unparseable or missing timestamps are excluded.
    Rows published after trade_date are excluded.
    Results are sorted by published time descending (most recent first).
    """
    # Identify the time column
    time_col: str | None = None
    for candidate in ("published_time", "published_at", "date"):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        # No identifiable time column — cannot filter safely
        return pd.DataFrame()

    # Parse timestamps — handle mixed tz-aware / tz-naive strings
    # pandas can't parse them together with utc=True, so we normalize each
    raw = df[time_col].tolist()
    parsed_list = []
    for val in raw:
        try:
            ts = pd.Timestamp(val)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            parsed_list.append(ts)
        except Exception:
            parsed_list.append(pd.NaT)
    parsed = pd.Series(parsed_list, index=df.index)

    # Build a tz-aware trade_date for comparison
    trade_ts = pd.Timestamp(trade_date)
    if trade_ts.tzinfo is None:
        trade_ts = trade_ts.tz_localize("UTC")
    else:
        trade_ts = trade_ts.tz_convert("UTC")

    # If trade_date is midnight (date-only), extend to end of day
    # so same-day news is included
    if trade_ts.hour == 0 and trade_ts.minute == 0 and trade_ts.second == 0:
        trade_ts = trade_ts.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Filter: valid timestamp AND on or before trade_date
    mask = parsed.notna() & (parsed <= trade_ts)
    filtered = df.loc[mask].copy()
    filtered["_parsed_time"] = parsed.loc[mask]

    # Sort most recent first, then take top N
    filtered = filtered.sort_values("_parsed_time", ascending=False).head(max_items)
    filtered = filtered.drop(columns=["_parsed_time"])

    return filtered


def _load_raw_evidence(
    ticker: str,
    trade_date: datetime,
    data_root: Path,
) -> tuple[str, str]:
    """Load raw OHLCV summary and news text from disk."""
    ticker_dir = data_root / ticker

    # OHLCV summary: last 20 trading days
    ohlcv_summary = "No OHLCV data available."
    ohlcv_path = ticker_dir / "ohlcv_2021_2025.parquet"
    if not ohlcv_path.exists():
        ohlcv_path = ticker_dir / "ohlcv_2021_now.parquet"
    if ohlcv_path.exists():
        try:
            df = pd.read_parquet(ohlcv_path)
            df["date"] = pd.to_datetime(df["date"])
            recent = df[df["date"] <= pd.Timestamp(trade_date)].tail(20)
            if not recent.empty:
                lines = []
                for _, row in recent.iterrows():
                    lines.append(
                        f"  {row['date'].date()}: "
                        f"O={row.get('open', 'N/A'):.2f} "
                        f"H={row.get('high', 'N/A'):.2f} "
                        f"L={row.get('low', 'N/A'):.2f} "
                        f"C={row.get('close', 'N/A'):.2f} "
                        f"V={row.get('volume', 'N/A')}"
                    )
                ohlcv_summary = "\n".join(lines)
        except Exception:
            pass

    # News headlines — point-in-time filtered
    news_text = "No news available."
    news_path = ticker_dir / "news_latest.parquet"
    if not news_path.exists():
        news_path = ticker_dir / "news_combined_latest.parquet"
    if news_path.exists():
        try:
            df = pd.read_parquet(news_path)
            if not df.empty:
                df = _filter_news_by_trade_date(df, trade_date)
                if not df.empty:
                    headlines = []
                    for _, row in df.iterrows():
                        title = row.get("title") or ""
                        source = row.get("source") or ""
                        pub = row.get("published_time") or ""
                        if title:
                            headlines.append(f"  - [{source}] {title} ({pub})")
                    if headlines:
                        news_text = "\n".join(headlines)
        except Exception:
            pass

    return ohlcv_summary, news_text


def baseline_evidence_no_kg(
    ticker: str,
    trade_date: datetime,
    data_root: Path | None = None,
) -> Dict[str, Any]:
    """Baseline 2: LLM with raw evidence text (no KG structure)."""
    if data_root is None:
        data_root = Path("data/raw/market_news")

    ohlcv_summary, news_text = _load_raw_evidence(ticker, trade_date, data_root)

    client = OpenAICompatibleClient()
    try:
        payload = client.generate_json(
            _EVIDENCE_SYSTEM_PROMPT,
            _EVIDENCE_USER_PROMPT.format(
                ticker=ticker,
                trade_date=trade_date.date().isoformat(),
                ohlcv_summary=ohlcv_summary,
                news_text=news_text,
            ),
        )
    except Exception:
        payload = {}

    stance = _normalize_stance(payload.get("stance"))
    confidence = _safe_float(payload.get("confidence"), 0.3, 0.0, 1.0)
    score = _safe_float(payload.get("score"), 0.0, -1.0, 1.0)
    summary = str(payload.get("summary") or "Evidence-only baseline report unavailable.")
    factors = list(payload.get("factors", []))

    # Map stance to action
    if stance == "bullish" and score >= 0.25:
        action = "buy"
    elif stance == "bearish" and score <= -0.25:
        action = "sell"
    elif stance in {"neutral", "uncertain"} or abs(score) < 0.25:
        action = "hold"
    else:
        action = "abstain"

    # Build a synthetic agent report for compatibility
    agent_report = AgentReport(
        role="evidence_llm",
        stance=stance,
        confidence=confidence,
        score=score,
        summary=summary,
        evidence_refs=[],
        claim_refs=[],
        factors=factors,
    )

    reason = f"Evidence-only LLM baseline: {summary}"

    return _build_baseline_result(
        ticker=ticker,
        trade_date=trade_date,
        action=action,
        final_score=score,
        confidence=confidence,
        conflict_level=0.0,
        reason=reason,
        evidence_refs=[],
        agent_reports=[asdict(agent_report)],
        baseline="evidence_no_kg",
    )


# ---------------------------------------------------------------------------
# Baseline 3: static_kg
# Uses the full multi-agent KG pipeline, but queries only data from before
# the experiment start date (frozen KG snapshot).
# ---------------------------------------------------------------------------

def baseline_static_kg(
    ticker: str,
    trade_date: datetime,
    kg_context=None,
    static_cutoff: datetime | None = None,
) -> Dict[str, Any]:
    """Baseline 3: Full KG pipeline but with frozen (static) graph.

    Args:
        kg_context: KGAgentContext instance for querying the graph.
        static_cutoff: Only use KG data before this date.
                       Defaults to 2024-12-31 (before experiment period).
    """
    if kg_context is None:
        # Fallback if no KG context provided
        return _build_baseline_result(
            ticker=ticker,
            trade_date=trade_date,
            action="abstain",
            final_score=0.0,
            confidence=0.1,
            conflict_level=0.0,
            reason="Static KG baseline: no KG context available.",
            evidence_refs=[],
            agent_reports=[],
            baseline="static_kg",
        )

    cutoff = static_cutoff or datetime(2024, 12, 31)

    # Load subgraph with the static cutoff date instead of trade_date
    subgraph = kg_context.load_subgraph(ticker, cutoff)

    # Run the same multi-agent pipeline as the dynamic system
    from src.agents.fundamental_agent import FundamentalAgent
    from src.agents.risk_agent import RiskAgent
    from src.agents.roles import news_agent, technical_agent

    fundamental_agent_runner = FundamentalAgent()
    risk_agent_runner = RiskAgent()

    reports: list[AgentReport] = []
    reports.append(news_agent(subgraph, ticker, trade_date))
    reports.append(technical_agent(subgraph, ticker, trade_date))
    reports.append(
        fundamental_agent_runner.run(
            {"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}
        )
    )
    reports.append(
        risk_agent_runner.run(
            {"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}
        )
    )

    # Use the same portfolio manager decision logic
    decision = portfolio_manager_decide(ticker, trade_date, reports, subgraph=subgraph)

    # Tag it as static_kg baseline
    decision["baseline"] = "static_kg"
    return decision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_baseline_result(
    ticker: str,
    trade_date: datetime,
    action: str,
    final_score: float,
    confidence: float,
    conflict_level: float,
    reason: str,
    evidence_refs: list[str],
    agent_reports: list[dict],
    baseline: str,
    retrieved_evidence_ids: list[str] | None = None,
    retrieved_claim_ids: list[str] | None = None,
) -> Dict[str, Any]:
    """Build a result dict compatible with the kg_dynamic format."""
    # Determine supporting/opposing roles from agent reports
    supporting_roles: list[str] = []
    opposing_roles: list[str] = []
    for report in agent_reports:
        role = report.get("role", "")
        stance = report.get("stance", "")
        if action == "buy" and stance == "bullish":
            supporting_roles.append(role)
        elif action == "sell" and stance == "bearish":
            supporting_roles.append(role)
        elif action in {"hold", "abstain"} and stance in {"neutral", "uncertain"}:
            supporting_roles.append(role)
        elif stance in {"bullish", "bearish"}:
            opposing_roles.append(role)

    evidence_alignment = "aligned"
    if opposing_roles and not supporting_roles:
        evidence_alignment = "contradicted"
    elif opposing_roles:
        evidence_alignment = "mixed"

    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": action,
        "final_score": round(final_score, 4),
        "confidence": round(confidence, 4),
        "conflict_level": round(conflict_level, 4),
        "decision_reason": reason,
        "evidence_refs": sorted(set(x for x in evidence_refs if x)),
        "claim_refs": [],
        "retrieved_evidence_ids": retrieved_evidence_ids if retrieved_evidence_ids is not None else [],
        "retrieved_claim_ids": retrieved_claim_ids if retrieved_claim_ids is not None else [],
        "supporting_roles": supporting_roles,
        "opposing_roles": opposing_roles,
        "stale_evidence_count": 0,
        "fresh_evidence_count": len(evidence_refs),
        "evidence_alignment": evidence_alignment,
        "trace": {},
        "agent_reports": agent_reports,
        "baseline": baseline,
    }
