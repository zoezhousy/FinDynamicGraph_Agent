from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal

from src.llm.client import OpenAICompatibleClient
from src.llm.prompts import AGENT_SYSTEM_PROMPT, build_agent_prompt


DecisionAction = Literal["buy", "sell", "hold", "abstain"]
Stance = Literal["bullish", "bearish", "neutral", "uncertain"]


@dataclass
class AgentFactor:
    name: str
    direction: str
    weight: float


@dataclass
class AgentReport:
    role: str
    stance: Stance
    confidence: float
    score: float
    summary: str
    evidence_refs: List[str]
    factors: List[Dict[str, Any]]


def _normalize_stance(value: str | None) -> Stance:
    if value in {"bullish", "bearish", "neutral", "uncertain"}:
        return value
    return "uncertain"


def _safe_float(value: Any, default: float = 0.0, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _llm_report(role: str, ticker: str, trade_date: datetime, subgraph: Dict[str, List[Dict[str, Any]]]) -> AgentReport:
    client = OpenAICompatibleClient()
    payload = client.generate_json(
        AGENT_SYSTEM_PROMPT,
        build_agent_prompt(role, ticker, trade_date.date().isoformat(), subgraph),
    )
    return AgentReport(
        role=role,
        stance=_normalize_stance(payload.get("stance")),
        confidence=_safe_float(payload.get("confidence"), 0.3, 0.0, 1.0),
        score=_safe_float(payload.get("score"), 0.0, -1.0, 1.0),
        summary=str(payload.get("summary") or f"{role} report unavailable."),
        evidence_refs=[str(x) for x in payload.get("evidence_refs", [])],
        factors=list(payload.get("factors", [])),
    )


def news_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime) -> AgentReport:
    return _llm_report("news", ticker, trade_date, subgraph)


def technical_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime) -> AgentReport:
    signals = subgraph.get("signals", [])
    if not signals:
        return AgentReport(
            role="technical",
            stance="neutral",
            confidence=0.2,
            score=0.0,
            summary="No technical signals available.",
            evidence_refs=[],
            factors=[],
        )
    bullish = sum(1 for s in signals if s.get("direction") == "bullish")
    bearish = sum(1 for s in signals if s.get("direction") == "bearish")
    total = max(1, bullish + bearish)
    score = (bullish - bearish) / total
    stance: Stance = "neutral"
    if score > 0.2:
        stance = "bullish"
    elif score < -0.2:
        stance = "bearish"
    return AgentReport(
        role="technical",
        stance=stance,
        confidence=min(0.9, 0.3 + total * 0.05),
        score=score,
        summary=f"{bullish} bullish vs {bearish} bearish technical signals.",
        evidence_refs=[str(s.get("entity_id") or s.get("name") or "") for s in signals[:20]],
        factors=[{"name": "technical_signal_balance", "direction": stance, "weight": round(abs(score), 3)}],
    )


def fundamental_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime) -> AgentReport:
    return _llm_report("fundamental", ticker, trade_date, subgraph)


def risk_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime) -> AgentReport:
    return _llm_report("risk", ticker, trade_date, subgraph)


def portfolio_manager_decide(
    ticker: str,
    trade_date: datetime,
    reports: List[AgentReport],
) -> Dict[str, Any]:
    weights = {
        "technical": 0.35,
        "news": 0.20,
        "fundamental": 0.25,
        "risk": 0.20,
    }
    weighted_scores: Dict[str, float] = {}
    evidence_refs: List[str] = []
    role_to_stance: Dict[str, str] = {}

    for report in reports:
        role_to_stance[report.role] = report.stance
        weighted_scores[report.role] = report.score * report.confidence * weights.get(report.role, 0.0)
        evidence_refs.extend(report.evidence_refs)

    technical_score = weighted_scores.get("technical", 0.0)
    news_score = weighted_scores.get("news", 0.0)
    fundamental_score = weighted_scores.get("fundamental", 0.0)
    risk_score = weighted_scores.get("risk", 0.0)
    final_score = technical_score + news_score + fundamental_score + risk_score

    conflict_level = 0.0
    if role_to_stance.get("technical") == "bullish" and role_to_stance.get("risk") == "bearish":
        conflict_level += 0.4
    if role_to_stance.get("fundamental") == "bullish" and role_to_stance.get("news") == "bearish":
        conflict_level += 0.3
    if len(set(s for s in role_to_stance.values() if s in {"bullish", "bearish"})) > 1:
        conflict_level += 0.2

    evidence_count = len([x for x in evidence_refs if x])
    confidence = max(0.05, min(0.95, sum(r.confidence for r in reports) / max(1, len(reports)) - conflict_level * 0.25))

    if evidence_count < 2 or confidence < 0.2:
        action: DecisionAction = "abstain"
        reason = "Evidence is too weak or confidence is too low."
    elif conflict_level >= 0.5 and abs(final_score) < 0.35:
        action = "abstain"
        reason = "Conflicting signals are too strong for a reliable action."
    elif final_score >= 0.25:
        action = "buy"
        reason = "Weighted agent aggregation is net bullish."
    elif final_score <= -0.25:
        action = "sell"
        reason = "Weighted agent aggregation is net bearish."
    else:
        action = "hold"
        reason = "Signals are mixed or not strong enough for directional action."

    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": action,
        "final_score": round(final_score, 4),
        "confidence": round(confidence, 4),
        "conflict_level": round(conflict_level, 4),
        "decision_reason": reason,
        "evidence_refs": sorted(set(x for x in evidence_refs if x)),
        "trace": {
            "technical_score": round(technical_score, 4),
            "news_score": round(news_score, 4),
            "fundamental_score": round(fundamental_score, 4),
            "risk_score": round(risk_score, 4),
        },
        "agent_reports": [asdict(r) for r in reports],
    }
