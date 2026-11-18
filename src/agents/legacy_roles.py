"""Original role execution; kept separate from deterministic aggregation."""
from datetime import datetime
from typing import Any, Dict, List
from src.agents.models import AgentReport, Stance
from src.llm.client import OpenAICompatibleClient
from src.llm.prompts import AGENT_SYSTEM_PROMPT, build_agent_prompt

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


def _llm_report(role: str, ticker: str, trade_date: datetime, subgraph: Dict[str, List[Dict[str, Any]]], *, client=None) -> AgentReport:
    client = client or OpenAICompatibleClient()
    payload = client.generate_json(
        AGENT_SYSTEM_PROMPT,
        build_agent_prompt(role, ticker, trade_date.date().isoformat(), subgraph),
    )
    # Extract claim_refs from subgraph claims if available
    claim_refs: List[str] = []
    for claim in subgraph.get("claims", []) or []:
        cid = claim.get("claim_id") or claim.get("entity_id")
        if cid:
            claim_refs.append(str(cid))

    return AgentReport(
        role=role,
        stance=_normalize_stance(payload.get("stance")),
        confidence=_safe_float(payload.get("confidence"), 0.3, 0.0, 1.0),
        score=_safe_float(payload.get("score"), 0.0, -1.0, 1.0),
        summary=str(payload.get("summary") or f"{role} report unavailable."),
        evidence_refs=[str(x) for x in payload.get("evidence_refs", [])],
        claim_refs=sorted(set(claim_refs)),
        factors=list(payload.get("factors", [])),
    )


def news_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime, *, client=None) -> AgentReport:
    return _llm_report("news", ticker, trade_date, subgraph, client=client)


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
            claim_refs=[],
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

    # Collect evidence_refs and claim_refs from signals and claims
    evidence_refs: List[str] = []
    for s in signals[:20]:
        eid = s.get("evidence_id") or s.get("entity_id") or s.get("name") or ""
        if eid:
            evidence_refs.append(str(eid))

    claim_refs: List[str] = []
    for s in signals:
        cid = s.get("claim_id")
        if cid:
            claim_refs.append(str(cid))
    for claim in subgraph.get("claims", []) or []:
        cid = claim.get("claim_id") or claim.get("entity_id")
        if cid:
            claim_refs.append(str(cid))

    return AgentReport(
        role="technical",
        stance=stance,
        confidence=min(0.9, 0.3 + total * 0.05),
        score=score,
        summary=f"{bullish} bullish vs {bearish} bearish technical signals.",
        evidence_refs=sorted(set(evidence_refs)),
        claim_refs=sorted(set(claim_refs)),
        factors=[{"name": "technical_signal_balance", "direction": stance, "weight": round(abs(score), 3)}],
    )


def fundamental_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime, *, client=None) -> AgentReport:
    return _llm_report("fundamental", ticker, trade_date, subgraph, client=client)


def risk_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime, *, client=None) -> AgentReport:
    return _llm_report("risk", ticker, trade_date, subgraph, client=client)


