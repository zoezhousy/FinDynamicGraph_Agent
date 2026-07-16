from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal

from src.llm.client import OpenAICompatibleClient
from src.llm.prompts import AGENT_SYSTEM_PROMPT, build_agent_prompt
from src.kg.schema import AgentAssessment, DecisionTrace


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
    claim_refs: List[str]
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


def fundamental_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime) -> AgentReport:
    return _llm_report("fundamental", ticker, trade_date, subgraph)


def risk_agent(subgraph: Dict[str, List[Dict[str, Any]]], ticker: str, trade_date: datetime) -> AgentReport:
    return _llm_report("risk", ticker, trade_date, subgraph)


def _is_stale_ref(evidence_ref: str, subgraph: Dict[str, List[Dict[str, Any]]], trade_date: datetime) -> bool:
    evidence_map = {
        str(item.get("evidence_id")): item
        for item in subgraph.get("evidences", [])
        if item.get("evidence_id")
    }
    signal_map = {
        str(item.get("entity_id")): item
        for item in subgraph.get("signals", [])
        if item.get("entity_id")
    }

    if evidence_ref in evidence_map:
        published_at = evidence_map[evidence_ref].get("published_at")
        if not published_at:
            return False
        try:
            return datetime.fromisoformat(str(published_at).replace("Z", "+00:00")) < trade_date
        except Exception:
            return False

    if evidence_ref in signal_map:
        valid_to = signal_map[evidence_ref].get("valid_to")
        if not valid_to:
            return False
        try:
            return datetime.fromisoformat(str(valid_to).replace("Z", "+00:00")) < trade_date
        except Exception:
            return False

    return False


def _aligns(action: DecisionAction, stance: Stance) -> bool:
    if action == "buy":
        return stance == "bullish"
    if action == "sell":
        return stance == "bearish"
    if action == "hold":
        return stance in {"neutral", "uncertain"}
    if action == "abstain":
        return stance in {"uncertain", "neutral"}
    return False

# collect fallback evidence refs from subgraph for risk agent 
# when explicit risk events are not available.
def _collect_fallback_evidence_refs(
    subgraph: Dict[str, List[Dict[str, Any]]] | None,
    limit: int = 20,
) -> List[str]:
    if subgraph is None:
        return []

    refs: list[str] = []

    for key, id_key in (
        ("risks", "evidence_id"),
        ("fundamentals", "evidence_id"),
        ("news", "evidence_id"),
        ("evidences", "evidence_id"),
        ("signals", "entity_id"),
    ):
        for item in subgraph.get(key, []) or []:
            value = item.get(id_key)
            if value:
                refs.append(str(value))

            if len(refs) >= limit:
                return sorted(set(refs))

    return sorted(set(refs))


def extract_retrieved_ids(
    subgraph: Dict[str, List[Dict[str, Any]]] | None,
) -> tuple[list[str], list[str]]:
    """Extract all evidence and claim IDs actually returned by a subgraph query.

    Returns:
        (retrieved_evidence_ids, retrieved_claim_ids) — sorted, deduplicated lists
        of string IDs.  Returns empty lists if subgraph is None.
    """
    if subgraph is None:
        return [], []

    evidence_ids: set[str] = set()
    claim_ids: set[str] = set()

    # Evidence nodes from the "evidences" key
    for item in subgraph.get("evidences", []) or []:
        eid = item.get("evidence_id")
        if eid:
            evidence_ids.add(str(eid).strip())

    # Evidence IDs from entity-level keys that carry evidence_id
    for key in ("news", "risks", "fundamentals"):
        for item in subgraph.get(key, []) or []:
            eid = item.get("evidence_id")
            if eid:
                evidence_ids.add(str(eid).strip())

    # Signal entities carry entity_id, not evidence_id
    for item in subgraph.get("signals", []) or []:
        eid = item.get("entity_id")
        if eid:
            evidence_ids.add(str(eid).strip())

    # Source documents
    for item in subgraph.get("sources", []) or []:
        sid = item.get("source_id")
        if sid:
            evidence_ids.add(str(sid).strip())

    # Claims
    for item in subgraph.get("claims", []) or []:
        cid = item.get("entity_id") or item.get("claim_id")
        if cid:
            claim_ids.add(str(cid).strip())

    return sorted(evidence_ids), sorted(claim_ids)

def portfolio_manager_decide(
    ticker: str,
    trade_date: datetime,
    reports: List[AgentReport],
    subgraph: Dict[str, List[Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    weights = {
        "technical": 0.35,
        "news": 0.20,
        "fundamental": 0.25,
        "risk": 0.20,
    }
    weighted_scores: Dict[str, float] = {}
    evidence_refs: List[str] = []
    claim_refs: List[str] = []
    role_to_stance: Dict[str, str] = {}

    for report in reports:
        role_to_stance[report.role] = report.stance
        weighted_scores[report.role] = report.score * report.confidence * weights.get(report.role, 0.0)
        evidence_refs.extend(report.evidence_refs)
        claim_refs.extend(report.claim_refs)

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

    # de-duplicate and filter out empty refs
    evidence_refs = sorted(set(x for x in evidence_refs if x))
    claim_refs = sorted(set(x for x in claim_refs if x))
    
    if not evidence_refs:
        evidence_refs = _collect_fallback_evidence_refs(subgraph)

    evidence_count = len(evidence_refs)

    # Compute base confidence as mean of agent confidences
    base_confidence = sum(r.confidence for r in reports) / max(1, len(reports))
    # Reduce penalty for conflict: conflict signals are informative, not disqualifying
    confidence = max(0.05, min(0.95, base_confidence - conflict_level * 0.15))

    # Decision logic with calibrated thresholds
    # Abstain only when there is truly no evidence at all
    if evidence_count == 0 and base_confidence < 0.15:
        action: DecisionAction = "abstain"
        reason = "No evidence available and agent confidence is very low."
    # High conflict + weak signal → hold (not abstain); conflict is informative
    elif conflict_level >= 0.7 and abs(final_score) < 0.15:
        action = "hold"
        reason = "Strong conflicting signals detected; holding position."
    # Directional thresholds: calibrated to produce meaningful buy/sell decisions
    elif final_score >= 0.12:
        action = "buy"
        reason = "Weighted agent aggregation is net bullish."
    elif final_score <= -0.12:
        action = "sell"
        reason = "Weighted agent aggregation is net bearish."
    else:
        action = "hold"
        reason = "Signals are mixed or not strong enough for directional action."

    bullish_support_count = sum(1 for r in reports if r.stance == "bullish")
    bearish_support_count = sum(1 for r in reports if r.stance == "bearish")
    neutral_support_count = sum(1 for r in reports if r.stance in {"neutral", "uncertain"})
    supporting_roles = [r.role for r in reports if _aligns(action, r.stance)]
    opposing_roles = [r.role for r in reports if r.role not in supporting_roles and r.stance in {"bullish", "bearish"}]

    stale_evidence_count = 0
    fresh_evidence_count = len(evidence_refs)
    if subgraph is not None:
        stale_evidence_count = sum(1 for ref in evidence_refs if _is_stale_ref(ref, subgraph, trade_date))
        fresh_evidence_count = max(0, len(evidence_refs) - stale_evidence_count)

    # Retrieve the full set of evidence/claim IDs from the subgraph query
    retrieved_evidence_ids, retrieved_claim_ids = extract_retrieved_ids(subgraph)

    evidence_alignment = "aligned"
    if action in {"buy", "sell"} and not supporting_roles:
        evidence_alignment = "unsupported"
    elif opposing_roles and not supporting_roles:
        evidence_alignment = "contradicted"
    elif stale_evidence_count > fresh_evidence_count:
        evidence_alignment = "stale"
    elif opposing_roles:
        evidence_alignment = "mixed"

    trace_payload = {
        "technical_score": round(technical_score, 4),
        "news_score": round(news_score, 4),
        "fundamental_score": round(fundamental_score, 4),
        "risk_score": round(risk_score, 4),
    }

    decision_id = f"decision:{ticker}:{trade_date.date().isoformat()}"
    assessments = [
        AgentAssessment(
            assessment_id=f"assessment:{report.role}:{ticker}:{trade_date.date().isoformat()}",
            ticker=ticker,
            trade_date=trade_date,
            agent_role=report.role,
            stance=report.stance,
            confidence=round(report.confidence, 4),
            score=round(report.score, 4),
            summary=report.summary,
            evidence_refs=report.evidence_refs,
            claim_refs=report.claim_refs,
            factors=report.factors,
            supports_decision=report.role in supporting_roles,
            opposes_decision=report.role in opposing_roles,
        )
        for report in reports
    ]
    decision_trace = DecisionTrace(
        decision_id=decision_id,
        ticker=ticker,
        trade_date=trade_date,
        action=action,
        final_score=round(final_score, 4),
        confidence=round(confidence, 4),
        conflict_level=round(conflict_level, 4),
        decision_reason=reason,
        bullish_support_count=bullish_support_count,
        bearish_support_count=bearish_support_count,
        neutral_support_count=neutral_support_count,
        evidence_ids=evidence_refs,
        claim_ids=claim_refs,
        supporting_roles=supporting_roles,
        opposing_roles=opposing_roles,
        stale_evidence_count=stale_evidence_count,
        fresh_evidence_count=fresh_evidence_count,
        evidence_alignment=evidence_alignment,
        trace=trace_payload,
    )

    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": action,
        "final_score": round(final_score, 4),
        "confidence": round(confidence, 4),
        "conflict_level": round(conflict_level, 4),
        "decision_reason": reason,
        "evidence_refs": evidence_refs,
        "claim_refs": claim_refs,
        "retrieved_evidence_ids": retrieved_evidence_ids,
        "retrieved_claim_ids": retrieved_claim_ids,
        "supporting_roles": supporting_roles,
        "opposing_roles": opposing_roles,
        "stale_evidence_count": stale_evidence_count,
        "fresh_evidence_count": fresh_evidence_count,
        "evidence_alignment": evidence_alignment,
        "trace": trace_payload,
        "agent_reports": [asdict(r) for r in reports],
        "decision_trace": decision_trace.model_dump(mode="json"),
        "agent_assessments": [assessment.model_dump(mode="json") for assessment in assessments],
    }
