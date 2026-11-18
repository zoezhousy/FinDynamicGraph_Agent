"""Canonical deterministic portfolio manager. No LLM or database calls."""
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List
from src.agents.models import AgentReport, DecisionAction
from src.agents.evidence import _is_stale_ref, _aligns, _collect_fallback_evidence_refs, extract_retrieved_ids, evidence_freshness
from src.kg.schema import AgentAssessment, DecisionTrace

ROLE_WEIGHTS = {"technical": 0.35, "news": 0.20, "fundamental": 0.25, "risk": 0.20}


def strong_disagreement(conflict_level, score):
    return conflict_level >= 0.7 and abs(score) < 0.15


def portfolio_manager_decide(
    ticker: str,
    trade_date: datetime,
    reports: List[AgentReport],
    subgraph: Dict[str, List[Dict[str, Any]]] | None = None,
    *, evaluation_version: str = "legacy",
) -> Dict[str, Any]:
    if evaluation_version not in {"legacy", "v2"}:
        raise ValueError("evaluation_version must be legacy or v2")
    weighted_scores: Dict[str, float] = {}
    evidence_refs: List[str] = []
    claim_refs: List[str] = []
    role_to_stance: Dict[str, str] = {}

    for report in reports:
        role_to_stance[report.role] = report.stance
        weighted_scores[report.role] = report.score * report.confidence * ROLE_WEIGHTS.get(report.role, 0.0)
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
    elif strong_disagreement(conflict_level, final_score):
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
    unknown_evidence_count = 0
    if evaluation_version == "v2":
        classifications = [evidence_freshness(ref, subgraph or {}, trade_date) for ref in evidence_refs]
        stale_evidence_count = classifications.count("stale")
        fresh_evidence_count = classifications.count("fresh")
        unknown_evidence_count = classifications.count("unknown")
    elif subgraph is not None:
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

    if evaluation_version == "v2":
        trace_payload.update(evaluation_version="v2", unknown_evidence_count=unknown_evidence_count, staleness_days=30)

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

    result = {
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

    if evaluation_version == "v2":
        result.update(evaluation_version="v2", unknown_evidence_count=unknown_evidence_count)
    return result
