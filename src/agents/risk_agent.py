# RiskAgent: assess the risk level of a stock based on signals from KG. 
# use explicit risk events if available, or fall back to LLM-based analysis if not.

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.roles import AgentReport, risk_agent as run_risk_agent


class RiskAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("risk_agent")

    def run(self, input_data: dict) -> AgentReport:
        ticker = input_data["ticker"]
        trade_date = input_data["trade_date"]
        subgraph = input_data["subgraph"]

        risks = subgraph.get("risks", [])
        if risks:
            return self._run_from_kg_risks(risks)

        # Fallback: keep previous LLM behavior if no explicit risk KG exists.
        return run_risk_agent(subgraph, ticker, trade_date)

    def _run_from_kg_risks(self, risks: list[dict[str, Any]]) -> AgentReport:
        bearish_risks = [
            r for r in risks
            if str(r.get("direction")) == "bearish"
        ]
        neutral_risks = [
            r for r in risks
            if str(r.get("direction")) in {"neutral", "uncertain", ""}
        ]

        total_severity = sum(_safe_float(r.get("severity") or r.get("strength"), 0.0) for r in bearish_risks)
        avg_severity = total_severity / max(1, len(bearish_risks))

        if len(bearish_risks) >= 2 or avg_severity >= 0.65:
            stance = "bearish"
            score = -min(1.0, max(0.35, avg_severity))
            confidence = min(0.9, 0.45 + 0.1 * len(bearish_risks))
        elif len(bearish_risks) == 1:
            stance = "neutral"
            score = -min(0.45, max(0.15, avg_severity))
            confidence = 0.55
        else:
            stance = "neutral"
            score = 0.0
            confidence = 0.45

        evidence_refs = []
        factors = []

        for r in risks[:20]:
            evidence_id = r.get("evidence_id")
            entity_id = r.get("entity_id")
            if evidence_id:
                evidence_refs.append(str(evidence_id))
            elif entity_id:
                evidence_refs.append(str(entity_id))

            factors.append(
                {
                    "name": str(r.get("name") or r.get("risk_type") or "risk_signal"),
                    "direction": str(r.get("direction") or "neutral"),
                    "weight": round(_safe_float(r.get("severity") or r.get("strength"), 0.3), 3),
                }
            )

        summary = (
            f"RiskAgent found {len(bearish_risks)} bearish risk signals "
            f"and {len(neutral_risks)} neutral/uncertain risk signals."
        )

        return AgentReport(
            role="risk",
            stance=stance,
            confidence=confidence,
            score=score,
            summary=summary,
            evidence_refs=sorted(set(evidence_refs)),
            factors=factors,
        )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default