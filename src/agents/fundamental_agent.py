from __future__ import annotations

from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.agents.roles import AgentReport, fundamental_agent as run_fundamental_agent_llm

# Metrics that are "higher is better" for bullish interpretation.
_BULLISH_HIGH_METRICS = {
    "profitmargins", "operatingmargins", "grossmargins",
    "returnonequity", "returnonassets",
    "revenuegrowth", "earningsgrowth",
    "currentratio", "quickratio",
    "freecashflow", "operatingcashflow",
    "totalrevenue", "grossprofits", "ebitda",
    "bookvalue",
}

# Metrics that are "lower is better" for bullish interpretation.
_BULLISH_LOW_METRICS = {
    "debttoequity", "totaldebt",
}

# Metrics that need special handling (e.g. PE ratios).
_SPECIAL_METRICS = {
    "trailingpe", "forwardpe", "pricetobook",
    "enterprisetorevenue", "enterprisetoebitda",
    "dividendyield", "beta",
}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _direction_to_score(direction: str, strength: float) -> float:
    """Map direction + strength to a signed score in [-1, 1]."""
    if direction == "bullish":
        return min(1.0, strength)
    elif direction == "bearish":
        return max(-1.0, -strength)
    return 0.0


class FundamentalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("fundamental_agent")

    def run(self, input_data: dict) -> AgentReport:
        ticker = input_data["ticker"]
        trade_date = input_data["trade_date"]
        subgraph = input_data["subgraph"]

        fundamentals = subgraph.get("fundamentals", [])
        if fundamentals:
            return self._run_from_kg_fundamentals(fundamentals, ticker)

        # Fallback to LLM if no structured fundamentals in KG.
        return run_fundamental_agent_llm(subgraph, ticker, trade_date)

    # ------------------------------------------------------------------
    # Rule-based analysis on structured KG fundamental signals
    # ------------------------------------------------------------------

    def _run_from_kg_fundamentals(
        self,
        fundamentals: List[Dict[str, Any]],
        ticker: str,
    ) -> AgentReport:
        """Rule-based fundamental analysis using structured KG FundamentalSignal nodes."""

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_strength = 0.0
        scored_directions: List[float] = []

        evidence_refs: List[str] = []
        claim_refs: List[str] = []
        factors: List[Dict[str, Any]] = []

        for f in fundamentals:
            direction = str(f.get("direction") or "neutral").lower()
            strength = _safe_float(f.get("strength"), 0.5)
            metric = str(f.get("metric") or f.get("name") or "unknown")
            numeric_value = _safe_float(f.get("numeric_value"))
            evidence_id = f.get("evidence_id")
            entity_id = f.get("entity_id")
            claim_id = f.get("claim_id")

            # Collect evidence refs
            if evidence_id:
                evidence_refs.append(str(evidence_id))
            elif entity_id:
                evidence_refs.append(str(entity_id))
            if claim_id:
                claim_refs.append(str(claim_id))

            # Count directions
            if direction == "bullish":
                bullish_count += 1
            elif direction == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1

            total_strength += strength
            scored_directions.append(_direction_to_score(direction, strength))

            # Build factor entry
            weight = round(abs(strength), 3)
            value_str = f"{numeric_value:.4f}" if numeric_value is not None else str(f.get("value", "N/A"))
            factors.append({
                "name": f"fundamental_{metric}",
                "direction": direction,
                "weight": weight,
                "value": value_str,
            })

        total = max(1, len(fundamentals))
        avg_score = sum(scored_directions) / total
        avg_strength = total_strength / total

        # Determine stance
        if avg_score > 0.15:
            stance = "bullish"
        elif avg_score < -0.15:
            stance = "bearish"
        else:
            stance = "neutral"

        # Confidence: higher when more signals agree and are strong
        agreement = max(bullish_count, bearish_count, neutral_count) / total
        confidence = min(0.9, 0.35 + 0.3 * agreement + 0.2 * avg_strength)

        # Summary
        summary_parts = [
            f"FundamentalAgent analyzed {total} structured fundamental metrics for {ticker}.",
            f"Distribution: {bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral.",
        ]
        if bullish_count > bearish_count:
            summary_parts.append("Overall fundamental picture leans bullish.")
        elif bearish_count > bullish_count:
            summary_parts.append("Overall fundamental picture leans bearish.")
        else:
            summary_parts.append("Fundamental signals are mixed or neutral.")

        # Mention key metrics
        top_factors = sorted(factors, key=lambda x: x["weight"], reverse=True)[:5]
        if top_factors:
            detail = ", ".join(
                f'{f["name"]}={f.get("value", "N/A")} ({f["direction"]})'
                for f in top_factors
            )
            summary_parts.append(f"Key metrics: {detail}.")

        return AgentReport(
            role="fundamental",
            stance=stance,
            confidence=round(confidence, 4),
            score=round(avg_score, 4),
            summary=" ".join(summary_parts),
            evidence_refs=sorted(set(evidence_refs)),
            claim_refs=sorted(set(claim_refs)),
            factors=factors,
        )
