# RiskAgent: assess the risk level of a stock based on signals from KG.
# Uses explicit RiskEvent nodes, high-volatility signals, missing/stale evidence,
# and bullish-tech vs bearish-risk/news conflict detection.

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent
from src.agents.roles import AgentReport, risk_agent as run_risk_agent_llm


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


class RiskAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("risk_agent")

    def run(self, input_data: dict) -> AgentReport:
        ticker = input_data["ticker"]
        trade_date = input_data["trade_date"]
        subgraph = input_data["subgraph"]

        risks = subgraph.get("risks", [])
        signals = subgraph.get("signals", [])
        news = subgraph.get("news", [])
        evidences = subgraph.get("evidences", [])

        # Always try to build a comprehensive risk report from KG data
        # even when explicit RiskEvent nodes are sparse.
        has_kg_data = risks or signals or news or evidences
        if has_kg_data:
            return self._run_comprehensive_risk(
                ticker=ticker,
                trade_date=trade_date,
                risks=risks,
                signals=signals,
                news=news,
                evidences=evidences,
            )

        # Fallback: keep previous LLM behavior if no KG data exists at all.
        return run_risk_agent_llm(subgraph, ticker, trade_date)

    # ------------------------------------------------------------------
    # Comprehensive rule-based risk analysis
    # ------------------------------------------------------------------

    def _run_comprehensive_risk(
        self,
        ticker: str,
        trade_date: datetime,
        risks: List[Dict[str, Any]],
        signals: List[Dict[str, Any]],
        news: List[Dict[str, Any]],
        evidences: List[Dict[str, Any]],
    ) -> AgentReport:
        evidence_refs: List[str] = []
        factors: List[Dict[str, Any]] = []
        risk_score = 0.0  # negative = bearish risk, positive = low risk / bullish

        # ── 1. Explicit RiskEvent nodes ──
        bearish_risks = [r for r in risks if str(r.get("direction")) == "bearish"]
        neutral_risks = [r for r in risks if str(r.get("direction")) in {"neutral", "uncertain", ""}]

        risk_severity_sum = sum(
            _safe_float(r.get("severity") or r.get("strength"), 0.0)
            for r in bearish_risks
        )
        avg_risk_severity = risk_severity_sum / max(1, len(bearish_risks))

        for r in risks[:20]:
            eid = r.get("evidence_id") or r.get("entity_id")
            if eid:
                evidence_refs.append(str(eid))
            factors.append({
                "name": str(r.get("name") or r.get("risk_type") or "risk_event"),
                "direction": str(r.get("direction") or "neutral"),
                "weight": round(_safe_float(r.get("severity") or r.get("strength"), 0.3), 3),
            })

        if bearish_risks:
            risk_score -= min(0.5, avg_risk_severity * 0.5 + len(bearish_risks) * 0.08)

        # ── 2. High volatility signals ──
        vol_signals = [
            s for s in signals
            if "volatility" in str(s.get("name", "")).lower()
            or "atr" in str(s.get("name", "")).lower()
        ]
        high_vol = [s for s in vol_signals if s.get("direction") == "bearish"]

        for s in high_vol:
            eid = s.get("entity_id") or s.get("evidence_id")
            if eid:
                evidence_refs.append(str(eid))
            strength = _safe_float(s.get("strength"), 0.5)
            factors.append({
                "name": str(s.get("name") or "high_volatility"),
                "direction": "bearish",
                "weight": round(strength, 3),
            })
            risk_score -= min(0.25, strength * 0.3)

        # ── 3. Stale evidence detection ──
        stale_count, fresh_count, stale_refs = self._count_stale_evidence(
            evidences, signals, trade_date
        )
        evidence_refs.extend(stale_refs)

        if stale_count > 0:
            stale_ratio = stale_count / max(1, stale_count + fresh_count)
            risk_score -= min(0.2, stale_ratio * 0.25)
            factors.append({
                "name": "stale_evidence",
                "direction": "bearish",
                "weight": round(min(0.5, stale_ratio), 3),
            })

        # ── 4. Missing evidence penalty ──
        total_evidence = len(evidences)
        if total_evidence == 0 and (risks or signals or news):
            # We have signals but no evidence backing them
            risk_score -= 0.15
            factors.append({
                "name": "missing_evidence",
                "direction": "bearish",
                "weight": 0.3,
            })

        # ── 5. Bullish-tech vs bearish-risk/news conflict ──
        bullish_tech = [
            s for s in signals
            if s.get("direction") == "bullish"
            and "volatility" not in str(s.get("name", "")).lower()
        ]
        bearish_non_tech = bearish_risks + [
            n for n in news if str(n.get("direction")) == "bearish"
        ]

        if bullish_tech and bearish_non_tech:
            conflict_weight = min(0.4, len(bearish_non_tech) * 0.1 + len(bullish_tech) * 0.05)
            risk_score -= conflict_weight
            factors.append({
                "name": "bullish_tech_vs_bearish_risk_news_conflict",
                "direction": "bearish",
                "weight": round(conflict_weight, 3),
            })

        # ── 6. Bearish news signals ──
        bearish_news = [n for n in news if str(n.get("direction")) == "bearish"]
        for n in bearish_news[:10]:
            eid = n.get("evidence_id") or n.get("entity_id")
            if eid:
                evidence_refs.append(str(eid))
            factors.append({
                "name": str(n.get("title") or n.get("name") or "bearish_news"),
                "direction": "bearish",
                "weight": round(_safe_float(n.get("strength"), 0.4), 3),
            })
        if bearish_news:
            risk_score -= min(0.2, len(bearish_news) * 0.05)

        # ── Aggregate scoring ──
        # Clamp risk_score to [-1, 0] range (risk is about downside)
        risk_score = max(-1.0, min(0.0, risk_score))

        # Determine stance
        if risk_score <= -0.4:
            stance = "bearish"
        elif risk_score <= -0.15:
            stance = "uncertain"
        else:
            stance = "neutral"

        # Confidence
        signal_agreement = 1.0 if len(bearish_risks) + len(high_vol) + len(bearish_news) > 2 else 0.6
        confidence = min(0.9, 0.35 + 0.15 * signal_agreement + 0.1 * abs(risk_score) * 10)

        # Summary
        summary_parts = [
            f"RiskAgent assessed {ticker} risk profile using {len(risks)} risk events, "
            f"{len(signals)} technical signals, {len(news)} news items, "
            f"and {len(evidences)} evidence nodes.",
        ]
        if bearish_risks:
            summary_parts.append(f"Found {len(bearish_risks)} bearish risk events (avg severity {avg_risk_severity:.2f}).")
        if high_vol:
            summary_parts.append(f"Detected {len(high_vol)} high-volatility signals.")
        if stale_count:
            summary_parts.append(f"{stale_count}/{stale_count + fresh_count} evidence references are stale.")
        if bullish_tech and bearish_non_tech:
            summary_parts.append(
                f"Conflict detected: {len(bullish_tech)} bullish technical signals "
                f"vs {len(bearish_non_tech)} bearish risk/news signals."
            )
        if not bearish_risks and not high_vol and not bearish_news:
            summary_parts.append("No significant risk signals found; risk profile appears neutral.")

        return AgentReport(
            role="risk",
            stance=stance,
            confidence=round(confidence, 4),
            score=round(risk_score, 4),
            summary=" ".join(summary_parts),
            evidence_refs=sorted(set(evidence_refs)),
            factors=factors,
        )

    # ------------------------------------------------------------------
    # Stale evidence detection
    # ------------------------------------------------------------------

    def _count_stale_evidence(
        self,
        evidences: List[Dict[str, Any]],
        signals: List[Dict[str, Any]],
        trade_date: datetime,
        stale_threshold_days: int = 90,
    ) -> tuple[int, int, List[str]]:
        """Count stale vs fresh evidence references.

        Returns (stale_count, fresh_count, stale_evidence_refs).
        """
        stale_refs: List[str] = []
        stale_count = 0
        fresh_count = 0

        cutoff = trade_date - timedelta(days=stale_threshold_days)

        for ev in evidences:
            published_at = _parse_dt(ev.get("published_at"))
            eid = ev.get("evidence_id")
            if published_at is None:
                continue
            if published_at < cutoff:
                stale_count += 1
                if eid:
                    stale_refs.append(str(eid))
            else:
                fresh_count += 1

        # Also check signals for staleness via valid_to
        for s in signals:
            valid_to = _parse_dt(s.get("valid_to"))
            eid = s.get("entity_id")
            if valid_to and valid_to < trade_date:
                stale_count += 1
                if eid:
                    stale_refs.append(str(eid))
            elif valid_to:
                fresh_count += 1

        return stale_count, fresh_count, stale_refs
