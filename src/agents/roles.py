from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal


DecisionAction = Literal["buy", "sell", "hold", "abstain"]


@dataclass
class AgentReport:
    role: str
    summary: str
    stance: Literal["bullish", "bearish", "neutral", "uncertain"]
    evidence_refs: List[Dict[str, Any]]


def news_agent(subgraph: Dict[str, List[Dict[str, Any]]]) -> AgentReport:
    news_nodes = subgraph.get("news", [])
    if not news_nodes:
        return AgentReport(
            role="news",
            summary="No recent news available.",
            stance="neutral",
            evidence_refs=[],
        )
    # 简化：按条数直接给一个“信息量多→更确定”的信号
    stance = "bullish" if len(news_nodes) >= 3 else "neutral"
    refs = [{"url": n.get("url"), "title": n.get("title")} for n in news_nodes[:5]]
    return AgentReport(
        role="news",
        summary=f"{len(news_nodes)} recent news items loaded.",
        stance=stance,
        evidence_refs=refs,
    )


def technical_agent(subgraph: Dict[str, List[Dict[str, Any]]]) -> AgentReport:
    signals = subgraph.get("signals", [])
    bullish = sum(1 for s in signals if s.get("direction") == "bullish")
    bearish = sum(1 for s in signals if s.get("direction") == "bearish")
    if bullish == bearish == 0:
        stance: Literal["bullish", "bearish", "neutral", "uncertain"] = "neutral"
    elif bullish > bearish:
        stance = "bullish"
    elif bearish > bullish:
        stance = "bearish"
    else:
        stance = "uncertain"
    return AgentReport(
        role="technical",
        summary=f"{bullish} bullish vs {bearish} bearish technical signals.",
        stance=stance,
        evidence_refs=[{"id": s.get("entity_id"), "direction": s.get("direction")} for s in signals],
    )


def portfolio_manager_decide(
    ticker: str,
    trade_date: datetime,
    reports: List[AgentReport],
) -> Dict[str, Any]:
    # 简单多数投票规则
    score = 0
    for r in reports:
        if r.stance == "bullish":
            score += 1
        elif r.stance == "bearish":
            score -= 1

    if score >= 2:
        action: DecisionAction = "buy"
    elif score <= -2:
        action = "sell"
    elif -1 <= score <= 1:
        action = "hold"
    else:
        action = "abstain"

    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": action,
        "score": score,
        "reports": [r.__dict__ for r in reports],
    }

