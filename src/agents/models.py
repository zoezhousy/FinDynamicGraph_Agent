"""Dependency-free domain reports shared by both agent backends."""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

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


