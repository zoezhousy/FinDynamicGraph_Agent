from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


SignalType = Literal[
    "technical",
    "fundamental",
    "news",
    "risk",
    "macro",
    "sentiment",
]

Direction = Literal["bullish", "bearish", "neutral", "uncertain"]
Action = Literal["buy", "sell", "hold", "abstain"]
EntityType = Literal[
    "Company",
    "IndicatorSignal",
    "NewsEvent",
    "FundamentalSignal",
    "RiskEvent",
    "Sector",
    "Index",
    "Event",
]
RelationType = Literal["HAS_SIGNAL", "MENTIONED_IN", "SUPPORTED_BY", "HAS_RISK", "RELATES_TO"]


class Evidence(BaseModel):
    evidence_id: str
    source_type: str
    source_name: str | None = None
    url: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    extracted_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class Entity(BaseModel):
    entity_id: str
    type: EntityType
    properties: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    start_id: str
    end_id: str
    type: RelationType
    as_of_date: datetime
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    direction: Direction | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    evidence_ids: list[str] | None = None


class FinancialSignal(BaseModel):
    signal_id: str
    ticker: str
    signal_type: SignalType
    direction: Direction
    strength: float = Field(ge=0.0, le=1.0)
    valid_from: datetime
    valid_to: datetime | None = None
    evidence_id: str
    description: str


class GraphUpdate(BaseModel):
    ticker: str
    signal: FinancialSignal
    evidence: Evidence


class TradingDecision(BaseModel):
    ticker: str
    action: Action
    bullish_score: float
    bearish_score: float
    neutral_score: float
    evidence_ids: list[str]
    reason: str
