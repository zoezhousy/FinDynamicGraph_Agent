# schema for the financial knowledge graph

# In-memory graph for MVP
# Neo4j for future development
# 

from datetime import datetime
from typing import Literal, Optional
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


class Evidence(BaseModel):
    evidence_id: str
    source_type: str
    source_name: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    extracted_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class FinancialEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: Literal["stock", "company", "index", "sector", "event"]


class FinancialSignal(BaseModel):
    signal_id: str
    ticker: str
    signal_type: SignalType
    direction: Direction
    strength: float = Field(ge=0.0, le=1.0)
    valid_from: datetime
    valid_to: Optional[datetime] = None
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