from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Optional


EntityType = Literal["Company", "IndicatorSignal", "NewsEvent", "RiskEvent"]
RelationType = Literal[
    "HAS_SIGNAL",
    "MENTIONED_IN",
    "AFFECTS",
    "CONTRADICTS",
]


@dataclass
class Evidence:
    evidence_id: str
    source_type: Literal["news", "price_signal"]
    url: Optional[str]
    published_time: Optional[datetime]
    snippet: Optional[str]


@dataclass
class Entity:
    entity_id: str
    type: EntityType
    properties: Dict[str, object]


@dataclass
class Relation:
    start_id: str
    end_id: str
    type: RelationType
    as_of_date: datetime
    confidence: float
    direction: Optional[Literal["bullish", "bearish", "neutral"]] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    evidence_ids: Optional[List[str]] = None

