"""Pydantic schemas for LangGraph agent structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FactorSchema(BaseModel):
    name: str
    direction: Literal["bullish", "bearish", "neutral", "uncertain"]
    weight: float = Field(ge=0.0, le=1.0)
    value: str | None = None


class AgentReportSchema(BaseModel):
    """Structured output from an analyst agent.

    The ``role`` field is NOT part of this schema — it is injected by the
    caller (e.g. ``AgentReport(role="news", **schema.model_dump())``).
    """

    stance: Literal["bullish", "bearish", "neutral", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    claim_refs: list[str] = Field(default_factory=list)
    factors: list[FactorSchema] = Field(default_factory=list)
