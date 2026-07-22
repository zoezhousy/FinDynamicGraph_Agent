"""LangGraph tool-calling analyst agents.

Each factory function creates a LangGraph agent that:
1. Receives a ticker + date prompt
2. Calls KG tools to gather data
3. Analyzes the data
4. Returns a structured JSON report matching ``AgentReportSchema``

The agents use prompt-based JSON extraction (not ``ToolStrategy``) because
the project's OpenAI-compatible endpoint does not support native structured
output via ``response_format``.
"""

from __future__ import annotations

import json
import logging
import os
import re

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.agents.schemas import AgentReportSchema

logger = logging.getLogger(__name__)

# ── JSON output instructions (appended to each agent's system prompt) ──

_JSON_INSTRUCTIONS = """

OUTPUT FORMAT — MANDATORY:
You MUST respond with a single valid JSON object matching this exact schema:
{{
  "stance": "bullish" | "bearish" | "neutral" | "uncertain",
  "confidence": <float 0.0 to 1.0>,
  "score": <float -1.0 to 1.0>,
  "summary": "<string>",
  "evidence_refs": ["<evidence_id>", ...],
  "claim_refs": ["<claim_id>", ...],
  "factors": [{{"name": "<string>", "direction": "bullish|bearish|neutral|uncertain", "weight": <float 0.0-1.0>, "value": "<string|null>"}}]
}}

Rules:
- evidence_refs and claim_refs MUST be IDs you actually saw in tool results. Do NOT invent IDs.
- Return ONLY the JSON object. No markdown fences, no explanation before or after.
"""


# ── LLM factory ────────────────────────────────────────────────────────


def make_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance from environment variables."""
    return ChatOpenAI(
        base_url=os.getenv("LLM_API_BASE"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )


# ── System prompts ─────────────────────────────────────────────────────

NEWS_SYSTEM_PROMPT = """\
You are a financial news analyst.
Your job is to analyze news events and evidence from the knowledge graph.
Use the query_news and query_claims tools to gather data, then assess the
overall news sentiment. Focus on recency, source credibility, and claim
support/contradiction relationships.
""" + _JSON_INSTRUCTIONS

TECHNICAL_SYSTEM_PROMPT = """\
You are a technical analyst.
Your job is to analyze technical indicator signals from the knowledge graph.
Use the query_signals tool to gather data, then assess the overall technical
picture. Consider signal strength, agreement between indicators, and trend
direction.
""" + _JSON_INSTRUCTIONS

FUNDAMENTAL_SYSTEM_PROMPT = """\
You are a fundamental analyst.
Your job is to analyze fundamental metrics from the knowledge graph.
Use the query_fundamentals and query_claims tools to gather data, then assess
the company's financial health. Consider margins, growth, valuation, and
evidence support.
""" + _JSON_INSTRUCTIONS

RISK_SYSTEM_PROMPT = """\
You are a risk analyst.
Your job is to assess risk events from the knowledge graph.
Use the query_risks and query_claims tools to gather data, then assess the
overall risk level. Consider risk severity, count, and evidence chains.
""" + _JSON_INSTRUCTIONS


# ── Agent factories ────────────────────────────────────────────────────


def _extract_report(content: str, role: str) -> AgentReportSchema | None:
    """Extract and validate an AgentReportSchema from LLM output."""
    # Strip markdown fences
    if content.startswith("```"):
        content = re.sub(r"^```json\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    # Try direct parse
    try:
        return AgentReportSchema.model_validate_json(content)
    except ValidationError:
        pass

    # Try extracting first {...} block
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return AgentReportSchema.model_validate_json(match.group(0))
        except ValidationError:
            pass

    logger.warning("Failed to parse %s agent output: %s", role, content[:200])
    return None


def create_news_agent(llm: ChatOpenAI, tools: list):
    """Create a news analyst LangGraph agent."""
    subset = [t for t in tools if t.name in ("query_subgraph", "query_news", "query_claims")]
    return create_agent(
        model=llm,
        tools=subset,
        system_prompt=NEWS_SYSTEM_PROMPT,
    )


def create_technical_agent(llm: ChatOpenAI, tools: list):
    """Create a technical analyst LangGraph agent."""
    subset = [t for t in tools if t.name in ("query_signals", "query_subgraph")]
    return create_agent(
        model=llm,
        tools=subset,
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
    )


def create_fundamental_agent(llm: ChatOpenAI, tools: list):
    """Create a fundamental analyst LangGraph agent."""
    subset = [t for t in tools if t.name in ("query_fundamentals", "query_claims", "query_subgraph")]
    return create_agent(
        model=llm,
        tools=subset,
        system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
    )


def create_risk_agent(llm: ChatOpenAI, tools: list):
    """Create a risk analyst LangGraph agent."""
    subset = [t for t in tools if t.name in ("query_risks", "query_claims", "query_subgraph")]
    return create_agent(
        model=llm,
        tools=subset,
        system_prompt=RISK_SYSTEM_PROMPT,
    )
