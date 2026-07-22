"""Multi-agent orchestrator for the KG-driven trading system.

Supports two backends controlled by ``AGENT_BACKEND`` env var:
- ``langgraph`` (default): LangGraph tool-calling agents with async parallel execution
- ``legacy``: Original direct LLM calls (for backward compatibility)

Portfolio Manager remains deterministic (weighted aggregation) in both modes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from src.agents.kg_tools import KGAgentContext, build_kg_tools, DynamicDatePolicy
from src.agents.roles import AgentReport, portfolio_manager_decide
from src.kg.schema import AgentAssessment, DecisionTrace
from src.kg.store_neo4j import Neo4jKGStore

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    max_news: int = 20
    max_signals: int = 50
    persist_decision_trace: bool = True


class KGBasedOrchestrator:
    """KG-driven multi-agent orchestrator."""

    def __init__(self, kg_context: KGAgentContext, config: OrchestratorConfig | None = None) -> None:
        self.kg_context = kg_context
        self.config = config or OrchestratorConfig()
        self.backend = os.getenv("AGENT_BACKEND", "langgraph")
        self.trace_store = self._init_trace_store() if self.config.persist_decision_trace else None

        if self.backend == "langgraph":
            self._init_langgraph()
        else:
            self._init_legacy()

    # ── Initialization ─────────────────────────────────────────────────

    def _init_langgraph(self) -> None:
        from src.agents.langgraph_agents import make_llm
        self.llm = make_llm()
        self.semaphore = asyncio.Semaphore(2)
        logger.info("Initialized LangGraph backend (model=%s)", os.getenv("LLM_MODEL"))

    def _init_legacy(self) -> None:
        from src.agents.fundamental_agent import FundamentalAgent
        from src.agents.risk_agent import RiskAgent
        from src.agents.roles import news_agent, technical_agent
        self.legacy_news = news_agent
        self.legacy_technical = technical_agent
        self.legacy_fundamental = FundamentalAgent()
        self.legacy_risk = RiskAgent()
        logger.info("Initialized legacy backend")

    def _init_trace_store(self) -> Neo4jKGStore | None:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        if not (neo4j_uri and neo4j_user and neo4j_password):
            return None
        store = Neo4jKGStore(neo4j_uri, neo4j_user, neo4j_password, database=neo4j_database)
        if not store.health_check():
            store.close()
            return None
        store.init_constraints()
        return store

    def close(self) -> None:
        if self.trace_store is not None:
            self.trace_store.close()

    # ── LangGraph backend ──────────────────────────────────────────────

    async def _run_agent_safely(
        self, agent, input_data: dict, role: str, ticker: str
    ) -> AgentReport:
        """Run a single LangGraph agent with timeout + error recovery."""
        from src.agents.langgraph_agents import _extract_report
        try:
            async with self.semaphore:
                result = await asyncio.wait_for(
                    agent.ainvoke(input_data), timeout=90
                )
            # Extract structured report from final message
            last_content = result["messages"][-1].content
            schema = _extract_report(last_content, role)
            if schema:
                return AgentReport(role=role, **schema.model_dump())
            logger.warning("%s agent returned unparseable output", role)
        except Exception as exc:
            logger.error("%s agent failed: %s", role, exc)

        # Fallback: neutral report
        return AgentReport(
            role=role, stance="uncertain", confidence=0.0, score=0.0,
            summary=f"{role} agent failed — no data available.",
            evidence_refs=[], claim_refs=[], factors=[],
        )

    async def _run_langgraph_async(
        self, ticker: str, trade_date: datetime
    ) -> Dict[str, Any]:
        from src.agents.langgraph_agents import (
            create_news_agent, create_technical_agent,
            create_fundamental_agent, create_risk_agent,
        )

        # Fresh tools with fresh cache per invocation
        tools = build_kg_tools(self.kg_context.query_client, DynamicDatePolicy())
        news_ag = create_news_agent(self.llm, tools)
        tech_ag = create_technical_agent(self.llm, tools)
        fund_ag = create_fundamental_agent(self.llm, tools)
        risk_ag = create_risk_agent(self.llm, tools)

        input_data = {
            "messages": [{"role": "user", "content": f"Analyze {ticker} as of {trade_date.date().isoformat()}"}]
        }

        reports = await asyncio.gather(
            self._run_agent_safely(news_ag, input_data, "news", ticker),
            self._run_agent_safely(tech_ag, input_data, "technical", ticker),
            self._run_agent_safely(fund_ag, input_data, "fundamental", ticker),
            self._run_agent_safely(risk_ag, input_data, "risk", ticker),
        )

        # Load subgraph once for PM (evidence validation, conflict detection)
        subgraph = self.kg_context.load_subgraph(ticker, trade_date)
        return portfolio_manager_decide(ticker, trade_date, list(reports), subgraph=subgraph)

    # ── Legacy backend ─────────────────────────────────────────────────

    def _run_legacy(self, ticker: str, trade_date: datetime) -> Dict[str, Any]:
        subgraph = self.kg_context.load_subgraph(ticker, trade_date)
        reports: list[AgentReport] = []
        reports.append(self.legacy_news(subgraph, ticker, trade_date))
        reports.append(self.legacy_technical(subgraph, ticker, trade_date))
        reports.append(
            self.legacy_fundamental.run(
                {"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}
            )
        )
        reports.append(
            self.legacy_risk.run(
                {"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}
            )
        )
        return portfolio_manager_decide(ticker, trade_date, reports, subgraph=subgraph)

    # ── Public API ─────────────────────────────────────────────────────

    def run_for_ticker(self, ticker: str, trade_date: datetime) -> Dict[str, Any]:
        if self.backend == "langgraph":
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                decision = asyncio.run(self._run_langgraph_async(ticker, trade_date))
            else:
                raise RuntimeError(
                    "An event loop is already running. Use await run_for_ticker_async() instead."
                )
        else:
            decision = self._run_legacy(ticker, trade_date)

        # Persist decision trace
        if self.trace_store is not None:
            try:
                decision_trace = DecisionTrace.model_validate(decision["decision_trace"])
                assessments = [AgentAssessment.model_validate(row) for row in decision["agent_assessments"]]
                self.trace_store.upsert_decision_trace(decision_trace, assessments)
            except Exception as exc:
                logger.error("Failed to persist decision trace: %s", exc)

        return decision
