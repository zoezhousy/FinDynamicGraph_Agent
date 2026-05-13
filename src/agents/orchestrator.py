from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from src.agents.fundamental_agent import FundamentalAgent
from src.agents.kg_tools import KGAgentContext
from src.agents.risk_agent import RiskAgent
from src.agents.roles import (
    AgentReport,
    news_agent,
    portfolio_manager_decide,
    technical_agent,
)
from src.kg.schema import AgentAssessment, DecisionTrace
from src.kg.store_neo4j import Neo4jKGStore


@dataclass
class OrchestratorConfig:
    max_news: int = 20
    max_signals: int = 50
    persist_decision_trace: bool = True


class KGBasedOrchestrator:
    """KG-driven multi-agent orchestrator for Milestone 2."""

    def __init__(self, kg_context: KGAgentContext, config: OrchestratorConfig | None = None) -> None:
        self.kg_context = kg_context
        self.config = config or OrchestratorConfig()
        self.fundamental_agent_runner = FundamentalAgent()
        self.risk_agent_runner = RiskAgent()
        self.trace_store = self._init_trace_store() if self.config.persist_decision_trace else None

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

    def run_for_ticker(self, ticker: str, trade_date: datetime) -> Dict[str, Any]:
        subgraph = self.kg_context.load_subgraph(ticker, trade_date)
        reports: list[AgentReport] = []
        reports.append(news_agent(subgraph, ticker, trade_date))
        reports.append(technical_agent(subgraph, ticker, trade_date))
        reports.append(
            self.fundamental_agent_runner.run(
                {"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}
            )
        )
        reports.append(
            self.risk_agent_runner.run(
                {"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}
            )
        )
        decision = portfolio_manager_decide(ticker, trade_date, reports, subgraph=subgraph)
        if self.trace_store is not None:
            decision_trace = DecisionTrace.model_validate(decision["decision_trace"])
            assessments = [AgentAssessment.model_validate(row) for row in decision["agent_assessments"]]
            self.trace_store.upsert_decision_trace(decision_trace, assessments)
        return decision
