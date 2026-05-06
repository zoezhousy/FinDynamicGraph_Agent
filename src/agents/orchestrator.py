from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from src.agents.kg_tools import KGAgentContext
from src.agents.roles import (
    AgentReport,
    news_agent,
    portfolio_manager_decide,
    technical_agent,
)


@dataclass
class OrchestratorConfig:
    max_news: int = 20
    max_signals: int = 50


class KGBasedOrchestrator:
    """Simplified KG-driven multi-agent orchestrator.

    1. 从 KG 读取子图（公司 + 技术信号 + 新闻）
    2. 调用不同角色的 agent 生成报告
    3. 由 Portfolio Manager 聚合为结构化决策
    """

    def __init__(self, kg_context: KGAgentContext, config: OrchestratorConfig | None = None) -> None:
        self.kg_context = kg_context
        self.config = config or OrchestratorConfig()

    def run_for_ticker(self, ticker: str, trade_date: datetime) -> Dict[str, Any]:
        subgraph = self.kg_context.load_subgraph(ticker, trade_date)
        reports: list[AgentReport] = []
        reports.append(news_agent(subgraph))
        reports.append(technical_agent(subgraph))
        decision = portfolio_manager_decide(ticker, trade_date, reports)
        return decision

