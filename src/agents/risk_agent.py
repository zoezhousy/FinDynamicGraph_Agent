from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.agents.roles import AgentReport, risk_agent as run_risk_agent


class RiskAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("risk_agent")

    def run(self, input_data: dict) -> AgentReport:
        ticker = input_data["ticker"]
        trade_date = input_data["trade_date"]
        subgraph = input_data["subgraph"]
        return run_risk_agent(subgraph, ticker, trade_date)
