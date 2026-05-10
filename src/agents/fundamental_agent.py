# fundamental agent to generate fundamental signals for a given ticker

from src.agents.base_agent import BaseAgent

class FundamentalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("fundamental_agent")

    def run(self, input_data: dict) -> dict:
        ticker = input_data["ticker"]
        financial_data = input_data.get("financial_data")
        return {
            "ticker": ticker,
            "financial_data": financial_data,
        }