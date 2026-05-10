from src.agents.technical_agent import TechnicalAgent
from src.decision.graph_decision import GraphDecisionEngine
from src.kg.graph_store import GraphStore


def main() -> None:
    ticker = "0005.HK"

    graph_store = GraphStore()
    technical_agent = TechnicalAgent()
    decision_engine = GraphDecisionEngine()

    graph_update = technical_agent.run(
        {
            "ticker": ticker,
            "rsi": 28,
            "ma_short": 42.5,
            "ma_long": 40.8,
        }
    )

    graph_store.apply_update(graph_update)

    signals = graph_store.get_signals_for_ticker(ticker)
    decision = decision_engine.decide(ticker, signals)

    print("Graph summary:")
    print(graph_store.get_graph_summary())

    print("\nDecision:")
    print(decision.model_dump())


if __name__ == "__main__":
    main()