from src.agents.technical_agent import TechnicalAgent
from src.kg.graph_store import GraphStore


def test_graph_store_applies_update():
    store = GraphStore()
    agent = TechnicalAgent()

    update = agent.run(
        {
            "ticker": "0005.HK",
            "rsi": 28,
            "ma_short": 42.5,
            "ma_long": 40.8,
        }
    )

    store.apply_update(update)

    summary = store.get_graph_summary()
    signals = store.get_signals_for_ticker("0005.HK")

    assert summary["num_nodes"] == 3
    assert summary["num_edges"] == 2
    assert len(signals) == 1