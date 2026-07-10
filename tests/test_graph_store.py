import numpy as np
import pandas as pd

from src.agents.technical_agent import TechnicalAgent
from src.kg.graph_store import GraphStore
from src.kg.schema import Entity, Relation


def _make_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 80.0 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 50)
    return pd.DataFrame({
        "date": dates,
        "close": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.3),
        "low": prices - np.abs(np.random.randn(n) * 0.3),
    })


def test_technical_agent_returns_entities_and_relations():
    agent = TechnicalAgent()
    ohlcv = _make_ohlcv()

    entities, relations = agent.run({"ticker": "0005.HK", "ohlcv": ohlcv})

    assert len(entities) > 0, "Expected at least some signal entities"
    assert len(relations) > 0, "Expected at least some relations"
    assert len(entities) == len(relations), "Each entity should have a matching relation"

    # Every entity should be an IndicatorSignal with required fields
    for ent in entities:
        assert ent.type == "IndicatorSignal"
        assert ent.entity_id.startswith("signal:0005.HK:")
        assert "direction" in ent.properties
        assert "strength" in ent.properties

    # Every relation should link Company -> IndicatorSignal
    for rel in relations:
        assert rel.start_id == "company:0005.HK"
        assert rel.type == "HAS_SIGNAL"
        assert rel.direction in ("bullish", "bearish", "neutral")


def test_graph_store_with_technical_agent_output():
    """GraphStore can ingest entities produced by TechnicalAgent."""
    store = GraphStore()
    agent = TechnicalAgent()
    ohlcv = _make_ohlcv()

    entities, relations = agent.run({"ticker": "0005.HK", "ohlcv": ohlcv})

    # GraphStore is in-memory MVP; add entities as nodes
    for ent in entities:
        store.add_entity(ent.entity_id, **ent.properties)

    summary = store.get_graph_summary()
    assert summary["num_nodes"] == len(entities)
