"""Integration test: Neo4j store + TechnicalAgent with real OHLCV data."""
from __future__ import annotations

import os
import pandas as pd
from datetime import datetime

from src.kg.store_neo4j import Neo4jKGStore
from src.agents.technical_agent import TechnicalAgent


URI = "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = os.environ.get("neo4j_password", "testpassword")


def test_store_init_and_health():
    """Test that Neo4j is reachable and constraints can be created."""
    store = Neo4jKGStore(URI, USER, PASSWORD)
    assert store.health_check(), "Neo4j health check failed"
    store.init_constraints()
    print("✅ Constraints initialized, health check passed")
    store.close()


def test_upsert_entities_and_relations():
    """Test upserting Company + IndicatorSignal entities and relations."""
    store = Neo4jKGStore(URI, USER, PASSWORD)
    store.init_constraints()

    from src.kg.schema import Entity, Relation

    entities = [
        Entity(
            entity_id="company:0005.HK",
            type="Company",
            properties={"ticker": "0005.HK", "name": "HSBC", "market": "HK"},
        ),
        Entity(
            entity_id="signal:0005.HK:rsi_oversold:2025-06-01",
            type="IndicatorSignal",
            properties={
                "ticker": "0005.HK",
                "name": "rsi_oversold",
                "direction": "bullish",
                "strength": 0.75,
                "as_of_date": "2025-06-01",
            },
        ),
    ]

    relations = [
        Relation(
            start_id="company:0005.HK",
            end_id="signal:0005.HK:rsi_oversold:2025-06-01",
            type="HAS_SIGNAL",
            as_of_date=datetime(2025, 6, 1),
            confidence=0.75,
            direction="bullish",
        ),
    ]

    store.upsert_entities(entities)
    store.upsert_relations(relations)

    # Verify data exists
    with store._driver.session(database=store._database) as session:
        result = session.run(
            "MATCH (c:Company {ticker: '0005.HK'}) RETURN c.name AS name"
        ).single()
        assert result["name"] == "HSBC", f"Expected HSBC, got {result['name']}"

        result = session.run(
            "MATCH (c:Company)-[:HAS_SIGNAL]->(s:IndicatorSignal) "
            "WHERE c.ticker = '0005.HK' RETURN count(s) AS cnt"
        ).single()
        assert result["cnt"] == 1, f"Expected 1 signal, got {result['cnt']}"

    print("✅ Entities and relations upserted and verified")
    store.close()


def test_technical_agent_with_ohlcv():
    """Test TechnicalAgent produces entities/relations from OHLCV data."""
    agent = TechnicalAgent()

    # Build a minimal OHLCV DataFrame (need at least 200 rows for MA200)
    import numpy as np
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 80.0
    prices = base + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 50)  # floor

    ohlcv = pd.DataFrame({
        "date": dates,
        "close": prices,
        "high": prices + np.abs(np.random.randn(n) * 0.3),
        "low": prices - np.abs(np.random.randn(n) * 0.3),
    })

    entities, relations = agent.run({"ticker": "0005.HK", "ohlcv": ohlcv})

    print(f"✅ TechnicalAgent produced {len(entities)} entities, {len(relations)} relations")

    # Show a sample of signals
    for ent in entities[:5]:
        p = ent.properties
        print(f"   {p.get('name', '?')} | {p.get('direction', '?')} | strength={p.get('strength', '?')} | {p.get('as_of_date', '?')}")

    assert len(entities) > 0, "Expected at least some signals"
    assert len(relations) > 0, "Expected at least some relations"

    # Now upsert to Neo4j
    store = Neo4jKGStore(URI, USER, PASSWORD)
    store.init_constraints()

    # Ensure company exists first
    from src.kg.schema import Entity
    store.upsert_entities([
        Entity(
            entity_id="company:0005.HK",
            type="Company",
            properties={"ticker": "0005.HK", "name": "HSBC", "market": "HK"},
        ),
    ])

    store.upsert_entities(entities)
    store.upsert_relations(relations)

    # Verify in Neo4j
    with store._driver.session(database=store._database) as session:
        result = session.run(
            "MATCH (c:Company)-[:HAS_SIGNAL]->(s:IndicatorSignal) "
            "WHERE c.ticker = '0005.HK' RETURN count(s) AS cnt"
        ).single()
        print(f"✅ Neo4j now has {result['cnt']} IndicatorSignal nodes for 0005.HK")
        assert result["cnt"] > 0

    store.close()


def test_decision_trace():
    """Test upserting a DecisionTrace with AgentAssessments."""
    store = Neo4jKGStore(URI, USER, PASSWORD)
    store.init_constraints()

    from src.kg.schema import DecisionTrace, AgentAssessment

    trace = DecisionTrace(
        decision_id="decision:test:0005.HK:2025-06-01",
        ticker="0005.HK",
        trade_date=datetime(2025, 6, 1),
        action="buy",
        final_score=0.72,
        confidence=0.8,
        conflict_level=0.1,
        decision_reason="Strong bullish signals from RSI oversold + MA crossover",
        evidence_ids=["ev_1", "ev_2"],
    )

    assessments = [
        AgentAssessment(
            assessment_id="assess:tech:0005.HK:2025-06-01",
            ticker="0005.HK",
            trade_date=datetime(2025, 6, 1),
            agent_role="technical",
            stance="bullish",
            confidence=0.8,
            score=0.75,
            summary="RSI oversold at 28, price above MA50",
            evidence_refs=["ev_1"],
            supports_decision=True,
        ),
    ]

    store.upsert_decision_trace(trace, assessments)

    # Verify
    with store._driver.session(database=store._database) as session:
        result = session.run(
            "MATCH (d:DecisionTrace {entity_id: $id}) RETURN d.action AS action",
            id="decision:test:0005.HK:2025-06-01",
        ).single()
        assert result["action"] == "buy"
        print("✅ DecisionTrace and AgentAssessment upserted and verified")

    store.close()


if __name__ == "__main__":
    test_store_init_and_health()
    test_upsert_entities_and_relations()
    test_technical_agent_with_ohlcv()
    test_decision_trace()
    print("\n🎉 All integration tests passed!")
