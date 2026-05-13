from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase, basic_auth


def main() -> None:
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not (uri and user and password):
        raise RuntimeError("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD are required.")

    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    queries = {
        "node_counts": """
            MATCH (n)
            RETURN labels(n) AS labels, count(n) AS count
            ORDER BY count DESC
        """,
        "fundamental_summary": """
            MATCH (c:Company)
            OPTIONAL MATCH (c)-[:HAS_SIGNAL]->(f:FundamentalSignal)
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(n:NewsEvent)
            OPTIONAL MATCH (d:DecisionTrace)-[:FOR_COMPANY]->(c)
            OPTIONAL MATCH (d)-[:HAS_OUTCOME]->(b:BacktestOutcome)
            RETURN
              c.ticker AS ticker,
              count(DISTINCT f) AS fundamental_signals,
              count(DISTINCT n) AS news_events,
              count(DISTINCT d) AS decision_traces,
              count(DISTINCT b) AS backtest_outcomes
            ORDER BY ticker
        """,
        "fundamental_chain_sample": """
            MATCH (c:Company)-[:HAS_SIGNAL]->(f:FundamentalSignal)
            OPTIONAL MATCH (f)-[:SUPPORTED_BY]->(e:Evidence)
            OPTIONAL MATCH (e)-[:SUPPORTS_CLAIM]->(cl:Claim)
            RETURN
              c.ticker AS ticker,
              f.metric AS metric,
              f.value AS value,
              f.direction AS direction,
              e.evidence_id AS evidence_id,
              cl.claim_id AS claim_id,
              cl.text AS claim_text
            LIMIT 20
        """,
    }

    try:
        with driver.session(database=database) as session:
            for name, cypher in queries.items():
                print(f"\n=== {name} ===")
                records = session.run(cypher)
                for record in records:
                    print(dict(record))
    finally:
        driver.close()


if __name__ == "__main__":
    main()