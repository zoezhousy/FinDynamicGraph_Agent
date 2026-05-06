import os
from neo4j import GraphDatabase

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", os.environ['neo4j_password'])

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connection established.")

    driver.execute_query("""
        CREATE (:Company {name: 'Tencent', market: 'HK'})
    """)

    records, summary, keys = driver.execute_query("""
        MATCH (c:Company)
        RETURN c.name AS name, c.market AS market
    """)

    for r in records:
        print(r["name"], r["market"])