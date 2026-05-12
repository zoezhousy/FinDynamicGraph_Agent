import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is not set in environment variables.")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

driver.verify_connectivity()
print("Neo4j connected successfully.")

with driver.session(database=NEO4J_DATABASE) as session:
    result = session.run("RETURN 'Hello Neo4j' AS msg")
    print(result.single()["msg"])

driver.close()