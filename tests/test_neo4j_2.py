import pytest
pytestmark = pytest.mark.integration

def test_neo4j_query(neo4j_driver):
    import os
    with neo4j_driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
        assert session.run("RETURN 'Hello Neo4j' AS msg").single()["msg"] == 'Hello Neo4j'
