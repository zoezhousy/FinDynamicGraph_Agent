import pytest
pytestmark = pytest.mark.integration

def test_neo4j_connectivity(neo4j_driver):
    neo4j_driver.verify_connectivity()
