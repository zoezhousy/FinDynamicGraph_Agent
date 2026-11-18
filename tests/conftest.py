"""Offline by default; explicit external checks never run during collection."""
import os
import socket
import pytest

EXTERNAL_FILES = {"test_neo4j.py", "test_neo4j_2.py", "test_neo4j_integration.py", "test_alpha_vantage.py", "test_yfinance_news.py", "test_hkex.py", "test_check_neo4j_write.py"}

def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true", help="Enable external-service tests")

def pytest_ignore_collect(collection_path, config):
    return collection_path.name in EXTERNAL_FILES and not config.getoption("--integration")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --integration to enable external checks"))

@pytest.fixture(autouse=True)
def block_offline_network(request, monkeypatch):
    if request.config.getoption("--integration") and "integration" in request.node.keywords:
        return
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is disabled in offline tests")
    original_connect = socket.socket.connect
    original_socketpair = socket.socketpair
    def local_socketpair(*args, **kwargs):
        # Windows asyncio creates its private wakeup pipe using loopback sockets.
        # Permit only socketpair construction; ordinary connections stay blocked.
        with monkeypatch.context() as context:
            context.setattr(socket.socket, "connect", original_connect)
            return original_socketpair(*args, **kwargs)
    monkeypatch.setattr(socket, "socketpair", local_socketpair)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)

@pytest.fixture
def neo4j_driver():
    from neo4j import GraphDatabase
    required = [os.getenv(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")]
    if not all(required):
        pytest.skip("NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD are required")
    with GraphDatabase.driver(required[0], auth=(required[1], required[2])) as driver:
        yield driver


_collection_guard = None

def pytest_sessionstart(session):
    global _collection_guard
    _collection_guard = pytest.MonkeyPatch()
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is disabled during test collection")
    _collection_guard.setattr(socket.socket, "connect", blocked)
    _collection_guard.setattr(socket.socket, "connect_ex", blocked)
    _collection_guard.setattr(socket, "getaddrinfo", blocked)

def pytest_collection_finish(session):
    global _collection_guard
    if _collection_guard is not None:
        _collection_guard.undo()
        _collection_guard = None

def pytest_sessionfinish(session, exitstatus):
    global _collection_guard
    if _collection_guard is not None:
        _collection_guard.undo()
        _collection_guard = None
