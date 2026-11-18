from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import json
import importlib
import logging
import pandas as pd
import pytest
from src.agents.models import AgentReport
from src.agents.pm_deterministic import portfolio_manager_decide
from src.collectors.cache import dedup_news_frames, resolve_ohlcv_path, fetch_ohlcv_with_cache
from src.kg.ingestion import write_graph_batch
from src.kg.connections import Neo4jSettings, initialized_store
from src.pipeline import AnalysisPipeline
from src.config import CollectionConfig

def test_saved_portfolio_behavior_is_unchanged():
    cases = json.loads((Path(__file__).parent / "fixtures" / "portfolio_legacy.json").read_text(encoding="utf-8"))
    for case in cases:
        actual = portfolio_manager_decide("0700.HK", datetime(2025, 6, 1), [AgentReport(**r) for r in case["reports"]], case["subgraph"])
        assert actual == case["expected"]

def test_old_imports_and_import_logging_side_effects():
    from src.agents.roles import AgentReport as OldReport, portfolio_manager_decide as old_pm
    assert OldReport is AgentReport
    assert old_pm is portfolio_manager_decide
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    import src.main_experiment as entry
    importlib.reload(entry)
    assert list(root.handlers) == handlers
    assert root.level == level

def test_news_profiles_preserve_source_precedence_and_order():
    frames = [pd.DataFrame({"url": ["u1", "u2"], "title": ["first", "second"], "published_time": ["2025-01-01", "2025-01-03"]}), pd.DataFrame({"url": ["u1", "u3"], "title": ["duplicate", "third"], "published_time": ["2025-01-04", "2025-01-02"]})]
    assert dedup_news_frames(frames, sort_by_time=False).title.tolist() == ["first", "second", "third"]
    assert dedup_news_frames(frames).title.tolist() == ["second", "third", "first"]
    assert dedup_news_frames([]).empty

def test_ohlcv_file_precedence_and_current_only(tmp_path):
    directory = tmp_path / "0700.HK"
    directory.mkdir()
    (directory / "ohlcv_2021_2025.parquet").touch()
    assert resolve_ohlcv_path(tmp_path, "0700.HK").name == "ohlcv_2021_2025.parquet"
    assert resolve_ohlcv_path(tmp_path, "0700.HK", current_only=True).name == "ohlcv_2021_now.parquet"
    (directory / "ohlcv_2021_now.parquet").touch()
    assert resolve_ohlcv_path(tmp_path, "0700.HK").name == "ohlcv_2021_now.parquet"
    assert resolve_ohlcv_path(tmp_path, "0700.HK", legacy_first=True).name == "ohlcv_2021_2025.parquet"

def test_failed_incremental_fetch_retains_cached_frame(tmp_path):
    cached = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "close": [100.]})
    cached.to_parquet(tmp_path / "ohlcv_2021_now.parquet", index=False)
    def fail(*args, **kwargs):
        raise OSError("synthetic provider failure")
    collector = SimpleNamespace(config=SimpleNamespace(end_date="2026-10-03"), fetch_ohlcv=fail)
    pd.testing.assert_frame_equal(fetch_ohlcv_with_cache("0700.HK", collector, tmp_path), cached)

def test_graph_batch_preserves_write_order_and_empty_batches():
    calls = []
    store = SimpleNamespace(upsert_entities=lambda x: calls.append(("entities", x)), upsert_evidences=lambda x: calls.append(("evidences", x)), upsert_relations=lambda x: calls.append(("relations", x)))
    write_graph_batch(store, [], [], [])
    assert [c[0] for c in calls] == ["entities", "evidences", "relations"]
    calls.clear()
    write_graph_batch(store, [], None, [])
    assert [c[0] for c in calls] == ["entities", "relations"]

@pytest.mark.parametrize("healthy, constraint_failure", [(False, False), (True, True)])
def test_failed_store_initialization_closes_once(healthy, constraint_failure):
    calls = []
    def constraints():
        if constraint_failure:
            raise OSError("synthetic")
    resource = SimpleNamespace(health_check=lambda: healthy, init_constraints=constraints, close=lambda: calls.append("close"))
    with pytest.raises((RuntimeError, OSError)):
        initialized_store(Neo4jSettings("uri", "user", "unused"), lambda *a, **k: resource)
    assert calls == ["close"]

def test_pipeline_does_not_close_injected_resources():
    calls = []
    resource = SimpleNamespace(close=lambda: calls.append("close"))
    pipeline = AnalysisPipeline(kg_store=resource, kg_client=resource, orchestrator=resource)
    pipeline.close()
    pipeline.close()
    assert calls == []

def test_query_creation_failure_closes_owned_store_and_allows_retry(monkeypatch):
    import src.pipeline as module
    calls = []
    store = SimpleNamespace(health_check=lambda: True, init_constraints=lambda: None, close=lambda: calls.append("close"))
    monkeypatch.setattr(module, "Neo4jKGStore", lambda *a, **k: store)
    def fail(*a, **k):
        raise OSError("synthetic initialization error")
    monkeypatch.setattr(module, "KGQueryClient", fail)
    for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        monkeypatch.setenv(key, "unused")
    pipeline = AnalysisPipeline()
    with pytest.raises(OSError):
        pipeline._ensure_kg()
    assert pipeline._kg_store is None
    assert calls == ["close"]

def test_single_ticker_collection_accepts_injected_providers(tmp_path):
    from src.collectors.service import AnalysisCollectionService
    writes = []
    store = SimpleNamespace(upsert_entities=lambda x: writes.append("entities"), upsert_evidences=lambda x: writes.append("evidences"), upsert_relations=lambda x: writes.append("relations"))
    empty = pd.DataFrame(columns=["url", "date", "close", "metric", "value", "as_of_date"])
    def optional_failure(*a, **k):
        raise ValueError("optional source unavailable")
    providers = {"market": SimpleNamespace(fetch_ohlcv=lambda *a, **k: pd.DataFrame({"date": [pd.Timestamp("2025-06-01")], "close": [100.]})), "news": SimpleNamespace(fetch_news=lambda *a, **k: empty), "newsapi": SimpleNamespace(fetch_news=optional_failure), "global_news": SimpleNamespace(fetch_global_news=lambda *a, **k: empty), "fundamental": SimpleNamespace(fetch_fundamentals=lambda *a, **k: empty)}
    tech = SimpleNamespace(build_signal_entities_from_ohlcv=lambda *a: ([], []))
    result = AnalysisCollectionService(CollectionConfig(output_root=tmp_path), store, collectors=providers, technical_agent=tech).run("0700.HK")
    assert result == {"ticker": "0700.HK", "ohlcv_rows": 1, "news_rows": 0, "fundamental_rows": 0, "global_news_rows": 0}
    assert writes[0] == "entities"
