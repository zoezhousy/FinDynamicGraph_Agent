from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

from src.agents.kg_tools import KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator, OrchestratorConfig
from src.agents.technical_agent import TechnicalAgent
from src.collectors.fundamental_collector import FundamentalCollector
from src.collectors.global_news_collector import GlobalNewsCollector
from src.collectors.market_collector import MarketCollector
from src.collectors.news_collector import NewsCollector
from src.collectors.newsapi_collector import NewsAPICollector
from src.config import CollectionConfig
from src.kg.query import KGQueryClient
from src.kg.schema import Entity
from src.kg.connections import Neo4jSettings, initialized_store
from src.kg.ingestion import write_graph_batch
from src.collectors.cache import dedup_news_frames, resolve_ohlcv_path
from src.kg.store_neo4j import Neo4jKGStore
from src.kg.update_pipeline import (
    build_fundamentals_from_frame,
    build_global_news_from_frame,
    build_news_from_frame,
    build_risk_events_from_frame,
)
from src.report_formatter import format_report
from src.sim.backtest import BacktestConfig, compute_trade_return


@dataclass
class PipelineResult:
    """一次完整分析的结果"""
    ticker: str
    trade_date: str
    decision: Dict[str, Any]
    backtest: Dict[str, Any]
    report_text: str
    raw_data: Dict[str, Any]


class AnalysisPipeline:
    """可复用的分析 Pipeline，封装整个流程：采集 → KG → Agent → 决策 → 报告"""

    def __init__(self, config: CollectionConfig | None = None, *, kg_store=None, kg_client=None, orchestrator=None, collectors=None, technical_agent=None):
        self.config = config or CollectionConfig()
        self._kg_store = kg_store
        self._kg_client = kg_client
        self._orchestrator = orchestrator
        self._owned = []
        self._collectors = collectors or {}
        self._technical_agent = technical_agent

    def _ensure_kg(self):
        if self._kg_store is not None and self._kg_client is not None:
            return
        settings = Neo4jSettings.from_env(self.config.neo4j_database)
        created = []
        try:
            store = self._kg_store
            client = self._kg_client
            if store is None:
                store = initialized_store(settings, Neo4jKGStore)
                created.append(store)
            if client is None:
                client = settings.create(KGQueryClient)
                created.append(client)
        except Exception:
            for resource in reversed(created):
                resource.close()
            raise
        self._kg_store, self._kg_client = store, client
        self._owned.extend(created)

    def _ensure_orchestrator(self):
        """懒初始化多智能体 Orchestrator"""
        if self._orchestrator is not None:
            return
        self._ensure_kg()
        kg_ctx = KGAgentContext(self._kg_client)
        self._orchestrator = KGBasedOrchestrator(
            kg_ctx,
            OrchestratorConfig(
                max_news=self.config.news_limit_per_ticker,
                persist_decision_trace=True,
            ),
        )
        self._owned.append(self._orchestrator)

    # ── Step 1: 数据采集 + KG 更新 ──

    def collect_and_update_kg(self, ticker: str) -> Dict[str, Any]:
        self._ensure_kg()
        from src.collectors.service import AnalysisCollectionService
        return AnalysisCollectionService(self.config, self._kg_store,
            collectors=self._collectors, technical_agent=self._technical_agent).run(ticker)

    # ── Step 2: 多智能体决策 ──

    def run_decision(self, ticker: str, trade_date: datetime | None = None) -> Dict[str, Any]:
        """运行多智能体决策（假设 KG 中已有数据）"""
        self._ensure_orchestrator()
        trade_date = trade_date or datetime.now()
        return self._orchestrator.run_for_ticker(ticker, trade_date)

    # ── Step 3: 一键运行 ──

    def run(
        self,
        ticker: str,
        trade_date: datetime | None = None,
        with_backtest: bool = False,
        skip_collect: bool = False,
    ) -> PipelineResult:
        """
        一键分析：(可选采集) → 多智能体决策 → 生成报告

        Args:
            ticker: 股票代码，如 "0700.HK"
            trade_date: 交易日期，默认今天
            with_backtest: 是否同时跑回测
            skip_collect: 跳过数据采集，直接用现有 KG 数据
        """
        trade_date = trade_date or datetime.now()

        collect_summary: Dict[str, Any] = {}

        # Step 1: 数据采集 + KG（可跳过）
        if not skip_collect:
            collect_summary = self.collect_and_update_kg(ticker)

        # Step 2: 决策
        decision = self.run_decision(ticker, trade_date)

        # Step 3: 可选回测
        bt_result: Dict[str, Any] = {}
        if with_backtest:
            bt_cfg = BacktestConfig()
            ticker_dir = self.config.output_root / ticker
            ohlcv_path = resolve_ohlcv_path(self.config.output_root, ticker, current_only=True)
            if ohlcv_path.exists():
                ohlcv_df = pd.read_parquet(ohlcv_path)
                bt_result = compute_trade_return(
                    ohlcv_df,
                    trade_date.date().isoformat(),
                    decision["action"],
                    bt_cfg,
                )

        # Step 4: 生成可读报告
        report_text = format_report(decision, collect_summary, bt_result)

        return PipelineResult(
            ticker=ticker,
            trade_date=trade_date.date().isoformat(),
            decision=decision,
            backtest=bt_result,
            report_text=report_text,
            raw_data={"collect_summary": collect_summary},
        )

    # ── 资源释放 ──

    def close(self):
        resources, self._owned = self._owned, []
        try:
            for resource in reversed(resources):
                try:
                    resource.close()
                except Exception:
                    logging.exception("Failed to close pipeline resource")
        finally:
            self._orchestrator = self._kg_client = self._kg_store = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
