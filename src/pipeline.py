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
from src.config import CollectionConfig
from src.kg.query import KGQueryClient
from src.kg.schema import Entity
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

    def __init__(self, config: CollectionConfig | None = None):
        load_dotenv()
        self.config = config or CollectionConfig()
        self._kg_store: Neo4jKGStore | None = None
        self._kg_client: KGQueryClient | None = None
        self._orchestrator: KGBasedOrchestrator | None = None

    # ── 连接管理 ──

    def _ensure_kg(self):
        """懒初始化 Neo4j 连接"""
        if self._kg_store is not None:
            return
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        pwd = os.getenv("NEO4J_PASSWORD")
        if not (uri and user and pwd):
            raise RuntimeError("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD must be set in .env")
        self._kg_store = Neo4jKGStore(uri, user, pwd, database=self.config.neo4j_database)
        if not self._kg_store.health_check():
            self._kg_store.close()
            raise RuntimeError("Neo4j health check failed — is it running?")
        self._kg_store.init_constraints()
        self._kg_client = KGQueryClient(uri, user, pwd, database=self.config.neo4j_database)

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

    # ── Step 1: 数据采集 + KG 更新 ──

    def collect_and_update_kg(self, ticker: str) -> Dict[str, Any]:
        """
        采集 OHLCV / News / Fundamentals 并写入 Neo4j KG。

        Returns:
            采集结果摘要 {"ticker", "ohlcv_rows", "news_rows", "fundamental_rows"}
        """
        self._ensure_kg()
        logging.info("Start collecting ticker=%s", ticker)

        market = MarketCollector(self.config)
        news = NewsCollector(self.config)
        global_news = GlobalNewsCollector(self.config)
        fundamental = FundamentalCollector(self.config)

        summary: Dict[str, Any] = {
            "ticker": ticker,
            "ohlcv_rows": 0,
            "news_rows": 0,
            "fundamental_rows": 0,
        }

        # ── OHLCV ──
        ohlcv_df = market.fetch_ohlcv(ticker)
        summary["ohlcv_rows"] = len(ohlcv_df)
        logging.info("OHLCV: %d rows", len(ohlcv_df))

        # ── News ──
        news_df = news.fetch_news(ticker, limit=self.config.news_limit_per_ticker)
        summary["news_rows"] = len(news_df)
        logging.info("News: %d rows", len(news_df))

        # ── Global macro news ──
        global_news_df = global_news.fetch_global_news(ticker=ticker, limit=50)
        summary["global_news_rows"] = len(global_news_df)
        logging.info("Global news: %d rows", len(global_news_df))

        # ── Fundamentals ──
        fundamentals_df = fundamental.fetch_fundamentals(ticker)
        summary["fundamental_rows"] = len(fundamentals_df)
        logging.info("Fundamentals: %d rows", len(fundamentals_df))

        # ── KG: Company entity ──
        self._kg_store.upsert_entities([
            Entity(entity_id=f"company:{ticker}", type="Company", properties={"ticker": ticker})
        ])

        # ── KG: Technical signals ──
        tech = TechnicalAgent()
        sig_ents, sig_rels = tech.build_signal_entities_from_ohlcv(ticker, ohlcv_df)
        self._kg_store.upsert_entities(sig_ents)
        self._kg_store.upsert_relations(sig_rels)
        logging.info("Technical KG updated: %d entities, %d relations", len(sig_ents), len(sig_rels))

        # ── KG: News ──
        n_ents, n_evs, n_rels = build_news_from_frame(ticker, news_df)
        self._kg_store.upsert_entities(n_ents)
        self._kg_store.upsert_evidences(n_evs)
        self._kg_store.upsert_relations(n_rels)
        logging.info("News KG updated: %d entities, %d evidences", len(n_ents), len(n_evs))

        # ── KG: Global macro news ──
        if not global_news_df.empty:
            gn_ents, gn_evs, gn_rels = build_global_news_from_frame(ticker, global_news_df)
            self._kg_store.upsert_entities(gn_ents)
            self._kg_store.upsert_evidences(gn_evs)
            self._kg_store.upsert_relations(gn_rels)
            logging.info("Global news KG updated: %d entities", len(gn_ents))

        # ── KG: Fundamentals ──
        f_ents, f_evs, f_rels = build_fundamentals_from_frame(ticker, fundamentals_df)
        self._kg_store.upsert_entities(f_ents)
        self._kg_store.upsert_evidences(f_evs)
        self._kg_store.upsert_relations(f_rels)
        logging.info("Fundamental KG updated: %d entities", len(f_ents))

        # ── KG: Risk events ──
        r_ents, r_evs, r_rels = build_risk_events_from_frame(ticker, ohlcv_df, fundamentals_df)
        self._kg_store.upsert_entities(r_ents)
        self._kg_store.upsert_evidences(r_evs)
        self._kg_store.upsert_relations(r_rels)
        logging.info("Risk KG updated: %d entities", len(r_ents))

        return summary

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
            ohlcv_path = ticker_dir / "ohlcv_2021_now.parquet"
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
        if self._orchestrator:
            self._orchestrator.close()
            self._orchestrator = None
        if self._kg_client:
            self._kg_client.close()
            self._kg_client = None
        if self._kg_store:
            self._kg_store.close()
            self._kg_store = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
