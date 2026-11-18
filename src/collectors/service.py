"""Collection orchestration. Analysis and batch retain distinct legacy profiles."""
import logging
from typing import Any, Dict
import pandas as pd
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

class AnalysisCollectionService:
    def __init__(self, config, store, *, collectors=None, technical_agent=None):
        self.config = config
        self.store = store
        self.collectors = collectors or {}
        self.technical_agent = technical_agent

    def run(self, ticker):
        logging.info("Start collecting ticker=%s", ticker)

        market = self.collectors.get("market") or MarketCollector(self.config)
        news = self.collectors.get("news") or NewsCollector(self.config)
        global_news = self.collectors.get("global_news") or GlobalNewsCollector(self.config)
        fundamental = self.collectors.get("fundamental") or FundamentalCollector(self.config)

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

        # ── News: Tavily → NewsAPI → dedup ──
        news_frames: list[pd.DataFrame] = []
        news_df = news.fetch_news(ticker, limit=self.config.news_limit_per_ticker)
        if not news_df.empty:
            news_frames.append(news_df)
        try:
            newsapi = self.collectors.get("newsapi") or NewsAPICollector(self.config)
            newsapi_df = newsapi.fetch_news(ticker, limit=self.config.news_limit_per_ticker)
            if not newsapi_df.empty:
                news_frames.append(newsapi_df)
        except ValueError:
            logging.info("NewsAPI key not set, skipping")
        except Exception as exc:
            logging.warning("NewsAPI fetch failed for %s: %s", ticker, exc)

        if len(news_frames) > 1:
            news_df = dedup_news_frames(news_frames, sort_by_time=False)
        elif news_frames:
            news_df = news_frames[0]

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
        self.store.upsert_entities([
            Entity(entity_id=f"company:{ticker}", type="Company", properties={"ticker": ticker})
        ])

        # ── KG: Technical signals ──
        tech = self.technical_agent or TechnicalAgent()
        sig_ents, sig_rels = tech.build_signal_entities_from_ohlcv(ticker, ohlcv_df)
        write_graph_batch(self.store, sig_ents, None, sig_rels)
        logging.info("Technical KG updated: %d entities, %d relations", len(sig_ents), len(sig_rels))

        # ── KG: News ──
        n_ents, n_evs, n_rels = build_news_from_frame(ticker, news_df)
        write_graph_batch(self.store, n_ents, n_evs, n_rels)
        logging.info("News KG updated: %d entities, %d evidences", len(n_ents), len(n_evs))

        # ── KG: Global macro news ──
        if not global_news_df.empty:
            gn_ents, gn_evs, gn_rels = build_global_news_from_frame(ticker, global_news_df)
            write_graph_batch(self.store, gn_ents, gn_evs, gn_rels)
            logging.info("Global news KG updated: %d entities", len(gn_ents))

        # ── KG: Fundamentals ──
        f_ents, f_evs, f_rels = build_fundamentals_from_frame(ticker, fundamentals_df)
        write_graph_batch(self.store, f_ents, f_evs, f_rels)
        logging.info("Fundamental KG updated: %d entities", len(f_ents))

        # ── KG: Risk events ──
        r_ents, r_evs, r_rels = build_risk_events_from_frame(ticker, ohlcv_df, fundamentals_df)
        write_graph_batch(self.store, r_ents, r_evs, r_rels)
        logging.info("Risk KG updated: %d entities", len(r_ents))

        return summary



from concurrent.futures import ThreadPoolExecutor, as_completed
from src.collectors.cache import check_existing_data_dates as _check_existing_data_dates, dedup_news_frames as _dedup_news_frames, merge_fundamentals_history as _merge_fundamentals_history, fetch_ohlcv_with_cache as _fetch_ohlcv_with_cache
from src.kg.ingestion import company_entity as build_company_entity
from src.utils.io import ensure_dir, save_parquet

def _init_kg_store(config):
    return initialized_store(Neo4jSettings.from_env(config.neo4j_database, required=False), Neo4jKGStore, optional=True)

def run_collection(config: CollectionConfig, *, collectors=None, technical_agent=None, store_factory=None) -> None:
    collectors = collectors or {}
    market_collector = collectors.get("market") or MarketCollector(config)
    news_collector = collectors.get("news") or NewsCollector(config)
    global_news_collector = collectors.get("global_news") or GlobalNewsCollector(config)
    newsapi_collector = collectors.get("newsapi") or NewsAPICollector(config)
    fundamental_collector = collectors.get("fundamental") or FundamentalCollector(config)
    technical_agent = technical_agent or TechnicalAgent()
    kg_store = (store_factory or _init_kg_store)(config)

    try:
        # Fetch OHLCV for all tickers in parallel (heaviest I/O)
        with ThreadPoolExecutor(max_workers=len(config.tickers)) as pool:
            ohlcv_futures = {
                pool.submit(
                    _fetch_ohlcv_with_cache, t, market_collector, config.output_root / t
                ): t
                for t in config.tickers
            }
            ohlcv_cache: dict[str, pd.DataFrame | None] = {}
            for fut in as_completed(ohlcv_futures):
                t = ohlcv_futures[fut]
                try:
                    ohlcv_cache[t] = fut.result()
                except Exception as exc:
                    logging.error("OHLCV parallel fetch failed for %s: %s", t, exc)
                    ohlcv_cache[t] = None

        # Process each ticker sequentially (news + KG updates)
        for ticker in config.tickers:
            ticker_dir = config.output_root / ticker
            ensure_dir(ticker_dir)
            logging.info("Start collecting ticker=%s", ticker)

            # Check existing data dates
            existing_dates = _check_existing_data_dates(ticker_dir, ticker)
            if existing_dates:
                logging.info("Existing data for %s: %s", ticker, existing_dates)
                print(f"[{ticker}] Existing data: {existing_dates}")

            ohlcv_df = ohlcv_cache.get(ticker)
            if ohlcv_df is not None:
                save_parquet(ohlcv_df, ticker_dir / "ohlcv_2021_now.parquet")

            news_df = None
            global_news_df = None
            fundamentals_df = None

            # ── News: fetch from both sources, then dedup ──
            news_frames: list[pd.DataFrame] = []
            newsapi_df = None
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(news_collector.fetch_news, ticker, config.news_limit_per_ticker): "tavily",
                    pool.submit(newsapi_collector.fetch_news, ticker, config.news_limit_per_ticker): "newsapi",
                }
                for fut in as_completed(futures):
                    source = futures[fut]
                    try:
                        df = fut.result()
                        if df is not None and not df.empty:
                            news_frames.append(df)
                            if source == "newsapi":
                                newsapi_df = df
                                save_parquet(df, ticker_dir / "newsapi_latest.parquet")
                            else:
                                save_parquet(df, ticker_dir / "news_latest.parquet")
                    except Exception as exc:
                        logging.exception("News (%s) failed for %s: %s", source, ticker, exc)

            news_df = _dedup_news_frames(news_frames)
            if not news_df.empty:
                save_parquet(news_df, ticker_dir / "news_combined_latest.parquet")
                logging.info("Combined deduped news for %s: %d articles", ticker, len(news_df))

            # ── Global news ──
            try:
                global_news_df = global_news_collector.fetch_global_news(ticker=ticker, limit=50)
                if global_news_df is not None and not global_news_df.empty:
                    save_parquet(global_news_df, ticker_dir / "global_news_latest.parquet")
            except Exception as exc:
                logging.exception("Global news collection failed for %s: %s", ticker, exc)

            # ── Fundamentals: append to historical file ──
            if config.collect_fundamentals:
                try:
                    fund_df = fundamental_collector.fetch_fundamentals(ticker)
                    history_path = ticker_dir / "fundamentals_history.parquet"
                    fundamentals_df = _merge_fundamentals_history(fund_df, history_path)
                    save_parquet(fundamentals_df, history_path)
                    save_parquet(fund_df, ticker_dir / "fundamentals_latest.parquet")
                except Exception as exc:
                    logging.warning("Fundamental fetch failed for %s, trying cached: %s", ticker, exc)
                    cached_fund = ticker_dir / "fundamentals_latest.parquet"
                    if cached_fund.exists():
                        try:
                            fundamentals_df = pd.read_parquet(cached_fund)
                            logging.info("Loaded cached fundamentals for %s: %d rows", ticker, len(fundamentals_df))
                        except Exception as exc2:
                            logging.error("Cached fundamentals also failed for %s: %s", ticker, exc2)
                    else:
                        logging.error("No cached fundamentals for %s", ticker)

            if kg_store and config.reset_graph_before_collection:
                try:
                    kg_store.clear_generated_data_for_ticker(
                        ticker,
                        preserve_historical_traces=config.preserve_historical_traces,
                    )
                    logging.info(
                        "Reset generated graph data for ticker=%s; preserve_historical_traces=%s",
                        ticker,
                        config.preserve_historical_traces,
                    )
                except Exception as exc:
                    logging.exception("Failed resetting graph data for %s: %s", ticker, exc)

            if kg_store:
                try:
                    kg_store.upsert_entities([build_company_entity(ticker)])
                except Exception as exc:
                    logging.exception("Company KG upsert failed for %s: %s", ticker, exc)

            # update Techinical KG
            if kg_store and ohlcv_df is not None:
                try:
                    signal_entities, signal_relations = technical_agent.build_signal_entities_from_ohlcv(
                        ticker,
                        ohlcv_df,
                    )

                    write_graph_batch(kg_store, signal_entities, None, signal_relations)
                    logging.info("Technical graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Technical KG update failed for %s: %s", ticker, exc)

            # Update News KG
            if kg_store and news_df is not None:
                try:
                    news_entities, evidences, news_relations = build_news_from_frame(ticker, news_df)
                    write_graph_batch(kg_store, news_entities, evidences, news_relations)
                    logging.info("News graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("News KG update failed for %s: %s", ticker, exc)

            if kg_store and newsapi_df is not None and not newsapi_df.empty:
                try:
                    na_entities, na_evidences, na_relations = build_news_from_frame(ticker, newsapi_df)
                    write_graph_batch(kg_store, na_entities, na_evidences, na_relations)
                    logging.info("NewsAPI graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("NewsAPI KG update failed for %s: %s", ticker, exc)

            if kg_store and global_news_df is not None and not global_news_df.empty:
                try:
                    gn_entities, gn_evidences, gn_relations = build_global_news_from_frame(ticker, global_news_df)
                    write_graph_batch(kg_store, gn_entities, gn_evidences, gn_relations)
                    logging.info("Global news graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Global news KG update failed for %s: %s", ticker, exc)

            # Update Fundamental KG 
            if kg_store and fundamentals_df is not None:
                try:
                    fundamental_entities, fundamental_evidences, fundamental_relations = (
                        build_fundamentals_from_frame(ticker, fundamentals_df)
                    )
                    write_graph_batch(kg_store, fundamental_entities, fundamental_evidences, fundamental_relations)
                    logging.info("Fundamental graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Fundamental KG update failed for %s: %s", ticker, exc)
            
            # Update risk events in KG based on OHLCV and fundamentals.
            if kg_store and ohlcv_df is not None:
                try:
                    risk_entities, risk_evidences, risk_relations = build_risk_events_from_frame(
                        ticker=ticker,
                        ohlcv=ohlcv_df,
                        fundamentals=fundamentals_df,
                    )
                    write_graph_batch(kg_store, risk_entities, risk_evidences, risk_relations)
                    logging.info("Risk graph updated for ticker=%s", ticker)
                except Exception as exc:
                    logging.exception("Risk KG update failed for %s: %s", ticker, exc)

            logging.info("Done ticker=%s", ticker)
    finally:
        if kg_store is not None:
            kg_store.close()


