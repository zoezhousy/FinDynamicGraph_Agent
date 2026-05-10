# FinDynamicGraph Agent
Dissertation: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading

## Goal

This project builds a dynamic financial knowledge graph as shared memory for a multi-agent trading decision-support system.
Agents do not only exchange natural-language summaries; instead, they add, update, query, and verify graph evidence before generating buy / sell / hold / abstain decisions.

## Current Neo4j-oriented MVP scope

- Unified KG schema for `Entity`, `Evidence`, `Relation`, `FinancialSignal`, `TradingDecision`
- Neo4j-backed graph store and query client
- Data collection pipeline for OHLCV + news
- Evidence-grounded graph updates for technical signals and news mentions
- Experiment pipeline for graph-based decisions vs baselines

## Key files

- `src/kg/schema.py` - graph schema models
- `src/kg/store_neo4j.py` - Neo4j write layer
- `src/kg/query.py` - Neo4j read layer
- `src/kg/update_pipeline.py` - build graph batch from OHLCV/news
- `src/main_collect.py` - collection and KG write entrypoint
- `src/main_experiment.py` - experiment and backtest entrypoint

## Linux quick start

```bash
unzip Multi_Agent_Trading_with_Dynamic_KG.zip
cd Multi_Agent_Trading_with_Dynamic_KG
python3 -m venv .venv-linux
source .venv-linux/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in project root:

```env
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
TAVILY_API_KEY=your_key
```

Run collection:

```bash
python -m src.main_collect
```

Run experiment:

```bash
python -m src.main_experiment
```

Outputs:

- `data/raw/market_news/<ticker>/ohlcv_2021_2025.parquet`
- `data/raw/market_news/<ticker>/news_latest.parquet`
- `data/experiments/trades.parquet`

## Windows quick start

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.main_collect
python -m src.main_experiment
```

Use the same `.env` values as above.

## Notes

- For local single-instance Neo4j, prefer `bolt://127.0.0.1:7687` instead of `neo4j://127.0.0.1:7687`.
- `main_collect.py` degrades gracefully if Neo4j is unavailable, but then KG writes are skipped.
- Yahoo Finance may rate-limit requests. A fallback chain is included: Yahoo chart API -> yfinance -> Stooq CSV.
