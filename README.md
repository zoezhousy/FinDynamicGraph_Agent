# FinDynamicGraph Agent v0.2.0
Dissertation: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading

## What is new in v0.2.0

- Neo4j relationship types are now written as real semantic edge types such as `HAS_SIGNAL` and `MENTIONED_IN` instead of a generic `REL` edge.
- KG query logic is aligned with semantic relationship types.
- Experiment logic is adjusted so the MVP can generate actual `buy` / `sell` actions instead of producing all-zero backtest output.
- Debug prints are added to `main_experiment.py` so you can inspect action distribution and trade execution.

## Current Neo4j-oriented MVP scope

- Unified KG schema for `Entity`, `Evidence`, `Relation`, `FinancialSignal`, `TradingDecision`
- Neo4j-backed graph store and query client
- Data collection pipeline for OHLCV + news
- Evidence-grounded graph updates for technical signals and news mentions
- Experiment pipeline for graph-based decisions vs baselines

## Linux quick start

```bash
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

## Outputs

- `data/raw/market_news/<ticker>/ohlcv_2021_2025.parquet`
- `data/raw/market_news/<ticker>/news_latest.parquet`
- `data/experiments/trades.parquet`

## Notes

- For local single-instance Neo4j, prefer `bolt://127.0.0.1:7687`.
- If Neo4j is unavailable, collection can still save raw files, but graph writes are skipped.
<!-- - Yahoo Finance may rate-limit requests. Fallback chain is included: Yahoo chart API -> yfinance -> Stooq CSV. -->
