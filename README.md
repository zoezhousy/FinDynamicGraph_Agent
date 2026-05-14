# FinDynamicGraph Agent
Dissertation: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading

### Milestone 1 finished
#### SourceDocument → Evidence → Claim → AgentAssessment → DecisionTrace → BacktestOutcome

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
LLM_API_BASE=your_llm_api_base
LLM_API_KEY=your_key
LLM_MODEL=your_model_name
LLM_TIMEOUT_SECONDS=60
LLM_TEMPERATURE=0.2
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

- `data/raw/market_news/<ticker>/ohlcv_2021_now.parquet`
- `data/raw/market_news/<ticker>/news_latest.parquet`
- `data/experiments/trades.parquet`

## Notes

- For local single-instance Neo4j, prefer `bolt://127.0.0.1:7687`.
- If Neo4j is unavailable, collection can still save raw files, but graph writes are skipped.
<!-- - Yahoo Finance may rate-limit requests. Fallback chain is included: Yahoo chart API -> yfinance -> Stooq CSV. -->

## Current Scope of the MVP

### Deliverables: 
- raw collected data under data/raw/...
- Neo4j graph with queryable nodes and relationships
- experiment output file data/experiments/trades.parquet
- runnable collection and experiment logs

the system supports:

- 3 Hong Kong stock tickers
- OHLCV market data ingestion
- Tavily news collection
- technical signal generation
- basic graph updates
- basic graph querying
- initial multi-agent decision generation
- initial backtest execution


## Current Limitation 
1. KG still relatively thin and currently dominated by technical signal nodes. 
2. Evidence grounding mechanism is in inital level, but graph does not yet fully presented in all intended source -> claim -> decision chain
3. The current decision logic is still simple

