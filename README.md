# FinDynamicGraph Agent
Dissertation: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading

### Milestone 2 partially implemented (See plan in Plan_in_Chinese.md)
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

<!-- 
    if need to choose the version of Python
    Command is
    ```cmd
    py -3.11 -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
 -->

## Outputs

- `data/raw/market_news/<ticker>/ohlcv_2021_now.parquet`
- `data/raw/market_news/<ticker>/news_latest.parquet`
- `data/experiments/trades.parquet`

## Export KG Graph Snapshot

Export the knowledge graph state for a given ticker at a specific date:

```bash
python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK \
    --as-of 2025-03-01 \
    --output data/experiments/snapshot_0700_2025-03-01.json
```

The output JSON contains:
- `ticker`, `as_of_date`
- `n_signals`, `n_news`, `n_fundamentals`, `n_risks`, `n_evidences`, `n_claims`
- `top_signals` (ranked by strength)
- `top_news` (most recent)
- `evidence_refs` (all unique evidence_ids)
- `fundamentals_summary`, `risks_summary`

Compare snapshots across different dates to demonstrate that the same ticker
has different graph states at different points in time:

```bash
python -m src.scripts.export_graph_snapshot --ticker 0700.HK --as-of 2025-01-01 --output data/experiments/snapshot_0700_jan.json
python -m src.scripts.export_graph_snapshot --ticker 0700.HK --as-of 2025-06-01 --output data/experiments/snapshot_0700_jun.json
```

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

