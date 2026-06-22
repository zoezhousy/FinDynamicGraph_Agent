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


## Temporal Divergence Proof

Prove that the same ticker has different graph states at different `as_of_date` values.

### Step 1: Export snapshots at two dates

```bash
# Snapshot at date A
python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK --as-of 2025-01-01 \
    --output data/experiments/snapshot_0700_2025-01-01.json

# Snapshot at date B
python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK --as-of 2025-03-01 \
    --output data/experiments/snapshot_0700_2025-03-01.json
```

### Step 2: Diff the snapshots

```bash
python -m src.scripts.diff_snapshots \
    --a data/experiments/snapshot_0700_2025-01-01.json \
    --b data/experiments/snapshot_0700_2025-03-01.json
```

Output shows:
- Node count changes (signals, news, fundamentals, risks, evidences, claims)
- Added/removed signals, news, evidence, and claims between the two dates
- Final verdict: IDENTICAL or DIFFERENT

If DIFFERENT → temporal divergence is proven: the KG evolves over time for the same ticker.

---

## Reproducibility Workflow

Full experiment protocol: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)

### 1. Data Collection

```bash
python -m src.main_collect
```

Collects OHLCV market data, news (Tavily), global news, and fundamentals for all three tickers. Writes raw parquet files to `data/raw/market_news/<ticker>/`. Also ingests data into the Neo4j knowledge graph if configured.

### 2. Run Experiment

```bash
python -m src.main_experiment
```

Runs four systems (`no_kg_no_evidence`, `evidence_no_kg`, `static_kg`, `kg_dynamic`) across all tickers and trade dates. Outputs:
- `data/experiments/trades_latest.csv`
- `data/experiments/trades_latest.parquet`
- `data/experiments/trades_<timestamp>.csv` (archive)
- `data/experiments/trades_<timestamp>.parquet` (archive)

### 3. Export Graph Snapshot

```bash
python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK --as-of 2025-03-01 \
    --output data/experiments/snapshot_0700_2025-03-01.json
```

Exports a point-in-time KG subgraph as JSON. Diff two snapshots to prove temporal divergence:

```bash
python -m src.scripts.diff_snapshots \
    --a data/experiments/snapshot_0700_2025-01-01.json \
    --b data/experiments/snapshot_0700_2025-03-01.json
```

### 4. Export Decision Trace

```bash
# JSON + Markdown case study
python -m src.scripts.export_decision_trace \
    --decision-id decision:0700.HK:2025-03-01 \
    --output data/experiments/trace_0700_2025-03-01.json \
    --markdown-output data/experiments/trace_0700_2025-03-01.md
```

Exports the full provenance chain: SourceDocument → Evidence → Claim → AgentAssessment → DecisionTrace → BacktestOutcome.

### 5. Expected Output Files

| Path | Format | Description |
|---|---|---|
| `data/experiments/trades_latest.csv` | CSV | Full experiment results |
| `data/experiments/trades_latest.parquet` | Parquet | Same, columnar format |
| `data/experiments/snapshot_<ticker>_<date>.json` | JSON | KG subgraph snapshot |
| `data/experiments/trace_<ticker>_<date>.json` | JSON | Decision trace |
| `data/experiments/trace_<ticker>_<date>.md` | Markdown | Human-readable case study |

### 6. Known Limitations

1. **KG density:** Currently dominated by technical signal nodes; news/fundamental nodes are sparser.
2. **Evidence grounding:** SourceDocument → Evidence → Claim chain is partially implemented; some traces fall back to signal IDs.
3. **Decision logic:** Simple weighted-average scoring with conflict detection; no sophisticated agent negotiation.
4. **LLM dependency:** `kg_dynamic` system requires a working LLM API endpoint.
5. **Neo4j dependency:** Experiments require a running Neo4j instance.
6. **Data coverage:** Yahoo Finance may have gaps around HK holidays; missing dates are handled gracefully.
7. **Backtest realism:** No slippage, volume constraints, or market impact modeling. Transaction cost is a flat 5 bp per side.

