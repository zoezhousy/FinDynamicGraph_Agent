# FinDynamicGraph Agent

Dissertation: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading

## Project Overview

This project implements a dynamic knowledge graph (KG) system for financial decision-making.
The core contribution is **temporal graph management**: the same ticker produces different
graph states at different `as_of_date` values, enabling backtest experiments without
look-ahead bias.

### Four-System Comparison

| System | Description |
|---|---|
| `no_kg_no_evidence` | Pure LLM with no data or KG |
| `evidence_no_kg` | LLM with raw OHLCV + news text, no KG |
| `static_kg` | Full multi-agent KG pipeline with frozen graph (pre-experiment cutoff) |
| `kg_dynamic` | Proposed system: temporal KG with dynamic updates |

---

## Environment Setup

### Prerequisites

- Python 3.11+
- Neo4j 5.x (local or remote)
- LLM API endpoint (OpenAI-compatible)

### Installation

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables

Create `.env` in project root:

```env
# Neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# News collection (Tavily)
TAVILY_API_KEY=your_key
NEWSAPI_API_KEY=your_api_key

# LLM (OpenAI-compatible endpoint)
LLM_API_BASE=your_llm_api_base
LLM_API_KEY=your_key
LLM_MODEL=your_model_name
LLM_TIMEOUT_SECONDS=60
LLM_TEMPERATURE=0.2
```

---

## Dissertation Experiment Usage

### 1. Data Collection

```bash
python -m src.main_collect
```

Collects OHLCV market data, news (Tavily), global news, and fundamentals for
configured tickers. Writes raw parquet files to `data/raw/market_news/<ticker>/`.
Ingests data into Neo4j if configured.

### 2. Run Experiment

```bash
python -m src.main_experiment
```

Runs all four systems across all tickers and trade dates. Outputs:
- `data/experiments/trades_latest.csv` — full trade-level results
- `data/experiments/trades_latest.parquet` — same, columnar format
- `data/experiments/results_summary_latest.csv` — aggregated metrics by system
- `data/experiments/case_study_candidates.csv` — selected cases for dissertation
- `data/experiments/trades_<timestamp>.csv` — timestamped archive

### 3. Inspect KG Snapshot

```bash
python -m src.scripts.inspect_snapshot --ticker 0700.HK --date 2024-06-01
python -m src.scripts.inspect_snapshot --ticker 0700.HK --date 2025-06-01
```

Shows active signals, claims, evidence, conflicts, and freshness at a given date.
Add `--save` to write JSON to `data/experiments/`.

### 4. Inspect Decision Trace

```bash
# By ticker + date (searches trades_latest first, then Neo4j)
python -m src.scripts.inspect_decision_trace --ticker 0700.HK --date 2025-06-01

# By decision ID (from Neo4j)
python -m src.scripts.inspect_decision_trace --decision-id decision:0700.HK:2025-06-01
```

Shows action, scores, evidence, claims, agent assessments, and backtest outcome.

### 5. Export Decision Trace (JSON + Markdown)

```bash
python -m src.scripts.export_decision_trace \
    --decision-id decision:0700.HK:2025-03-01 \
    --output data/experiments/trace_0700_2025-03-01.json \
    --markdown-output data/experiments/trace_0700_2025-03-01.md
```

Exports full provenance chain: SourceDocument → Evidence → Claim → AgentAssessment → DecisionTrace → BacktestOutcome.

### 6. Temporal Divergence Proof

```bash
# Export snapshots at two dates
python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK --as-of 2025-01-01 \
    --output data/experiments/snapshot_0700_2025-01-01.json

python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK --as-of 2025-03-01 \
    --output data/experiments/snapshot_0700_2025-03-01.json

# Diff the snapshots
python -m src.scripts.diff_snapshots \
    --a data/experiments/snapshot_0700_2025-01-01.json \
    --b data/experiments/snapshot_0700_2025-03-01.json
```

### 7. Run Tests

```bash
# All tests
pytest

# Specific test files
pytest tests/test_grounding_metrics.py
pytest tests/test_experiment_outputs.py
pytest tests/test_decision_trace.py
pytest tests/test_dynamic_query.py
pytest tests/test_evidence_chain.py
```

---

## Output Files

| Path | Format | Description |
|---|---|---|
| `data/experiments/trades_latest.csv` | CSV | Full experiment results (all systems, all dates) |
| `data/experiments/trades_latest.parquet` | Parquet | Same, columnar format |
| `data/experiments/results_summary_latest.csv` | CSV | Aggregated metrics by system |
| `data/experiments/case_study_candidates.csv` | CSV | Selected cases for dissertation analysis |
| `data/experiments/snapshot_<ticker>_<date>.json` | JSON | KG subgraph at a point in time |
| `data/experiments/trace_<ticker>_<date>.json` | JSON | Decision trace with full provenance |
| `data/experiments/trace_<ticker>_<date>.md` | Markdown | Human-readable case study |
| `data/raw/market_news/<ticker>/ohlcv_2021_now.parquet` | Parquet | Raw OHLCV data |
| `data/raw/market_news/<ticker>/news_latest.parquet` | Parquet | Raw news data |

---

## Metrics

### Trade Metrics

| Metric | Description |
|---|---|
| `n_decisions` | Total decisions per system |
| `n_trades` | Executed trades (buy/sell) |
| `trade_execution_rate` | n_trades / n_decisions |
| `win_rate` | Fraction of profitable trades |
| `directional_accuracy` | Fraction with positive return |
| `abstain_rate` | Fraction of abstain decisions |
| `mean_return` / `median_return` | Return statistics |
| `mean_confidence` | Average decision confidence |
| `mean_conflict_level` | Average inter-agent conflict |

### Grounding Metrics

| Metric | Description |
|---|---|
| `evidence_coverage` | Decisions with ≥1 evidence ref / total |
| `citation_precision` | Valid evidence refs / all refs |
| `claim_coverage` | Decisions with ≥1 claim ref / total |
| `unsupported_claim_rate` | Decisions with reason but no claims / total with reason |
| `stale_evidence_rate` | Stale evidence / total evidence |

---

## Current Limitations

1. **KG density:** Technical signal nodes dominate; news/fundamental nodes are sparser.
2. **Grounding metrics:** Use conservative approximations (presence-based, not semantic matching).
3. **LLM dependency:** `kg_dynamic`, `evidence_no_kg`, and `no_kg_no_evidence` require a working LLM endpoint.
4. **Neo4j dependency:** Experiment pipeline requires a running Neo4j instance.
5. **Data coverage:** Limited to 3 HK stock tickers. Yahoo Finance may have gaps around HK holidays.
6. **Backtest realism:** No slippage, volume constraints, or market impact. Transaction cost is 5 bp per side.
7. **Case study selection:** Heuristic-based; does not use outcome information for all criteria.

---

## Architecture

```
src/
├── agents/           # Multi-agent system (news, technical, fundamental, risk, orchestrator)
├── collectors/       # Data collection (OHLCV, news, fundamentals, global news)
├── eval/             # Evaluation (metrics, baselines, grounding)
├── kg/               # Knowledge graph (schema, update_pipeline, query, store_neo4j)
├── llm/              # LLM client and prompts
├── scripts/          # Inspection and export scripts
├── sim/              # Backtest simulation
├── utils/            # Retry, rate limiting, I/O
├── main_collect.py   # Data collection entry point
└── main_experiment.py # Experiment entry point
```

---

## Reproducibility

Full experiment protocol: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)
