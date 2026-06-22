# Experiment Protocol

Formal protocol for the dissertation experiment: *Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading*.

---

## 1. Systems Compared

| System | Description |
|---|---|
| `no_kg_no_evidence` | Baseline — no knowledge graph, no evidence grounding. Pure rule-based heuristics on raw price data. |
| `evidence_no_kg` | Evidence-grounded decisions without KG. Uses collected news and fundamental data but no graph structure or temporal queries. |
| `static_kg` | Static knowledge graph. KG built up to a fixed cutoff date (`2025-01-01`); no graph updates after cutoff. Demonstrates value of dynamic graph evolution. |
| `kg_dynamic` | Full system — dynamic KG with continuous evidence-grounded updates, multi-agent orchestration, and temporal graph queries. |

---

## 2. Tickers

All experiments run on three Hong Kong-listed equities:

| Ticker | Name |
|---|---|
| `0005.HK` | HSBC Holdings |
| `0700.HK` | Tencent Holdings |
| `1299.HK` | AIA Group |

---

## 3. Experiment Date Range

- **Start:** 2025-01-01
- **End:** 2026-04-30
- **Frequency:** Monthly (first calendar day of each month)
- **Total trade dates:** 16 per ticker (Jan 2025 – Apr 2026)
- **Total observations:** 48 per system × 4 systems = 192

---

## 4. Backtest Assumptions

| Parameter | Value | Notes |
|---|---|---|
| **Execution model** | Next-period | Enter at next trading day's open after signal date |
| **Holding period** | 5 trading days | Exit at close of day T+5 |
| **Transaction cost** | 5 basis points | Applied per side (total round-trip = 10 bp) |
| **Position sizing** | Full notional | 100% of capital per trade |
| **Actions** | `buy`, `sell`, `hold` | `hold` yields zero return with no execution |
| **Data source** | Yahoo Finance via yfinance | OHLCV daily bars from 2021-01-01 |

Implementation: `src/sim/backtest.py` → `BacktestConfig(holding_days=5, transaction_cost_bp=5.0)`

---

## 5. Static KG Cutoff

For the `static_kg` system, the knowledge graph is frozen at **2025-01-01**. No signals, news, or evidence generated after this date are ingested. This isolates the value of *dynamic* graph updates.

---

## 6. Reproducing the Experiment

### Prerequisites

- Python 3.12+
- Neo4j 5.x running locally (bolt://127.0.0.1:7687)
- `.env` configured with Neo4j credentials, LLM API, and Tavily API key

### Step 1: Data Collection

```bash
cd FinDynamicGraph_Agent
source .venv-linux/bin/activate   # or .venv\Scripts\activate on Windows
python -m src.main_collect
```

**Outputs:**
- `data/raw/market_news/<ticker>/ohlcv_2021_now.parquet`
- `data/raw/market_news/<ticker>/news_latest.parquet`
- `data/raw/market_news/<ticker>/global_news_latest.parquet`
- `data/raw/market_news/<ticker>/fundamentals_latest.parquet`

### Step 2: Run Experiment

```bash
python -m src.main_experiment
```

**Outputs:**
- `data/experiments/trades_latest.parquet`
- `data/experiments/trades_latest.csv`
- `data/experiments/trades_<timestamp>.parquet` (archived run)
- `data/experiments/trades_<timestamp>.csv` (archived run)

### Step 3: Export Graph Snapshot (optional)

```bash
# Single ticker + date
python -m src.scripts.export_graph_snapshot \
    --ticker 0700.HK --as-of 2025-03-01 \
    --output data/experiments/snapshot_0700_2025-03-01.json

# Diff two snapshots
python -m src.scripts.diff_snapshots \
    --a data/experiments/snapshot_0700_2025-01-01.json \
    --b data/experiments/snapshot_0700_2025-03-01.json
```

### Step 4: Export Decision Trace (optional)

```bash
# JSON only
python -m src.scripts.export_decision_trace \
    --decision-id decision:0700.HK:2025-03-01 \
    --output data/experiments/trace_0700_2025-03-01.json

# JSON + Markdown case study
python -m src.scripts.export_decision_trace \
    --decision-id decision:0700.HK:2025-03-01 \
    --output data/experiments/trace_0700_2025-03-01.json \
    --markdown-output data/experiments/trace_0700_2025-03-01.md
```

### Step 5: Verify Compilation

```bash
python -m compileall src
```

---

## 7. Expected Output Files

| Path | Format | Description |
|---|---|---|
| `data/experiments/trades_latest.csv` | CSV | Full experiment results (all systems, all tickers, all dates) |
| `data/experiments/trades_latest.parquet` | Parquet | Same as above, columnar format |
| `data/experiments/trades_<timestamp>.csv` | CSV | Timestamped archive of each run |
| `data/experiments/trades_<timestamp>.parquet` | Parquet | Timestamped archive |
| `data/experiments/snapshot_<ticker>_<date>.json` | JSON | Point-in-time KG subgraph export |
| `data/experiments/trace_<ticker>_<date>.json` | JSON | Decision trace with full provenance chain |
| `data/experiments/trace_<ticker>_<date>.md` | Markdown | Human-readable case study |

---

## 8. Known Limitations

1. **KG density:** The knowledge graph is currently dominated by technical signal nodes. News and fundamental nodes are sparser.
2. **Evidence grounding:** The SourceDocument → Evidence → Claim chain is partially implemented; some traces fall back to signal IDs rather than full provenance.
3. **Decision logic:** The multi-agent scoring is relatively simple (weighted average with conflict detection). More sophisticated negotiation/voting is out of scope for the MVP.
4. **LLM dependency:** Decision generation requires a working LLM API endpoint. Without it, the `kg_dynamic` system cannot produce decisions.
5. **Neo4j dependency:** Both collection and experiment require a running Neo4j instance. Collection can save raw files without Neo4j, but experiments cannot run.
6. **Data coverage:** Yahoo Finance may have gaps for HK stocks around holidays. Missing dates are handled gracefully (trade not executed).
7. **Single-market scope:** Only HK equities are tested. Generalization to other markets is not validated.
8. **Backtest realism:** No slippage modeling, no volume constraints, no market impact. The 5 bp cost is a simplified assumption.
