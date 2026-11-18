# AGENTS.md

## Project overview

Dissertation project: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading. Python 3.11+ project using Neo4j, pandas, pydantic. No monorepo — single flat package under `src/`.

## Project Rules

Do not read or print the contents of:
- .env
- .env.local
- secrets/
- credentials/
- private_data/
- runtime files under data/ or CSV experiment outputs, unless the user explicitly asks to analyze them

Do not run destructive commands without asking:
- rm -rf
- DROP DATABASE
- DETACH DELETE
- git reset --hard
- git push --force

Use environment variables for credentials.
Never hard-code API keys, passwords, or tokens.

## Documentation

- README.md — setup, usage, architecture, metrics, and experiment protocol.
- PLAN.md — verified progress, remaining work, priorities, and dissertation-ready criteria.
- AGENTS.md — canonical operational and safety rules for coding agents.

Update these canonical documents instead of creating duplicate status or architecture files.
Preserve unrelated uncommitted changes.

## Commands

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux
pip install -r requirements.txt
```

Create `.env` in the project root using `.env.example` and README.md as references. Keep all credentials in environment variables.

### Run collection (OHLCV + news + fundamentals → Neo4j)

```bash
python -m src.main_collect
```

### Run experiment (multi-agent decisions + backtest → parquet)

```bash
python -m src.main_experiment
```

### Run focused temporal-safety experiment

```bash
python -m src.eval.run_preliminary --ticker 0700.HK
```

### CLI (analyze single ticker, cleanup old KG data)

```bash
python -m src.cli analyze 0700.HK
python -m src.cli analyze 0700.HK --date 2025-06-01 --backtest
python -m src.cli cleanup 0700.HK --days 90
```

### Tests

```bash
pytest
```

Default pytest is offline and blocks networking during collection and execution. Python 3.11 offline CI is configured in `.github/workflows/offline-tests.yml`; lint, typecheck and formatting are not configured. External tests require explicit `--integration -m integration`, environment credentials and a disposable test database. Do not enable integration checks as part of ordinary refactoring. See `PLAN.md` for current verified results.

## Architecture

Entry points:
- `src/main_collect.py` — data collection pipeline (OHLCV, news, global news, fundamentals, technical signals, risk events → Neo4j)
- `src/main_experiment.py` — runs 4 systems (kg_dynamic, no_kg_no_evidence, evidence_no_kg, static_kg) with backtest
- `src/cli.py` — CLI wrapper around `AnalysisPipeline` (analyze + cleanup subcommands)
- `src/pipeline.py` — `AnalysisPipeline` class: collect → KG → multi-agent decision → backtest → report

Key directories:
- `src/agents/` — `orchestrator.py` runs 4 LangGraph tool-calling agents (technical, news, fundamental, risk); `pm_deterministic.py` contains deterministic portfolio-manager aggregation; `models.py`, `legacy_roles.py`, and `evidence.py` separate reports, model calls, and evidence helpers; `roles.py` retains compatibility exports
- `src/collectors/` — data fetchers: `market_collector.py` (yfinance), `news_collector.py` (Tavily), `global_news_collector.py`, `fundamental_collector.py`
- `src/kg/` — knowledge graph layer: `schema.py` (Pydantic models for all node/relation types), `store_neo4j.py` (Neo4j CRUD), `query.py` (read-only queries), `update_pipeline.py` (frame → entities/relations)
- `src/decision/` — `graph_decision.py` (legacy simple signal-based engine; actual aggregation is in `src/agents/pm_deterministic.py`)
- `src/eval/` — baselines, trade/grounding metrics, significance tests, temporal leakage audit, and focused preliminary runner
- `src/llm/` — LLM client and prompts
- `src/sim/` — `backtest.py` (trade return computation)
- `src/scripts/` — inspection/export tools (snapshot, decision trace, diff)
- `data/raw/` — collected parquet files, gitignored

KG schema (Pydantic, in `src/kg/schema.py`):
- Entity types: Company, IndicatorSignal, NewsEvent, FundamentalSignal, RiskEvent, etc.
- Relation types: HAS_SIGNAL, MENTIONED_IN, SUPPORTED_BY, FOR_COMPANY, etc.
- Key models: `Entity`, `Relation`, `Evidence`, `Claim`, `FinancialSignal`, `DecisionTrace`, `AgentAssessment`, `BacktestOutcome`

Neo4j is optional — if unavailable, collection still saves parquet files but graph writes are skipped. Experiment pipeline requires Neo4j.

Default tickers: `0005.HK`, `0700.HK`, `1299.HK` (Hong Kong stocks).

## Gotchas

- Modules must be run with `-m` (e.g. `python -m src.main_collect`), not directly as scripts.
- `data/` directory is gitignored — collection must be run before experiments.
- The experiment loader supports both `ohlcv_2021_now.parquet` and the legacy `ohlcv_2021_2025.parquet` filename.
- `src/decision/graph_decision.py` is legacy — actual aggregation is in `src/agents/pm_deterministic.py`.
- `src/kg/graph_store.py` is legacy in-memory store — actual store is `Neo4jKGStore` in `store_neo4j.py`.
- See `PLAN.md` for verified progress, remaining work, and current test status.


## Refactoring and experiment contracts

- Preserve existing command names, import compatibility, four-system names, prompts, weights and thresholds unless a change is explicitly requested.
- Collection orchestration/cache live in `src/collectors/service.py` and `cache.py`; batch graph writes/connections live in `src/kg/ingestion.py` and `connections.py`.
- Full experiment execution, backtest persistence and export live in `src/eval/runner.py`, `outcomes.py` and `reporting.py`. Keep preliminary result columns compatible.
- Internally created resources are owned and closed by their creator, including initialization failure. Never close injected clients; make close idempotent.
- `ExperimentConfig.protocol_version` defaults to `legacy`. Changes affecting temporal inputs or comparison eligibility belong behind explicit `pit_v1` configuration.
- New-protocol tools, aggregation and audit must share one frozen ticker/cutoff snapshot. Reject out-of-scope tool requests and unknown availability; never quietly replace a failed source.
- Preserve task diagnostics independently of neutral/abstain decisions. Paired new-protocol metrics require successful, settled rows and matched ticker/date keys; report sample counts.
- Run records whitelist non-sensitive configuration and selected model messages. Never serialize client objects, request headers, credential-bearing URLs, or entire environment dictionaries.
- Use synthetic inputs and temporary files for verification. Offline replay is `python -m src.eval.replay <run-dir> --output <csv>`; do not read real run directories without user authorization.
- Resume only against matching observed input/configuration hashes. Keep prior failed/degraded attempt evidence, and isolate new protocol reports by run ID.
- Offline passing tests prove code behavior only. Do not infer service integration, verified historical availability, or dissertation completion.

- Evaluation revisions are explicit: `ExperimentConfig.evaluation_version` defaults to `legacy`; `v2` requires `pit_v1`. Record the version in run identity and task inputs, replay missing-version records as legacy, and never mix versions or rewrite historical reports. The focused preliminary schema remains unchanged.


## Independent core research contracts

- New core innovation runs use `src.eval.run_research` and `research_v1`; never relabel historical four-system results as controlled ablations or semantic validation.
- Construction and independent verification precede a frozen decision snapshot. Models cannot execute Cypher. Commit only exact-quote-supported, in-scope, time-eligible deltas; preserve same-metric/period lifecycle, prior support, conflicts, explicit supersession and expiry history.
- The research ledger is run-owned. Optional Neo4j writes use isolated Research labels, uniqueness constraints, one transaction and state-digest conflict checks. Do not merge these into production ingestion implicitly.
- Graph/text, grounding enforcement and integrated/multiple analyst comparisons share evidence hashes, PM rules and total response-token caps. Report prompt length/call-count differences; representation and static freshness ablations have different causal scopes.
- Semantic scores require validated judge outputs with exact quotes and versioned caches. Failed judgments are failed tasks. Unannotated key-evidence coverage stays missing; human agreement needs real supplied human labels.
- Record per-variant execution/settlement counts, failed graph/model attempts and conditional risk behavior. Synthetic repetitions are timing/policy checks, not independent market observations or proof of model understanding.
- The research CLI reads credentials only from process environment variables and does not load .env. Saved-run adapters require user authorization for those runtime inputs; refuse missing or mismatched snapshots, do not substitute another source.
- Preserve historical original files when reviewing. Write case/result artifacts outside the original run and retain selection criteria, input hashes, same-date comparisons and source-coverage limitations.
