# Plan: Convert Analyst Agents to LangGraph Tool-Calling Agents

## Research Definition

```
LangGraph tool-calling analyst agents
+ dynamic/static temporal KG tool policy
+ deterministic evidence-aware portfolio aggregation
```

核心研究假设：改进来自 dynamic evidence-grounded KG，而不是来自更复杂的 agent framework。因此 `kg_dynamic` 和 `static_kg` 必须使用相同的 agent 架构，仅在 KG cutoff policy 上不同。

## Dependencies

```bash
pip install langchain==1.3.14 langgraph==1.2.9 langchain-openai==1.3.5
```

安装后 `pip freeze > requirements.txt` 固定完整环境。

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Orchestrator (async)         │
                    │   AGENT_BACKEND=langgraph|legacy     │
                    └─────────┬───────────────────────────┘
                              │ asyncio.gather + per-agent timeout
           ┌──────────┬───────┴────────┬──────────┐
           ▼          ▼                ▼          ▼
    ┌──────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐
    │  News    │ │Technical │  │Fundament.│ │  Risk    │
    │  Agent   │ │  Agent   │  │  Agent   │ │  Agent   │
    │(LG agent)│ │(LG agent)│  │(LG agent)│ │(LG agent)│
    └──────────┘ └──────────┘  └──────────┘ └──────────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
    AgentReport   AgentReport  AgentReport   AgentReport
                              │
                    ┌─────────┴───────────┐
                    │ Deterministic PM    │
                    │ (weighted agg,      │
                    │  same as current)   │
                    └─────────────────────┘
                              │
                              ▼
                       Decision Dict
```

Key differences from original plan:
- PM stays deterministic (phase 1); agentic PM is optional ablation (phase 2)
- `static_kg` reuses same agents with `FixedCutoffPolicy`
- Tools use single-query + request-scoped cache
- Async with per-agent timeout + semaphore

## Step 0: API Capability Smoke Test

Before any code changes, verify the OpenAI-compatible endpoint supports LangGraph requirements:

```python
# test_langgraph_smoke.py
# 1. ChatOpenAI basic call
# 2. Single @tool invocation
# 3. ToolStrategy(PydanticSchema) structured output
# 4. Four concurrent ainvoke calls
```

All 4 must pass before proceeding.

## Step 1: Fix Existing AgentReport Bug

`FundamentalAgent._run_from_kg_fundamentals()` is missing `claim_refs`:

```python
# src/agents/fundamental_agent.py
# Add claim_refs collection and pass to AgentReport
```

## Step 2: New file — `src/agents/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Literal

class FactorSchema(BaseModel):
    name: str
    direction: Literal["bullish", "bearish", "neutral", "uncertain"]
    weight: float = Field(ge=0.0, le=1.0)
    value: str | None = None

class AgentReportSchema(BaseModel):
    """Structured output from an analyst agent. 'role' is injected by the caller."""
    stance: Literal["bullish", "bearish", "neutral", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(-1.0, le=1.0)
    summary: str
    evidence_refs: list[str]
    claim_refs: list[str]
    factors: list[FactorSchema]

class DecisionSchema(BaseModel):
    """Structured output from the portfolio manager agent."""
    action: Literal["buy", "sell", "hold", "abstain"]
    final_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    conflict_level: float = Field(ge=0.0, le=1.0)
    decision_reason: str
    supporting_roles: list[str]
    opposing_roles: list[str]
```

## Step 3: Refactor `src/agents/kg_tools.py` — KG Tools with Cache + Date Policy

```python
from dataclasses import dataclass
from langchain_core.tools import tool

@dataclass
class DynamicDatePolicy:
    """Use trade_date as-is (current behavior)."""
    def resolve(self, as_of_date: str) -> str:
        return as_of_date

@dataclass
class FixedCutoffPolicy:
    """Always query with a fixed cutoff date (for static_kg baseline)."""
    cutoff: str  # e.g. "2025-01-01"
    def resolve(self, as_of_date: str) -> str:
        return self.cutoff

def build_kg_tools(query_client, date_policy):
    """Build tool set with request-scoped cache.

    Each ticker/date combination executes at most one full KG query.
    Tools filter the cached snapshot by category.
    """
    _cache: dict[tuple[str, str], dict] = {}

    def _get_snapshot(ticker: str, as_of_date: str) -> dict:
        effective = date_policy.resolve(as_of_date)
        key = (ticker, effective)
        if key not in _cache:
            _cache[key] = query_client.get_ticker_subgraph(ticker, effective)
        return _cache[key]

    @tool
    def query_subgraph(ticker: str, as_of_date: str) -> dict:
        """Load the full temporal KG subgraph for a ticker."""
        return _get_snapshot(ticker, as_of_date)

    @tool
    def query_signals(ticker: str, as_of_date: str) -> list[dict]:
        """Get technical indicator signals from the KG."""
        return _get_snapshot(ticker, as_of_date).get("signals", [])

    @tool
    def query_fundamentals(ticker: str, as_of_date: str) -> list[dict]:
        """Get fundamental metrics from the KG."""
        return _get_snapshot(ticker, as_of_date).get("fundamentals", [])

    @tool
    def query_risks(ticker: str, as_of_date: str) -> list[dict]:
        """Get risk events from the KG."""
        return _get_snapshot(ticker, as_of_date).get("risks", [])

    @tool
    def query_news(ticker: str, as_of_date: str) -> list[dict]:
        """Get news events from the KG."""
        return _get_snapshot(ticker, as_of_date).get("news", [])

    @tool
    def query_claims(ticker: str, as_of_date: str) -> dict:
        """Get claims, evidence chains, and source documents from the KG."""
        snap = _get_snapshot(ticker, as_of_date)
        return {
            "claims": snap.get("claims", []),
            "evidences": snap.get("evidences", []),
            "sources": snap.get("sources", []),
            "conflicts": snap.get("conflicts", []),
        }

    all_tools = [query_subgraph, query_signals, query_fundamentals,
                 query_risks, query_news, query_claims]
    return all_tools
```

Cache is per `build_kg_tools()` call (i.e. per `run_for_ticker_async` invocation), not global. Different systems (dynamic vs static) get separate caches because they use different `date_policy` instances.

## Step 4: New file — `src/agents/langgraph_agents.py`

Uses `langchain.agents.create_agent` with `ToolStrategy`:

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI

def _make_llm():
    return ChatOpenAI(
        base_url=os.getenv("LLM_API_BASE"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL"),
        temperature=0.2,
        timeout=60,
    )

def create_news_agent(llm, tools):
    subset = [t for t in tools if t.name in ("query_subgraph", "query_news", "query_claims")]
    return create_agent(
        model=llm,
        tools=subset,
        system_prompt=NEWS_SYSTEM_PROMPT,
        response_format=ToolStrategy(AgentReportSchema),
    )

def create_technical_agent(llm, tools):
    subset = [t for t in tools if t.name in ("query_signals", "query_subgraph")]
    return create_agent(
        model=llm,
        tools=subset,
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
        response_format=ToolStrategy(AgentReportSchema),
    )

# ... similar for fundamental, risk
```

**Tool assignment per agent:**
| Agent | Tools | Reason |
|---|---|---|
| News | `query_news`, `query_claims`, `query_subgraph` | Needs news + evidence chains |
| Technical | `query_signals`, `query_subgraph` | Needs indicator signals |
| Fundamental | `query_fundamentals`, `query_claims`, `query_subgraph` | Needs fundamental metrics + evidence |
| Risk | `query_risks`, `query_claims`, `query_subgraph` | Needs risk events + evidence |

System prompts instruct each agent to:
1. Call tools to gather data
2. Analyze ONLY based on retrieved data
3. Cite specific evidence IDs in `evidence_refs` / `claim_refs`
4. Return structured `AgentReportSchema`

## Step 5: Modify `src/agents/orchestrator.py`

```python
import asyncio
from src.agents.langgraph_agents import (
    create_news_agent, create_technical_agent,
    create_fundamental_agent, create_risk_agent,
    _make_llm,
)
from src.agents.kg_tools import build_kg_tools, DynamicDatePolicy, FixedCutoffPolicy
from src.agents.roles import portfolio_manager_decide  # deterministic PM

AGENT_BACKEND = os.getenv("AGENT_BACKEND", "langgraph")  # "langgraph" | "legacy"

class KGBasedOrchestrator:
    def __init__(self, kg_context, config=None):
        self.kg_context = kg_context
        self.config = config or OrchestratorConfig()
        self.llm = _make_llm()
        self.semaphore = asyncio.Semaphore(2)  # rate limit

    async def _run_agent_safely(self, agent, input_data, role, ticker, trade_date):
        """Run a single agent with timeout + error recovery."""
        try:
            async with self.semaphore:
                result = await asyncio.wait_for(
                    agent.ainvoke(input_data), timeout=90
                )
            structured = result.get("structured_response")
            if structured:
                return AgentReport(role=role, **structured.model_dump())
        except Exception as exc:
            pass
        # Fallback: neutral report
        return AgentReport(
            role=role, stance="uncertain", confidence=0.0, score=0.0,
            summary=f"{role} agent failed: {type(exc).__name__}",
            evidence_refs=[], claim_refs=[], factors=[],
        )

    async def run_for_ticker_async(self, ticker, trade_date):
        tools = build_kg_tools(self.kg_context.kg_client, DynamicDatePolicy())
        input_data = {"messages": [HumanMessage(
            content=f"Analyze {ticker} as of {trade_date.date().isoformat()}"
        )]}

        news_ag = create_news_agent(self.llm, tools)
        tech_ag = create_technical_agent(self.llm, tools)
        fund_ag = create_fundamental_agent(self.llm, tools)
        risk_ag = create_risk_agent(self.llm, tools)

        reports = await asyncio.gather(
            self._run_agent_safely(news_ag, input_data, "news", ticker, trade_date),
            self._run_agent_safely(tech_ag, input_data, "technical", ticker, trade_date),
            self._run_agent_safely(fund_ag, input_data, "fundamental", ticker, trade_date),
            self._run_agent_safely(risk_ag, input_data, "risk", ticker, trade_date),
        )

        # Validate evidence refs against actually retrieved IDs
        _validate_evidence_refs(reports, tools)

        # Deterministic PM (same logic as current)
        decision = portfolio_manager_decide(ticker, trade_date, list(reports))
        return decision

    def run_for_ticker(self, ticker, trade_date):
        """Sync wrapper."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_for_ticker_async(ticker, trade_date))
        raise RuntimeError("Event loop already running; use await run_for_ticker_async()")
```

## Step 6: Evidence Reference Validation

After each agent run, validate that cited evidence/claim IDs actually exist in the tool results:

```python
def _validate_evidence_refs(reports, tools):
    """Check that evidence_refs and claim_refs are valid.

    - Remove invalid refs
    - Log invalid_reference_count per agent
    - If >50% invalid, downgrade stance to 'uncertain'
    """
```

These metrics go into the experiment CSV for grounding evaluation.

## Step 7: `static_kg` Uses Same Agents

`baseline_static_kg` in `src/eval/baselines.py` is refactored to use the same LangGraph agents with `FixedCutoffPolicy`:

```python
def baseline_static_kg(ticker, trade_date, kg_context, static_cutoff):
    tools = build_kg_tools(kg_context.kg_client, FixedCutoffPolicy(static_cutoff))
    # Same agent creation + parallel execution as kg_dynamic
    # Only difference: tools use FixedCutoffPolicy
```

This ensures the ONLY variable between `kg_dynamic` and `static_kg` is the KG cutoff date.

Baseline 1 (`no_kg_no_evidence`) and Baseline 2 (`evidence_no_kg`) keep the current direct LLM call approach — they don't use the KG at all.

## Step 8: Move Deterministic PM to `src/agents/pm_deterministic.py`

Extract `portfolio_manager_decide()` from `roles.py` into its own module. `roles.py` becomes thin (just helper functions and the `_llm_report` for legacy baselines).

## Step 9: Env Var Configuration

```bash
# .env
AGENT_BACKEND=langgraph    # "langgraph" | "legacy"
PM_MODE=deterministic       # "deterministic" | "langgraph" (phase 2)
```

`main_experiment.py` reads `AGENT_BACKEND` to choose orchestrator.

## Step 10: Agentic PM as Optional Ablation (Phase 2)

Not implemented in this phase. When ready:
- Add `query_past_decisions(ticker, as_of_date, limit)` with strict temporal guard
- Enforce: `past_trade_date < current_as_of_date`, `outcome_exit_date <= current_as_of_date`
- Enable via `PM_MODE=langgraph`
- Add as 5th system in experiment: `kg_dynamic_agentic_pm`

## New File Structure

```
src/agents/
├── __init__.py
├── base_agent.py          # unchanged
├── schemas.py             # NEW: FactorSchema, AgentReportSchema, DecisionSchema
├── kg_tools.py            # REFACTORED: build_kg_tools, date policies, cache
├── langgraph_agents.py    # NEW: 4 analyst agent factories
├── orchestrator.py        # MODIFIED: async parallel, AGENT_BACKEND switch
├── pm_deterministic.py    # MOVED: portfolio_manager_decide from roles.py
├── roles.py               # SIMPLIFIED: _llm_report for legacy baselines only
├── fundamental_agent.py   # KEEP: legacy rule-based (for legacy backend)
├── risk_agent.py          # KEEP: legacy rule-based (for legacy backend)
├── technical_agent.py     # KEEP: OHLCV → Entity/Relation generation
└── tools.py               # REMOVED: merged into kg_tools.py
```

## Implementation Order

| Step | Description | Risk |
|---|---|---|
| 0 | API smoke test (4 checks) | High — if endpoint doesn't support tools/structured output, entire plan changes |
| 1 | Fix `FundamentalAgent.claim_refs` bug | Low |
| 2 | Create `schemas.py` | Low |
| 3 | Refactor `kg_tools.py` (date policy + cache) | Medium — must not break existing tests |
| 4 | Create `langgraph_agents.py` (one agent first: news) | Medium — API compatibility |
| 5 | Convert remaining 3 analysts | Low |
| 6 | Modify `orchestrator.py` (async + AGENT_BACKEND switch) | Medium |
| 7 | Move PM to `pm_deterministic.py` | Low |
| 8 | `static_kg` reuses same agents with FixedCutoffPolicy | Medium |
| 9 | Single ticker/date smoke test | — |
| 10 | Full experiment run | — |

## Expected Impact

Analyst execution can be parallelized (4 concurrent LLM calls instead of 3 sequential), but total latency depends on:
- Tool-call rounds per agent (1-2 expected)
- Provider concurrency limits (semaphore=2)
- Structured output retries
- Whether agents skip tool calls when data is empty

Runtime improvement will be measured empirically, not assumed.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Endpoint doesn't support tool_calls | Step 0 smoke test; fallback to legacy backend |
| LangGraph structured output parsing fails | `ToolStrategy` handles retry natively; fallback to neutral report |
| Async event loop conflicts | Explicit `asyncio.run()` with RuntimeError guard |
| Tool cache stale across agents | Cache scoped per `run_for_ticker_async()` call, cleared after |
| PM deterministic logic breaks when moved | Run existing tests after Step 7 |
| static_kg produces different results | Use identical agents + prompts; only cutoff differs |
