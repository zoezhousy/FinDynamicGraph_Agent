# FinDynamicGraph Agent
Dissertation: Dynamic Evidence-Grounded Financial Knowledge Graph for Multi-Agent Simulated Trading

## Goal

This project builds a dynamic financial knowledge graph as shared memory for a multi-agent trading decision-support system. 
Agents do not only exchange natural-language summaries; instead, they add, update, query, and verify graph evidence before generating buy / sell / hold / abstain decisions.

## Core Modules

1. Data collection: market prices, financial reports, news, and optional sentiment data.
2. Knowledge graph construction: entities, events, technical signals, fundamentals, risks, and evidence sources.
3. Dynamic graph update: new information updates graph state without deleting old evidence.
4. Multi-agent reasoning: news, technical, fundamental, risk, and portfolio agents.
5. Simulated trading: evaluate decisions in a controlled backtesting environment.
6. Evaluation: grounding quality, hallucination rate, decision accuracy, and risk-aware behavior.

## MVP Plan

- Phase 1: Build local graph schema and mock data pipeline.
- Phase 2: Add market data from Yahoo Finance.
- Phase 3: Add evidence-grounded graph updates.
- Phase 4: Add simple agents.
- Phase 5: Add backtesting and baseline comparison.