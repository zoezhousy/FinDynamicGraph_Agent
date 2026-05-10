from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class DummyLLM:
    """Placeholder for real LLM calls; for now just encodes simple rules."""

    def ask(self, prompt: str) -> str:
        if "Should I buy this stock" in prompt:
            return "buy"
        return "hold"


def baseline_no_kg_no_evidence(ticker: str, trade_date: datetime, llm: DummyLLM | None = None) -> Dict[str, Any]:
    llm = llm or DummyLLM()
    action = llm.ask(f"Should I buy this stock {ticker} on {trade_date.date().isoformat()}?")
    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": action,
        "baseline": "no_kg_no_evidence",
    }


def baseline_evidence_no_kg(ticker: str, trade_date: datetime) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": "buy",
        "baseline": "evidence_no_kg",
    }


def baseline_static_kg(ticker: str, trade_date: datetime) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": "sell",
        "baseline": "static_kg",
    }
