from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class DummyLLM:
    """Placeholder for real LLM calls; for now just encodes simple rules."""

    def ask(self, prompt: str) -> str:
        # 为了 MVP 和可复现成本，这里暂时不用真实 LLM。
        if "Should I buy this stock" in prompt:
            return "hold"
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
    # 简化：假设检索了一些文本证据，但不构建 KG。
    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": "hold",
        "baseline": "evidence_no_kg",
    }


def baseline_static_kg(ticker: str, trade_date: datetime) -> Dict[str, Any]:
    # 简化：静态 KG 视作总是给出“中性”信号。
    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "action": "hold",
        "baseline": "static_kg",
    }

