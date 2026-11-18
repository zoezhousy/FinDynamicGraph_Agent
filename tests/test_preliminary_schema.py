from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.eval.run_preliminary import RESULT_COLUMNS, _normalise_result
from src.eval.temporal_audit import audit_items
from src.sim.backtest import BacktestConfig


def test_all_systems_share_exact_preliminary_schema() -> None:
    ohlcv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-02", "2025-06-03"]),
            "open": [100.0, 101.0],
            "close": [101.0, 102.0],
        }
    )
    audit = audit_items(
        [{"evidence_id": "ev1", "published_at": "2025-05-30"}],
        datetime(2025, 6, 1),
    )
    base = {
        "action": "hold",
        "confidence": 0.5,
        "final_score": 0.0,
        "evidence_refs": ["ev1"],
        "retrieved_evidence_ids": ["ev1"],
    }

    for system in ("direct_llm", "text_rag", "static_kg", "kg_dynamic"):
        row = _normalise_result(
            base,
            "0700.HK",
            datetime(2025, 6, 1),
            system,
            audit,
            0.01,
            0.02,
            "hash",
            {"evidences": 1},
            ohlcv,
            BacktestConfig(),
        )
        assert tuple(row) == RESULT_COLUMNS
        assert row["future_count"] == 0
