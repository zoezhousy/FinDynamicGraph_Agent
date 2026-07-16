"""Tests for paired significance tests in src/eval/metrics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import paired_significance_tests


# ── Helpers ────────────────────────────────────────────────────────────


def _make_pair(
    ticker: str,
    trade_date: str,
    system: str,
    raw_return: float,
    action: str = "buy",
) -> dict:
    """Build a single trade row for the significance test fixtures."""
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "system": system,
        "action": action,
        "trade_executed": action != "abstain",
        "raw_return": raw_return,
    }


def _paired_df(
    returns_a: list[float],
    returns_b: list[float],
    sys_a: str = "kg_dynamic",
    sys_b: str = "static_kg",
) -> pd.DataFrame:
    """Build a two-system paired DataFrame from aligned return lists.

    Both systems share the same (ticker, trade_date) pairs.
    """
    assert len(returns_a) == len(returns_b), "paired lists must match"
    rows = []
    for i, (ra, rb) in enumerate(zip(returns_a, returns_b)):
        ticker = "0700.HK"
        date = f"2025-{i + 1:02d}-01"
        rows.append(_make_pair(ticker, date, sys_a, ra))
        rows.append(_make_pair(ticker, date, sys_b, rb))
    return pd.DataFrame(rows)


# ── Return values (point estimates) ───────────────────────────────────


class TestReturnPointEstimates:
    """Basic sanity: function runs and produces expected columns."""

    def test_columns(self):
        df = _paired_df([0.01, 0.02, -0.01], [0.00, -0.01, 0.00])
        result = paired_significance_tests(df)
        expected_cols = {"system_a", "system_b", "test", "statistic", "p_value", "significant"}
        assert set(result.columns) == expected_cols

    def test_two_systems_two_rows(self):
        """Two systems → 1 pair × 2 tests = 2 rows."""
        df = _paired_df([0.01, 0.02], [0.00, -0.01])
        result = paired_significance_tests(df)
        assert len(result) == 2
        assert set(result["test"]) == {"wilcoxon_raw_return", "mcnemar_directional"}

    def test_empty_for_single_system(self):
        df = pd.DataFrame([_make_pair("0700.HK", "2025-01-01", "kg_dynamic", 0.01)])
        result = paired_significance_tests(df)
        assert result.empty

    def test_empty_input(self):
        result = paired_significance_tests(pd.DataFrame())
        assert result.empty


# ── Clear difference: both tests should be significant ─────────────────


class TestClearDifference:
    """System A consistently outperforms system B → p < alpha."""

    @pytest.fixture()
    def result(self):
        # System A: consistently positive; System B: consistently negative
        # 20 paired observations with clear separation
        np.random.seed(42)
        n = 20
        returns_a = np.random.normal(0.03, 0.005, n).tolist()  # ~3% mean
        returns_b = np.random.normal(-0.02, 0.005, n).tolist()  # ~-2% mean
        df = _paired_df(returns_a, returns_b)
        return paired_significance_tests(df)

    def test_wilcoxon_significant(self, result):
        row = result[result["test"] == "wilcoxon_raw_return"].iloc[0]
        assert row["p_value"] < 0.05
        assert row["significant"] == True  # noqa: E712

    def test_mcnemar_significant(self, result):
        row = result[result["test"] == "mcnemar_directional"].iloc[0]
        # A always positive return → always correct; B always negative → always incorrect
        # So b=20, c=0 → very significant
        assert row["p_value"] < 0.05
        assert row["significant"] == True  # noqa: E712


# ── No difference: both tests should NOT be significant ────────────────


class TestNoDifference:
    """Both systems have identical returns → p >= alpha."""

    @pytest.fixture()
    def result(self):
        # Identical returns for both systems
        np.random.seed(123)
        n = 30
        shared = np.random.normal(0.01, 0.02, n).tolist()
        df = _paired_df(shared, shared)
        return paired_significance_tests(df)

    def test_wilcoxon_not_significant(self, result):
        row = result[result["test"] == "wilcoxon_raw_return"].iloc[0]
        # All diffs = 0 → NaN p-value
        assert np.isnan(row["p_value"]) or row["p_value"] >= 0.05
        assert row["significant"] == False  # noqa: E712

    def test_mcnemar_not_significant(self, result):
        row = result[result["test"] == "mcnemar_directional"].iloc[0]
        # Same correctness → b=c=0 → NaN
        assert np.isnan(row["p_value"]) or row["p_value"] >= 0.05
        assert row["significant"] == False  # noqa: E712


# ── Near-zero difference: should not be significant ────────────────────


class TestSmallDifference:
    """Tiny difference in means with moderate variance → not significant."""

    @pytest.fixture()
    def result(self):
        np.random.seed(99)
        n = 15
        returns_a = np.random.normal(0.010, 0.03, n).tolist()
        returns_b = np.random.normal(0.008, 0.03, n).tolist()  # 0.2pp difference
        df = _paired_df(returns_a, returns_b)
        return paired_significance_tests(df)

    def test_wilcoxon_not_significant(self, result):
        row = result[result["test"] == "wilcoxon_raw_return"].iloc[0]
        assert np.isnan(row["p_value"]) or row["p_value"] >= 0.05


# ── McNemar edge case: only off-diagonal matters ──────────────────────


class TestMcNemarEdgeCases:
    def test_all_agree_no_discordant(self):
        """When both systems are always correct or always incorrect together, NaN."""
        # Both always positive → always correct → b=c=0
        df = _paired_df([0.01, 0.02, 0.03], [0.01, 0.02, 0.03])
        result = paired_significance_tests(df)
        mcnemar = result[result["test"] == "mcnemar_directional"].iloc[0]
        assert np.isnan(mcnemar["p_value"])
        assert mcnemar["significant"] == False  # noqa: E712

    def test_all_disagree_one_direction(self):
        """When b=10, c=0 (all discordant same way), should be very significant."""
        # A always positive, B always negative
        df = _paired_df(
            [0.01] * 10,
            [-0.01] * 10,
        )
        result = paired_significance_tests(df)
        mcnemar = result[result["test"] == "mcnemar_directional"].iloc[0]
        assert mcnemar["p_value"] < 0.01
        assert mcnemar["significant"] == True  # noqa: E712

    def test_only_two_pairs(self):
        """With only 2 pairs, still runs without error."""
        df = _paired_df([0.01, -0.01], [0.01, 0.01])
        result = paired_significance_tests(df)
        assert len(result) == 2


# ── Three-system comparison ───────────────────────────────────────────


class TestThreeSystems:
    """With 3 systems we get C(3,2) = 3 pairs × 2 tests = 6 rows."""

    def test_six_rows(self):
        np.random.seed(7)
        n = 10
        rows = []
        for i in range(n):
            date = f"2025-{i + 1:02d}-01"
            rows.append(_make_pair("0700.HK", date, "kg_dynamic", 0.03))
            rows.append(_make_pair("0700.HK", date, "static_kg", 0.01))
            rows.append(_make_pair("0700.HK", date, "no_kg_no_evidence", -0.01))
        df = pd.DataFrame(rows)
        result = paired_significance_tests(df)
        assert len(result) == 6
        pairs = set(zip(result["system_a"], result["system_b"]))
        assert len(pairs) == 3
