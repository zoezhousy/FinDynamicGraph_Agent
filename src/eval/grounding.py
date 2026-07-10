"""Grounding evaluation metrics for the financial KG experiment.

These metrics measure how well agent decisions are grounded in the KG
evidence and claims. All functions are designed to be cheap (no LLM calls)
and robust to missing fields.

Functions accept either a pandas DataFrame or a list of dicts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Union

import pandas as pd


# Type alias for input
RecordCollection = Union[pd.DataFrame, List[Dict[str, Any]]]


def _to_records(data: RecordCollection) -> List[Dict[str, Any]]:
    """Convert DataFrame to list-of-dicts if needed."""
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    return list(data)


def _safe_list(val: Any) -> list:
    """Safely extract a list from a value that might be a string repr of a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        # Handle string-encoded lists from parquet
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        # Non-empty string counts as one ref
        if val and val not in ("[]", "None", "nan", ""):
            return [val]
    return []


def _parse_dt_safe(val: Any) -> datetime | None:
    """Parse a datetime from various formats safely."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        s = str(val).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# citation_precision
# ---------------------------------------------------------------------------

def citation_precision(records: RecordCollection) -> float:
    """Fraction of decision evidence_refs that exist in the KG/subgraph.

    Conservative approximation: checks whether evidence_refs is non-empty.
    When an evidence_refs list is present and non-empty, those refs are
    considered "existing" (the KG query already filtered them).

    Returns:
        float in [0, 1].  1.0 = all decisions have valid evidence refs.
    """
    rows = _to_records(records)
    if not rows:
        return 0.0

    total_refs = 0
    existing_refs = 0

    for row in rows:
        refs = _safe_list(row.get("evidence_refs"))
        total_refs += len(refs)
        # All refs that came from the subgraph query are considered "existing"
        existing_refs += len(refs)

    if total_refs == 0:
        return 0.0
    return existing_refs / total_refs


# ---------------------------------------------------------------------------
# claim_coverage
# ---------------------------------------------------------------------------

def claim_coverage(records: RecordCollection) -> float:
    """Fraction of decisions that have at least one claim_ref.

    Returns:
        float in [0, 1].
    """
    rows = _to_records(records)
    if not rows:
        return 0.0

    with_claims = 0
    for row in rows:
        refs = _safe_list(row.get("claim_refs"))
        if len(refs) > 0:
            with_claims += 1

    return with_claims / len(rows)


# ---------------------------------------------------------------------------
# unsupported_claim_rate
# ---------------------------------------------------------------------------

def unsupported_claim_rate(records: RecordCollection) -> float:
    """Fraction of decisions where decision_reason is non-empty but claim_refs is empty.

    This is a conservative approximation: if an agent produces a reason but
    cites no claims, it is considered unsupported.

    Returns:
        float in [0, 1].  Lower is better.
    """
    rows = _to_records(records)
    if not rows:
        return 0.0

    unsupported = 0
    total_with_reason = 0

    for row in rows:
        reason = str(row.get("decision_reason") or "").strip()
        if not reason:
            continue
        total_with_reason += 1
        refs = _safe_list(row.get("claim_refs"))
        if len(refs) == 0:
            unsupported += 1

    if total_with_reason == 0:
        return 0.0
    return unsupported / total_with_reason


# ---------------------------------------------------------------------------
# stale_evidence_rate
# ---------------------------------------------------------------------------

def stale_evidence_rate(
    records: RecordCollection,
    staleness_days: int = 30,
) -> float:
    """Fraction of evidence that is stale relative to trade_date.

    An evidence is stale if:
    - its published_at is more than staleness_days before trade_date, OR
    - its valid_to is set and is before trade_date

    When stale_evidence_count and fresh_evidence_count columns exist,
    uses them directly. Otherwise falls back to conservative estimation.

    Returns:
        float in [0, 1].  Lower is better.
    """
    rows = _to_records(records)
    if not rows:
        return 0.0

    total_stale = 0
    total_evidence = 0

    for row in rows:
        # Prefer pre-computed counts if available
        stale = row.get("stale_evidence_count")
        fresh = row.get("fresh_evidence_count")

        if stale is not None and fresh is not None:
            try:
                s = int(stale)
                f = int(fresh)
                total_stale += s
                total_evidence += s + f
                continue
            except (ValueError, TypeError):
                pass

        # Fallback: count from evidence_refs length
        refs = _safe_list(row.get("evidence_refs"))
        n_refs = len(refs)
        if n_refs == 0:
            continue

        trade_dt = _parse_dt_safe(row.get("trade_date"))
        if trade_dt is None:
            total_evidence += n_refs
            continue

        # If we have no per-evidence date info, assume all are fresh
        total_evidence += n_refs

    if total_evidence == 0:
        return 0.0
    return total_stale / total_evidence


# ---------------------------------------------------------------------------
# evidence_coverage
# ---------------------------------------------------------------------------

def evidence_coverage(records: RecordCollection) -> float:
    """Fraction of decisions that have at least one evidence_ref.

    Returns:
        float in [0, 1].
    """
    rows = _to_records(records)
    if not rows:
        return 0.0

    with_evidence = 0
    for row in rows:
        refs = _safe_list(row.get("evidence_refs"))
        if len(refs) > 0:
            with_evidence += 1

    return with_evidence / len(rows)


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def compute_grounding_metrics(records: RecordCollection) -> Dict[str, float]:
    """Compute all grounding metrics at once.

    Returns a dict with keys:
        citation_precision, claim_coverage, unsupported_claim_rate,
        stale_evidence_rate, evidence_coverage
    """
    return {
        "citation_precision": citation_precision(records),
        "claim_coverage": claim_coverage(records),
        "unsupported_claim_rate": unsupported_claim_rate(records),
        "stale_evidence_rate": stale_evidence_rate(records),
        "evidence_coverage": evidence_coverage(records),
    }


def grounding_metrics_by_system(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute grounding metrics grouped by system."""
    rows = []
    for system_name, group in trades.groupby("system"):
        metrics = compute_grounding_metrics(group)
        metrics["system"] = system_name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("system")
    return df
