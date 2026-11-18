"""Temporal leakage audit for point-in-time retrieval results.

The audit is intentionally independent of Neo4j so it can be unit-tested and
applied to KG snapshots as well as raw text-RAG evidence records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence


AUDITED_COLLECTIONS = (
    "signals",
    "fundamentals",
    "risks",
    "news",
    "evidences",
    "claims",
)

_ID_FIELDS = (
    "entity_id",
    "evidence_id",
    "claim_id",
    "signal_id",
    "source_id",
    "id",
)

_AVAILABILITY_FIELDS = ("effective_at", "published_at", "as_of_date", "valid_from")


class TemporalLeakageError(AssertionError):
    """Raised when retrieved records contain future information."""


@dataclass(frozen=True)
class TemporalAuditResult:
    retrieved_count: int
    valid_count: int
    stale_count: int
    future_count: int
    temporal_precision: float
    stale_evidence_rate: float
    future_evidence_rate: float
    future_ids: tuple[str, ...]
    stale_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["future_ids"] = list(self.future_ids)
        result["stale_ids"] = list(self.stale_ids)
        return result


def audit_subgraph(
    subgraph: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    as_of_date: datetime,
    *,
    staleness_days: int = 30,
) -> TemporalAuditResult:
    """Audit all evidence-bearing records returned for a decision date.

    ``valid_count`` means the record was temporally valid at ``as_of_date``.
    ``stale_count`` is a subset of valid records whose latest availability date
    is older than ``staleness_days``; records expired by ``valid_to`` are also
    stale but not valid.  ``future_count`` is always invalid and must be zero in
    a point-in-time result.
    """
    return audit_items(
        flatten_subgraph(subgraph),
        as_of_date,
        staleness_days=staleness_days,
    )


def audit_items(
    items: Iterable[Mapping[str, Any]],
    as_of_date: datetime,
    *,
    staleness_days: int = 30,
) -> TemporalAuditResult:
    cutoff = _end_of_day(_normalise_datetime(as_of_date))
    stale_before = cutoff - timedelta(days=staleness_days)

    rows = list(items)
    valid_count = 0
    stale_count = 0
    future_ids: list[str] = []
    stale_ids: list[str] = []

    for index, item in enumerate(rows):
        item_id = _item_id(item, index)
        availability_dates = [
            parsed
            for field in _AVAILABILITY_FIELDS
            if (parsed := parse_datetime(item.get(field))) is not None
        ]
        valid_to = parse_datetime(item.get("valid_to"))

        is_future = any(value > cutoff for value in availability_dates)
        is_expired = valid_to is not None and valid_to <= cutoff
        is_valid = not is_future and not is_expired

        if is_future:
            future_ids.append(item_id)
        if is_valid:
            valid_count += 1

        anchor = max(availability_dates) if availability_dates else None
        is_stale = is_expired or (is_valid and anchor is not None and anchor < stale_before)
        if is_stale:
            stale_count += 1
            stale_ids.append(item_id)

    retrieved_count = len(rows)
    denominator = max(1, retrieved_count)
    return TemporalAuditResult(
        retrieved_count=retrieved_count,
        valid_count=valid_count,
        stale_count=stale_count,
        future_count=len(future_ids),
        temporal_precision=round(valid_count / denominator, 6) if retrieved_count else 0.0,
        stale_evidence_rate=round(stale_count / denominator, 6) if retrieved_count else 0.0,
        future_evidence_rate=round(len(future_ids) / denominator, 6) if retrieved_count else 0.0,
        future_ids=tuple(sorted(future_ids)),
        stale_ids=tuple(sorted(stale_ids)),
    )


def assert_no_future(audit: TemporalAuditResult, *, context: str = "") -> None:
    """Fail fast when an experiment retrieves post-decision information."""
    if audit.future_count:
        prefix = f"{context}: " if context else ""
        raise TemporalLeakageError(
            f"{prefix}future_count={audit.future_count}; ids={list(audit.future_ids)}"
        )


def flatten_subgraph(
    subgraph: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Return de-duplicated evidence-bearing records from a KG snapshot."""
    if not subgraph:
        return []

    flattened: list[dict[str, Any]] = []
    seen: set[str] = set()
    for collection in AUDITED_COLLECTIONS:
        for index, raw in enumerate(subgraph.get(collection, ()) or ()):
            item = dict(raw)
            identity = _item_id(item, index, collection)
            dedupe_key = f"{collection}:{identity}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            item.setdefault("_collection", collection)
            flattened.append(item)
    return flattened


def snapshot_fingerprint(
    subgraph: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> str:
    """Stable digest used to prove whether two snapshots differ."""
    canonical: list[dict[str, Any]] = []
    for item in flatten_subgraph(subgraph):
        canonical.append(
            {
                "collection": item.get("_collection"),
                "id": next((str(item[key]) for key in _ID_FIELDS if item.get(key)), ""),
                "valid_from": _json_value(item.get("valid_from")),
                "valid_to": _json_value(item.get("valid_to")),
                "effective_at": _json_value(item.get("effective_at")),
                "published_at": _json_value(item.get("published_at")),
                "as_of_date": _json_value(item.get("as_of_date")),
            }
        )
    canonical.sort(key=lambda row: (str(row["collection"]), str(row["id"])))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_counts(
    subgraph: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, int]:
    return {
        collection: len((subgraph or {}).get(collection, ()) or ())
        for collection in AUDITED_COLLECTIONS
    }


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _normalise_datetime(value)
    if hasattr(value, "to_native"):
        value = value.to_native()
        if isinstance(value, datetime):
            return _normalise_datetime(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _normalise_datetime(parsed)


def _normalise_datetime(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _end_of_day(value: datetime) -> datetime:
    if value.hour == value.minute == value.second == value.microsecond == 0:
        return value.replace(hour=23, minute=59, second=59, microsecond=999999)
    return value


def _item_id(item: Mapping[str, Any], index: int, collection: str = "item") -> str:
    for field in _ID_FIELDS:
        if item.get(field):
            return str(item[field]).strip()
    return f"{collection}:{index}"


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return None if value is None else str(value)
