"""Pure evidence-reference helpers; no model clients or graph connections."""
from datetime import datetime
from typing import Any, Dict, List
from src.agents.models import DecisionAction, Stance

def _is_stale_ref(evidence_ref: str, subgraph: Dict[str, List[Dict[str, Any]]], trade_date: datetime) -> bool:
    evidence_map = {
        str(item.get("evidence_id")): item
        for item in subgraph.get("evidences", [])
        if item.get("evidence_id")
    }
    signal_map = {
        str(item.get("entity_id")): item
        for item in subgraph.get("signals", [])
        if item.get("entity_id")
    }

    if evidence_ref in evidence_map:
        published_at = evidence_map[evidence_ref].get("published_at")
        if not published_at:
            return False
        try:
            return datetime.fromisoformat(str(published_at).replace("Z", "+00:00")) < trade_date
        except Exception:
            return False

    if evidence_ref in signal_map:
        valid_to = signal_map[evidence_ref].get("valid_to")
        if not valid_to:
            return False
        try:
            return datetime.fromisoformat(str(valid_to).replace("Z", "+00:00")) < trade_date
        except Exception:
            return False

    return False


def evidence_freshness(evidence_ref, subgraph, trade_date, *, staleness_days=30):
    """Return fresh/stale/unknown under the v2 30-day policy (HK for naive times)."""
    from datetime import timedelta
    from src.eval.protocol import available_time
    if staleness_days < 0:
        raise ValueError("staleness_days must be non-negative")
    cutoff = available_time(trade_date)
    if cutoff is None:
        return "unknown"
    for key, id_key in (("evidences", "evidence_id"), ("signals", "entity_id"), ("news", "evidence_id"), ("fundamentals", "evidence_id"), ("risks", "evidence_id"), ("sources", "source_id")):
        for item in subgraph.get(key, []) or []:
            if str(item.get(id_key)) != evidence_ref:
                continue
            expiry_value = item.get("valid_to")
            expiry = available_time(expiry_value)
            if expiry_value not in (None, "") and expiry is None:
                return "unknown"
            if expiry is not None and expiry <= cutoff:
                return "stale"
            timestamps = [available_time(item[field]) for field in ("effective_at", "published_at", "published_time", "as_of_date", "valid_from") if item.get(field) is not None]
            if not timestamps or any(value is None or value > cutoff for value in timestamps):
                return "unknown"
            # All availability constraints must be met; use the latest known availability.
            return "stale" if cutoff - max(timestamps) > timedelta(days=staleness_days) else "fresh"
    return "unknown"


def _aligns(action: DecisionAction, stance: Stance) -> bool:
    if action == "buy":
        return stance == "bullish"
    if action == "sell":
        return stance == "bearish"
    if action == "hold":
        return stance in {"neutral", "uncertain"}
    if action == "abstain":
        return stance in {"uncertain", "neutral"}
    return False

# collect fallback evidence refs from subgraph for risk agent 
# when explicit risk events are not available.
def _collect_fallback_evidence_refs(
    subgraph: Dict[str, List[Dict[str, Any]]] | None,
    limit: int = 20,
) -> List[str]:
    if subgraph is None:
        return []

    refs: list[str] = []

    for key, id_key in (
        ("risks", "evidence_id"),
        ("fundamentals", "evidence_id"),
        ("news", "evidence_id"),
        ("evidences", "evidence_id"),
        ("signals", "entity_id"),
    ):
        for item in subgraph.get(key, []) or []:
            value = item.get(id_key)
            if value:
                refs.append(str(value))

            if len(refs) >= limit:
                return sorted(set(refs))

    return sorted(set(refs))


def extract_retrieved_ids(
    subgraph: Dict[str, List[Dict[str, Any]]] | None,
) -> tuple[list[str], list[str]]:
    """Extract all evidence and claim IDs actually returned by a subgraph query.

    Returns:
        (retrieved_evidence_ids, retrieved_claim_ids) — sorted, deduplicated lists
        of string IDs.  Returns empty lists if subgraph is None.
    """
    if subgraph is None:
        return [], []

    evidence_ids: set[str] = set()
    claim_ids: set[str] = set()

    # Evidence nodes from the "evidences" key
    for item in subgraph.get("evidences", []) or []:
        eid = item.get("evidence_id")
        if eid:
            evidence_ids.add(str(eid).strip())

    # Evidence IDs from entity-level keys that carry evidence_id
    for key in ("news", "risks", "fundamentals"):
        for item in subgraph.get(key, []) or []:
            eid = item.get("evidence_id")
            if eid:
                evidence_ids.add(str(eid).strip())

    # Signal entities carry entity_id, not evidence_id
    for item in subgraph.get("signals", []) or []:
        eid = item.get("entity_id")
        if eid:
            evidence_ids.add(str(eid).strip())

    # Source documents
    for item in subgraph.get("sources", []) or []:
        sid = item.get("source_id")
        if sid:
            evidence_ids.add(str(sid).strip())

    # Claims
    for item in subgraph.get("claims", []) or []:
        cid = item.get("entity_id") or item.get("claim_id")
        if cid:
            claim_ids.add(str(cid).strip())

    return sorted(evidence_ids), sorted(claim_ids)

