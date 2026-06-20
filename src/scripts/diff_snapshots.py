"""Diff two KG snapshots to prove temporal divergence.

Usage:
    python -m src.scripts.diff_snapshots \
        --a data/experiments/snapshot_0700_2025-01-01.json \
        --b data/experiments/snapshot_0700_2025-03-01.json

Output: human-readable diff + exit code (0 = identical, 1 = different).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _diff_counts(a: dict, b: dict, keys: list[str]) -> list[str]:
    lines = []
    for k in keys:
        va, vb = a.get(k, 0), b.get(k, 0)
        if va != vb:
            delta = vb - va
            sign = "+" if delta > 0 else ""
            lines.append(f"  {k}: {va} → {vb}  ({sign}{delta})")
    return lines


def _diff_id_set(a_items: list[dict], b_items: list[dict], id_key: str) -> dict:
    """Return added / removed / changed item ids."""
    a_map = {x[id_key]: x for x in a_items if x.get(id_key)}
    b_map = {x[id_key]: x for x in b_items if x.get(id_key)}
    a_ids = set(a_map)
    b_ids = set(b_map)
    return {
        "added": sorted(b_ids - a_ids),
        "removed": sorted(a_ids - b_ids),
        "shared": sorted(a_ids & b_ids),
        "a_map": a_map,
        "b_map": b_map,
    }


def diff_snapshots(a_path: str, b_path: str) -> tuple[str, bool]:
    """Return (diff_text, is_different)."""

    sa = _load(a_path)
    sb = _load(b_path)

    lines: list[str] = []
    lines.append(f"═══ Snapshot Diff ═══")
    lines.append(f"  A: {a_path}  (as_of={sa.get('as_of_date')})")
    lines.append(f"  B: {b_path}  (as_of={sb.get('as_of_date')})")
    lines.append("")

    # ── 1. count-level diff ──
    count_keys = ["n_signals", "n_news", "n_fundamentals", "n_risks",
                  "n_evidences", "n_sources", "n_claims"]
    count_diffs = _diff_counts(sa, sb, count_keys)
    if count_diffs:
        lines.append("── Node Count Changes ──")
        lines.extend(count_diffs)
        lines.append("")
    else:
        lines.append("── Node Counts: identical ──\n")

    # ── 2. signal-level diff ──
    sig_diff = _diff_id_set(sa.get("top_signals", []), sb.get("top_signals", []),
                             "entity_id")
    if sig_diff["added"] or sig_diff["removed"]:
        lines.append("── Signal Diff ──")
        for sid in sig_diff["added"]:
            s = sig_diff["b_map"][sid]
            lines.append(f"  + ADDED  {sid}  ({s.get('as_of_date', '?')})")
        for sid in sig_diff["removed"]:
            s = sig_diff["a_map"][sid]
            lines.append(f"  - REMOVED {sid}  ({s.get('as_of_date', '?')})")
        lines.append("")

    # ── 3. news-level diff ──
    news_diff = _diff_id_set(sa.get("top_news", []), sb.get("top_news", []),
                              "entity_id")
    if news_diff["added"] or news_diff["removed"]:
        lines.append("── News Diff ──")
        for nid in news_diff["added"]:
            n = news_diff["b_map"][nid]
            title = (n.get("title") or n.get("headline") or "")[:80]
            lines.append(f"  + ADDED  {nid}  {title}")
        for nid in news_diff["removed"]:
            n = news_diff["a_map"][nid]
            title = (n.get("title") or n.get("headline") or "")[:80]
            lines.append(f"  - REMOVED {nid}  {title}")
        lines.append("")

    # ── 4. fundamental / risk diff ──
    for label, key in [("Fundamentals", "fundamentals_summary"), ("Risks", "risks_summary")]:
        diff = _diff_id_set(sa.get(key, []), sb.get(key, []), "entity_id")
        if diff["added"] or diff["removed"]:
            lines.append(f"── {label} Diff ──")
            for eid in diff["added"]:
                lines.append(f"  + ADDED  {eid}")
            for eid in diff["removed"]:
                lines.append(f"  - REMOVED {eid}")
            lines.append("")

    # ── 5. evidence / claim diff ──
    ev_diff = _diff_id_set(sa.get("evidence_refs", []), sb.get("evidence_refs", []),
                            "evidence_id")
    if ev_diff["added"] or ev_diff["removed"]:
        lines.append("── Evidence Diff ──")
        for eid in ev_diff["added"]:
            e = ev_diff["b_map"][eid]
            lines.append(f"  + ADDED  {eid}  ({e.get('source_name', '')})")
        for eid in ev_diff["removed"]:
            e = ev_diff["a_map"][eid]
            lines.append(f"  - REMOVED {eid}  ({e.get('source_name', '')})")
        lines.append("")

    cl_diff = _diff_id_set(sa.get("claim_refs", []), sb.get("claim_refs", []),
                            "claim_id")
    if cl_diff["added"] or cl_diff["removed"]:
        lines.append("── Claim Diff ──")
        for cid in cl_diff["added"]:
            c = cl_diff["b_map"][cid]
            lines.append(f"  + ADDED  {cid}  {c.get('text', '')[:60]}")
        for cid in cl_diff["removed"]:
            c = cl_diff["a_map"][cid]
            lines.append(f"  - REMOVED {cid}  {c.get('text', '')[:60]}")
        lines.append("")

    # ── verdict ──
    is_different = bool(count_diffs or sig_diff["added"] or sig_diff["removed"]
                        or news_diff["added"] or news_diff["removed"]
                        or ev_diff["added"] or ev_diff["removed"]
                        or cl_diff["added"] or cl_diff["removed"])

    if is_different:
        lines.append("═══ VERDICT: Snapshots are DIFFERENT — temporal divergence PROVEN ═══")
    else:
        lines.append("═══ VERDICT: Snapshots are IDENTICAL ═══")

    return "\n".join(lines), is_different


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two KG snapshots")
    parser.add_argument("--a", required=True, help="Path to snapshot A")
    parser.add_argument("--b", required=True, help="Path to snapshot B")
    args = parser.parse_args()

    text, different = diff_snapshots(args.a, args.b)
    print(text)
    sys.exit(1 if different else 0)


if __name__ == "__main__":
    main()
