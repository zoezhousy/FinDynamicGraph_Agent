"""Inspect KG snapshot for a ticker at a given as_of_date.

Usage::

    python -m src.scripts.inspect_snapshot --ticker 0700.HK --date 2024-06-01
    python -m src.scripts.inspect_snapshot --ticker 0700.HK --date 2025-06-01
    python -m src.scripts.inspect_snapshot --ticker 0700.HK --date 2025-06-01 --save

Options:
    --save   Also write JSON output to data/experiments/snapshot_<ticker>_<date>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _get_neo4j_env() -> tuple[str, str, str, str]:
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "neo4j")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    return uri, user, password, database


def _try_query_client():
    """Try to build a KGQueryClient from env vars; return None if Neo4j unreachable."""
    try:
        from src.kg.query import KGQueryClient

        uri, user, password, database = _get_neo4j_env()
        client = KGQueryClient(uri, user, password, database)
        # Quick health probe
        with client._driver.session(database=database) as session:
            session.run("RETURN 1").single()
        return client
    except Exception as exc:
        print(f"[warn] Cannot connect to Neo4j ({exc}); falling back to offline mode.")
        return None


def _offline_summary(ticker: str, as_of_date: datetime) -> dict:
    """Generate an empty/offline summary when Neo4j is unavailable."""
    from src.kg.query import _empty_summary
    return _empty_summary(ticker, as_of_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect KG snapshot for a ticker at a given date.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. 0700.HK")
    parser.add_argument("--date", required=True, help="As-of date in YYYY-MM-DD format")
    parser.add_argument("--save", action="store_true", help="Save JSON output to data/experiments/")
    args = parser.parse_args()

    ticker = args.ticker
    as_of_date = datetime.fromisoformat(args.date)

    client = _try_query_client()
    if client:
        summary = client.get_snapshot_summary(ticker, as_of_date)
        subgraph = client.get_ticker_subgraph(ticker, as_of_date)
        client.close()
    else:
        summary = _offline_summary(ticker, as_of_date)
        subgraph = {}

    # ── Print readable output ──────────────────────────────────────────
    print("=" * 60)
    print(f"  KG Snapshot Inspection: {ticker}")
    print(f"  As-of Date: {as_of_date.strftime('%Y-%m-%d')}")
    print("=" * 60)

    print(f"\n{'─' * 40}")
    print("  Summary")
    print(f"{'─' * 40}")
    print(f"  Active signals:        {summary['active_signal_count']}")
    print(f"  Active claims:         {summary['active_claim_count']}")
    print(f"    Bullish (supports):  {summary['bullish_claim_count']}")
    print(f"    Bearish (contradicts): {summary['bearish_claim_count']}")
    print(f"    Neutral:             {summary['neutral_claim_count']}")
    print(f"  Evidence count:        {summary['evidence_count']}")
    print(f"    Fresh evidence:      {summary['fresh_evidence_count']}")
    print(f"    Stale evidence:      {summary['stale_evidence_count']}")
    print(f"  Conflict count:        {summary['conflict_count']}")

    # Top claims
    top_claims = summary.get("top_claims", [])
    if top_claims:
        print(f"\n{'─' * 40}")
        print("  Top 5 Claims (by confidence)")
        print(f"{'─' * 40}")
        for i, claim in enumerate(top_claims[:5], 1):
            polarity = claim.get("polarity", "?")
            conf = claim.get("confidence", 0)
            text = claim.get("text", "")[:80]
            claim_id = claim.get("claim_id", "?")
            print(f"  {i}. [{polarity}] conf={conf:.2f}  {text}")
            print(f"     id={claim_id}")

    # Top sources
    top_sources = summary.get("top_sources", [])
    if top_sources:
        print(f"\n{'─' * 40}")
        print("  Top 5 Evidence Sources")
        print(f"{'─' * 40}")
        for i, src in enumerate(top_sources[:5], 1):
            print(f"  {i}. {src['source']}  (count={src['count']})")

    # Latest evidence dates
    latest_dates = summary.get("latest_evidence_dates", [])
    if latest_dates:
        print(f"\n{'─' * 40}")
        print("  Latest Evidence Dates")
        print(f"{'─' * 40}")
        for d in latest_dates[:5]:
            print(f"  - {d}")

    # Conflicts from subgraph
    conflicts = subgraph.get("conflicts", []) if subgraph else []
    if conflicts:
        print(f"\n{'─' * 40}")
        print("  Conflict Pairs")
        print(f"{'─' * 40}")
        for pair in conflicts[:10]:
            print(f"  - {pair['claim_a']}  <->  {pair['claim_b']}")

    # Signal details
    signals = subgraph.get("signals", []) if subgraph else []
    if signals:
        print(f"\n{'─' * 40}")
        print(f"  Active Signals ({len(signals)})")
        print(f"{'─' * 40}")
        for sig in signals[:5]:
            direction = sig.get("direction", "?")
            name = sig.get("name", sig.get("signal_type", "?"))
            strength = sig.get("strength", 0)
            as_of = sig.get("as_of_date", "?")
            print(f"  - [{direction}] {name}  strength={strength:.2f}  as_of={as_of}")

    # Freshness summary
    print(f"\n{'─' * 40}")
    print("  Freshness Summary")
    print(f"{'─' * 40}")
    total_ev = summary["evidence_count"]
    if total_ev > 0:
        fresh_pct = summary["fresh_evidence_count"] / total_ev * 100
        print(f"  Fresh: {fresh_pct:.1f}% ({summary['fresh_evidence_count']}/{total_ev})")
    else:
        print("  No evidence found.")
    print()

    # ── Save JSON ──────────────────────────────────────────────────────
    if args.save:
        out_dir = Path("data/experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_ticker = ticker.replace(".", "")
        out_path = out_dir / f"snapshot_{safe_ticker}_{as_of_date.strftime('%Y-%m-%d')}.json"
        output = {
            "summary": summary,
            "subgraph_overview": {
                "signal_count": len(signals),
                "claim_count": len(subgraph.get("claims", [])) if subgraph else 0,
                "evidence_count": len(subgraph.get("evidences", [])) if subgraph else 0,
                "conflict_count": len(conflicts),
            },
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
