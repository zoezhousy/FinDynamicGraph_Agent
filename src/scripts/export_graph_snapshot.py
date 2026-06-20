"""Export a point-in-time KG subgraph snapshot for a given ticker + as_of_date.

Usage:
    python -m src.scripts.export_graph_snapshot \
        --ticker 0700.HK --as-of 2025-03-01 \
        --output data/experiments/snapshot_0700_2025-03-01.json

The output JSON is a self-contained record that can be diffed against
snapshots taken at other dates to prove temporal divergence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.kg.query import KGQueryClient  # noqa: E402


def _summarise_snapshot(
    ticker: str,
    as_of: datetime,
    subgraph: dict,
) -> dict:
    """Flatten a subgraph into a compact, diff-friendly summary."""

    def _top(items: list[dict], key: str, n: int = 5) -> list[dict]:
        return sorted(items, key=lambda x: str(x.get(key, "")), reverse=True)[:n]

    signals = subgraph.get("signals", [])
    news = subgraph.get("news", [])
    fundamentals = subgraph.get("fundamentals", [])
    risks = subgraph.get("risks", [])
    evidences = subgraph.get("evidences", [])
    sources = subgraph.get("sources", [])
    claims = subgraph.get("claims", [])

    return {
        "ticker": ticker,
        "as_of_date": as_of.isoformat(),
        "exported_at": datetime.utcnow().isoformat() + "Z",

        # ── counts ──
        "n_signals": len(signals),
        "n_news": len(news),
        "n_fundamentals": len(fundamentals),
        "n_risks": len(risks),
        "n_evidences": len(evidences),
        "n_sources": len(sources),
        "n_claims": len(claims),

        # ── top items ──
        "top_signals": _top(signals, "as_of_date"),
        "top_news": _top(news, "published_at"),
        "fundamentals_summary": _top(fundamentals, "as_of_date"),
        "risks_summary": _top(risks, "as_of_date"),
        "evidence_refs": [
            {"evidence_id": e.get("evidence_id"), "source_name": e.get("source_name")}
            for e in evidences[:10]
        ],
        "claim_refs": [
            {"claim_id": c.get("claim_id"), "text": c.get("text", "")[:120]}
            for c in claims[:10]
        ],
    }


def export_snapshot(
    ticker: str,
    as_of: datetime,
    output_path: str | Path,
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
) -> dict:
    """Export and return the snapshot dict (also written to *output_path*)."""

    uri = uri or os.environ["NEO4J_URI"]
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    password = password or os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", database)

    client = KGQueryClient(uri, user, password, database)
    try:
        subgraph = client.get_ticker_subgraph(ticker, as_of)
    finally:
        client.close()

    snapshot = _summarise_snapshot(ticker, as_of, subgraph)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return snapshot


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export KG snapshot")
    parser.add_argument("--ticker", required=True, help="e.g. 0700.HK")
    parser.add_argument("--as-of", required=True, help="ISO date, e.g. 2025-03-01")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    load_dotenv()
    as_of = datetime.fromisoformat(args.as_of)
    snapshot = export_snapshot(args.ticker, as_of, args.output)

    print(f"✅ Snapshot exported → {args.output}")
    print(f"   ticker={snapshot['ticker']}  as_of={snapshot['as_of_date']}")
    print(f"   signals={snapshot['n_signals']}  news={snapshot['n_news']}  "
          f"fundamentals={snapshot['n_fundamentals']}  risks={snapshot['n_risks']}  "
          f"evidences={snapshot['n_evidences']}  claims={snapshot['n_claims']}")


if __name__ == "__main__":
    main()
