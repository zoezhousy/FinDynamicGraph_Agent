"""Identity Anonymization Probe.

Re-runs kg_dynamic agent decisions on the same subgraphs but with
company name / ticker replaced by fictional identities.  Compares
real vs anonymized decisions to detect whether the agents rely on
recognizable brand names or on the structural graph evidence alone.

Usage:
    python -m src.probes.identity_probe              # default 5 cases
    python -m src.probes.identity_probe --n 3        # fewer cases
    python -m src.probes.identity_probe --cases 0700.HK:2025-06-01,0005.HK:2025-03-01

Requires: Neo4j running with experiment data loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# ── Anonymization mapping ──────────────────────────────────────────────

ANON_MAP: Dict[str, Dict[str, str]] = {
    "0005.HK": {"ticker": "ALPHA.HK", "name": "AlphaCorp"},
    "0700.HK": {"ticker": "BETA.HK",  "name": "BetaTech"},
    "1299.HK": {"ticker": "GAMMA.HK", "name": "GammaHoldings"},
}

# Real company names (used for text replacement in anonymization)
REAL_NAMES: Dict[str, str] = {
    "0005.HK": "HSBC",
    "0700.HK": "Tencent",
    "1299.HK": "AIA",
}

# Pre-compiled regex for each real name (case-insensitive, whole word)
_NAME_PATTERNS = {
    ticker: re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
    for ticker, name in REAL_NAMES.items()
}


def anonymize_subgraph(
    subgraph: Dict[str, List[Dict[str, Any]]],
    real_ticker: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deep-copy *subgraph* and replace ticker / company name with fictional identity.

    Numeric values, dates, entity_ids, and graph structure are preserved.
    Only human-readable text fields (company name, news headlines, signal
    names that embed the company name) are rewritten.
    """
    anon = ANON_MAP.get(real_ticker)
    if anon is None:
        raise ValueError(f"No anonymization mapping for {real_ticker}")

    sg = copy.deepcopy(subgraph)

    # ── company node ───────────────────────────────────────────────────
    for co in sg.get("company", []):
        if co.get("ticker") == real_ticker:
            co["ticker"] = anon["ticker"]
        if "name" in co:
            co["name"] = _replace_name(str(co["name"]), real_ticker)

    # ── ticker field on every node ─────────────────────────────────────
    for _key, items in sg.items():
        if not isinstance(items, list):
            continue
        for node in items:
            if isinstance(node, dict) and node.get("ticker") == real_ticker:
                node["ticker"] = anon["ticker"]

    # ── text fields that might embed the company name ──────────────────
    _text_keys = {"headline", "title", "summary", "description", "name", "text"}
    for _key, items in sg.items():
        if not isinstance(items, list):
            continue
        for node in items:
            if not isinstance(node, dict):
                continue
            for tkey in _text_keys:
                if tkey in node and isinstance(node[tkey], str):
                    node[tkey] = _replace_name(node[tkey], real_ticker)

    return sg


def _replace_name(text: str, real_ticker: str) -> str:
    """Replace the real company name in *text* with the anonymized version."""
    anon = ANON_MAP[real_ticker]
    pat = _NAME_PATTERNS.get(real_ticker)
    if pat:
        return pat.sub(anon["name"], text)
    return text


# ── Agent execution ────────────────────────────────────────────────────


def run_decision(
    ticker: str,
    trade_date: datetime,
    subgraph: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Run the full 4-agent pipeline on a subgraph and return the decision dict.

    This is a self-contained re-execution; it does NOT touch Neo4j or the
    formal experiment path in ``src.agents.roles``.
    """
    from src.agents.roles import (
        AgentReport,
        news_agent,
        portfolio_manager_decide,
        technical_agent,
    )
    from src.agents.fundamental_agent import FundamentalAgent
    from src.agents.risk_agent import RiskAgent

    reports: List[AgentReport] = [
        technical_agent(subgraph, ticker, trade_date),
        news_agent(subgraph, ticker, trade_date),
        FundamentalAgent().run({"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}),
        RiskAgent().run({"ticker": ticker, "trade_date": trade_date, "subgraph": subgraph}),
    ]

    decision = portfolio_manager_decide(ticker, trade_date, reports, subgraph=subgraph)

    # Attach per-agent summaries for diff reporting
    decision["_agent_reports"] = [
        {
            "role": r.role,
            "stance": r.stance,
            "confidence": r.confidence,
            "score": r.score,
            "summary": r.summary[:300],
        }
        for r in reports
    ]
    return decision


# ── Diff logic ─────────────────────────────────────────────────────────


def diff_decisions(
    real: Dict[str, Any],
    anon: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two decisions and produce a structured diff."""
    diffs: List[Dict[str, Any]] = []

    # Top-level fields
    for field in ("action", "final_score", "confidence", "conflict_level"):
        rv = real.get(field)
        av = anon.get(field)
        if _approx_differs(rv, av):
            diffs.append({"field": field, "real": rv, "anonymized": av})

    # Per-agent
    real_agents = {r["role"]: r for r in real.get("_agent_reports", [])}
    anon_agents = {r["role"]: r for r in anon.get("_agent_reports", [])}

    agent_diffs: List[Dict[str, Any]] = []
    for role in sorted(set(real_agents) | set(anon_agents)):
        ra = real_agents.get(role, {})
        aa = anon_agents.get(role, {})
        ad: Dict[str, Any] = {"role": role}

        if ra.get("stance") != aa.get("stance"):
            ad["stance_real"] = ra.get("stance")
            ad["stance_anon"] = aa.get("stance")

        if _approx_differs(ra.get("confidence"), aa.get("confidence")):
            ad["confidence_real"] = ra.get("confidence")
            ad["confidence_anon"] = aa.get("confidence")

        if _approx_differs(ra.get("score"), aa.get("score")):
            ad["score_real"] = ra.get("score")
            ad["score_anon"] = aa.get("score")

        if len(ad) > 1:  # more than just "role"
            agent_diffs.append(ad)

    return {
        "action_real": real.get("action"),
        "action_anon": anon.get("action"),
        "top_level_diffs": diffs,
        "agent_diffs": agent_diffs,
        "identical": len(diffs) == 0 and len(agent_diffs) == 0,
    }


def _approx_differs(a: Any, b: Any, tol: float = 0.005) -> bool:
    """Check if two values differ (numeric tolerance for floats)."""
    if a is None and b is None:
        return False
    try:
        return abs(float(a) - float(b)) > tol
    except (TypeError, ValueError):
        return str(a) != str(b)


# ── Probe runner ───────────────────────────────────────────────────────


def run_probe(
    cases: List[tuple[str, str]],
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> List[Dict[str, Any]]:
    """Run the identity probe on a list of (ticker, date_str) cases."""
    from src.kg.query import KGQueryClient

    client = KGQueryClient(neo4j_uri, neo4j_user, neo4j_password)
    results: List[Dict[str, Any]] = []

    try:
        for ticker, date_str in cases:
            trade_dt = datetime.fromisoformat(date_str)
            print(f"\n{'='*60}")
            print(f"Case: {ticker} @ {date_str}")
            print(f"{'='*60}")

            # Fetch real subgraph
            subgraph = client.get_ticker_subgraph(ticker, trade_dt)
            n_nodes = sum(len(v) for v in subgraph.values() if isinstance(v, list))
            print(f"  Subgraph: {n_nodes} nodes across {len(subgraph)} keys")

            if n_nodes == 0:
                print(f"  SKIP — empty subgraph")
                results.append({
                    "ticker": ticker,
                    "trade_date": date_str,
                    "status": "skipped_empty_subgraph",
                })
                continue

            # Run real decision
            print("  Running real decision...")
            real_decision = run_decision(ticker, trade_dt, subgraph)
            print(f"    action={real_decision['action']}  "
                  f"confidence={real_decision['confidence']:.3f}  "
                  f"score={real_decision['final_score']:+.4f}")

            # Anonymize and run
            anon_subgraph = anonymize_subgraph(subgraph, ticker)
            anon_ticker = ANON_MAP[ticker]["ticker"]
            print(f"  Running anonymized decision ({anon_ticker})...")
            anon_decision = run_decision(anon_ticker, trade_dt, anon_subgraph)
            print(f"    action={anon_decision['action']}  "
                  f"confidence={anon_decision['confidence']:.3f}  "
                  f"score={anon_decision['final_score']:+.4f}")

            # Diff
            diff = diff_decisions(real_decision, anon_decision)
            if diff["identical"]:
                print("  ✓ Identical decisions")
            else:
                print("  ✗ Differences found:")
                for d in diff["top_level_diffs"]:
                    print(f"    {d['field']}: {d['real']} → {d['anonymized']}")
                for ad in diff["agent_diffs"]:
                    parts = [f"{k}: {ad[k]}" for k in ad if k != "role"]
                    print(f"    [{ad['role']}] {', '.join(parts)}")

            results.append({
                "ticker": ticker,
                "trade_date": date_str,
                "anon_ticker": anon_ticker,
                "status": "ok",
                "subgraph_nodes": n_nodes,
                "real_action": real_decision["action"],
                "real_score": real_decision["final_score"],
                "real_confidence": real_decision["confidence"],
                "anon_action": anon_decision["action"],
                "anon_score": anon_decision["final_score"],
                "anon_confidence": anon_decision["confidence"],
                "diff": diff,
            })
    finally:
        client.close()

    return results


# ── Report generation ──────────────────────────────────────────────────


def generate_report(results: List[Dict[str, Any]], out_path: Path) -> None:
    """Write a human-readable markdown diff report."""
    lines = [
        "# Identity Anonymization Probe Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Purpose",
        "",
        "Test whether agent decisions change when real company names / tickers",
        "are replaced with fictional identities, while keeping all graph",
        "structure, numbers, dates, and relationships identical.",
        "",
        "## Anonymization Mapping",
        "",
        "| Real Ticker | Real Name | Anon Ticker | Anon Name |",
        "|---|---|---|---|",
    ]

    # Reverse lookup for names
    _name_lookup = {
        "0005.HK": "HSBC Holdings",
        "0700.HK": "Tencent Holdings",
        "1299.HK": "AIA Group",
    }
    for real, info in ANON_MAP.items():
        lines.append(
            f"| {real} | {_name_lookup.get(real, 'N/A')} "
            f"| {info['ticker']} | {info['name']} |"
        )

    lines.extend(["", "## Results", ""])

    ok_results = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") != "ok"]

    if skipped:
        lines.append(f"**Skipped:** {len(skipped)} case(s) — empty subgraph.\n")

    n_identical = sum(1 for r in ok_results if r["diff"]["identical"])
    n_different = len(ok_results) - n_identical

    lines.append(f"**Total cases:** {len(ok_results)}  ")
    lines.append(f"**Identical decisions:** {n_identical}  ")
    lines.append(f"**Different decisions:** {n_different}  ")
    lines.append("")

    for r in ok_results:
        d = r["diff"]
        lines.append(f"### {r['ticker']} @ {r['trade_date']}")
        lines.append("")

        lines.append("| Metric | Real | Anonymized | Δ |")
        lines.append("|---|---|---|---|")

        def _delta(a, b):
            try:
                return f"{float(b) - float(a):+.4f}"
            except (TypeError, ValueError):
                return "changed" if a != b else "—"

        lines.append(
            f"| Action | {r['real_action']} | {r['anon_action']} "
            f"| {'**changed**' if r['real_action'] != r['anon_action'] else '—'} |"
        )
        lines.append(
            f"| Final score | {r['real_score']:.4f} | {r['anon_score']:.4f} "
            f"| {_delta(r['real_score'], r['anon_score'])} |"
        )
        lines.append(
            f"| Confidence | {r['real_confidence']:.4f} | {r['anon_confidence']:.4f} "
            f"| {_delta(r['real_confidence'], r['anon_confidence'])} |"
        )

        # Per-agent diffs
        if d["agent_diffs"]:
            lines.append("")
            lines.append("**Agent-level differences:**")
            lines.append("")
            lines.append("| Agent | Field | Real | Anonymized |")
            lines.append("|---|---|---|---|")
            for ad in d["agent_diffs"]:
                for key in ("stance", "confidence", "score"):
                    rk = f"{key}_real"
                    ak = f"{key}_anon"
                    if rk in ad:
                        lines.append(
                            f"| {ad['role']} | {key} | {ad[rk]} | {ad[ak]} |"
                        )

        if d["identical"]:
            lines.append("")
            lines.append("✓ **Identical** — anonymization had no effect on this case.")

        lines.append("")

    # Summary
    lines.extend([
        "## Interpretation",
        "",
        "- **Identical decisions** across real/anonymized suggest the agents",
        "  rely on structural graph evidence (signals, fundamentals, risks)",
        "  rather than brand recognition.",
        "- **Different decisions** suggest the agents are influenced by",
        "  company identity — potentially via news headlines or the ticker",
        "  string itself leaking into the LLM prompt.",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────

# Default test cases: variety of tickers, dates, and actions
DEFAULT_CASES = [
    ("0700.HK", "2025-03-01"),   # hold, high confidence
    ("0700.HK", "2025-08-01"),   # buy
    ("0005.HK", "2025-03-01"),   # buy
    ("0005.HK", "2025-08-01"),   # buy
    ("1299.HK", "2025-06-01"),   # variety
]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Identity anonymization probe")
    parser.add_argument("--n", type=int, default=5, help="Number of default cases")
    parser.add_argument(
        "--cases", type=str, default=None,
        help="Comma-separated TICKER:DATE pairs, e.g. 0700.HK:2025-06-01,0005.HK:2025-03-01",
    )
    parser.add_argument("--out", type=str, default="data/probes/identity_probe_report.md")
    args = parser.parse_args()

    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if not (neo4j_uri and neo4j_user and neo4j_password):
        print("ERROR: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD must be set.", file=sys.stderr)
        sys.exit(1)

    # Parse cases
    if args.cases:
        cases = []
        for pair in args.cases.split(","):
            ticker, date_str = pair.strip().split(":")
            cases.append((ticker.strip(), date_str.strip()))
    else:
        cases = DEFAULT_CASES[: args.n]

    print(f"Running identity probe on {len(cases)} case(s)...")
    results = run_probe(cases, neo4j_uri, neo4j_user, neo4j_password)

    # Save JSON for programmatic access
    json_path = Path(args.out).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip agent reports from JSON (too verbose)
    json_results = []
    for r in results:
        rc = dict(r)
        if "diff" in rc:
            rc["diff"] = dict(rc["diff"])
        json_results.append(rc)
    json_path.write_text(json.dumps(json_results, indent=2, default=str), encoding="utf-8")

    # Generate markdown report
    generate_report(results, Path(args.out))


if __name__ == "__main__":
    main()
