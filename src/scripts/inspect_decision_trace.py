"""Inspect a DecisionTrace from Neo4j or trades CSV/Parquet.

Usage::

    # By decision ID (from Neo4j)
    python -m src.scripts.inspect_decision_trace --decision-id decision:0700.HK:2025-06-01

    # By ticker + date (searches trades_latest first, then Neo4j)
    python -m src.scripts.inspect_decision_trace --ticker 0700.HK --date 2025-06-01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _load_trades_latest() -> list[dict[str, Any]]:
    """Load trades from trades_latest.parquet or trades_latest.csv."""
    out_dir = Path("data/experiments")

    pq_path = out_dir / "trades_latest.parquet"
    if pq_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq_path)
            return df.to_dict("records")
        except Exception:
            pass

    csv_path = out_dir / "trades_latest.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            return df.to_dict("records")
        except Exception:
            pass

    return []


def _find_decision_in_trades(
    ticker: str,
    trade_date: str,
) -> dict[str, Any] | None:
    """Find a kg_dynamic decision in trades_latest."""
    trades = _load_trades_latest()
    for row in trades:
        if (
            str(row.get("system")) == "kg_dynamic"
            and str(row.get("ticker")) == ticker
            and str(row.get("trade_date", "")).startswith(trade_date)
        ):
            return row
    return None


def _try_neo4j_query(decision_id: str) -> dict[str, Any] | None:
    """Try to fetch decision trace from Neo4j."""
    try:
        from src.kg.query import KGQueryClient

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "neo4j")
        database = os.environ.get("NEO4J_DATABASE", "neo4j")

        client = KGQueryClient(uri, user, password, database)
        try:
            return client.get_decision_trace(decision_id)
        finally:
            client.close()
    except Exception as exc:
        print(f"[info] Neo4j unavailable ({exc}), using local data only.")
        return None


def _safe_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def _print_decision_from_trades(row: dict[str, Any]) -> None:
    """Print a decision found in trades data."""
    ticker = row.get("ticker", "?")
    trade_date = row.get("trade_date", "?")

    print("=" * 60)
    print(f"  Decision Trace: {ticker} @ {trade_date}")
    print("=" * 60)

    print(f"\n  Action:           {row.get('action', '?').upper()}")
    print(f"  Final Score:      {row.get('final_score', '?')}")
    print(f"  Confidence:       {row.get('confidence', '?')}")
    print(f"  Conflict Level:   {row.get('conflict_level', '?')}")
    print(f"  System:           {row.get('system', '?')}")

    print(f"\n  {'─' * 40}")
    print("  Roles")
    print(f"  {'─' * 40}")

    supporting = _safe_list(row.get("supporting_roles"))
    opposing = _safe_list(row.get("opposing_roles"))
    print(f"  Supporting: {', '.join(supporting) if supporting else '—'}")
    print(f"  Opposing:   {', '.join(opposing) if opposing else '—'}")

    print(f"\n  {'─' * 40}")
    print("  Evidence & Claims")
    print(f"  {'─' * 40}")

    evidence_refs = _safe_list(row.get("evidence_refs"))
    claim_refs = _safe_list(row.get("claim_refs"))
    print(f"  Evidence count:   {len(evidence_refs)}")
    print(f"  Claim count:      {len(claim_refs)}")
    print(f"  Fresh evidence:   {row.get('fresh_evidence_count', '?')}")
    print(f"  Stale evidence:   {row.get('stale_evidence_count', '?')}")

    if evidence_refs:
        print(f"\n  Top 5 evidence refs:")
        for ref in evidence_refs[:5]:
            print(f"    - {ref}")

    if claim_refs:
        print(f"\n  Top 5 claim refs:")
        for ref in claim_refs[:5]:
            print(f"    - {ref}")

    print(f"\n  {'─' * 40}")
    print("  Reason")
    print(f"  {'─' * 40}")
    reason = str(row.get("decision_reason", ""))
    print(f"  {reason[:500]}")

    # Backtest outcome
    trade_executed = row.get("trade_executed")
    raw_return = row.get("raw_return")
    direction_outcome = row.get("direction_outcome")
    if trade_executed is not None:
        print(f"\n  {'─' * 40}")
        print("  Backtest Outcome")
        print(f"  {'─' * 40}")
        print(f"  Executed:         {trade_executed}")
        print(f"  Raw Return:       {raw_return}")
        print(f"  Direction:        {direction_outcome}")
        print(f"  Entry Date:       {row.get('entry_date', '?')}")
        print(f"  Exit Date:        {row.get('exit_date', '?')}")
        print(f"  Entry Price:      {row.get('entry_price', '?')}")
        print(f"  Exit Price:       {row.get('exit_price', '?')}")

    # Agent reports
    agent_reports = _safe_list(row.get("agent_reports"))
    if agent_reports:
        print(f"\n  {'─' * 40}")
        print("  Agent Reports")
        print(f"  {'─' * 40}")
        for report in agent_reports:
            if isinstance(report, dict):
                role = report.get("role", "?")
                stance = report.get("stance", "?")
                conf = report.get("confidence", "?")
                score = report.get("score", "?")
                print(f"    {role}: stance={stance}, confidence={conf}, score={score}")

    print()


def _print_decision_from_neo4j(trace: dict[str, Any]) -> None:
    """Print a decision trace from Neo4j."""
    decision = trace.get("decision") or {}
    company = trace.get("company") or {}
    assessments = trace.get("assessments") or []
    sources = trace.get("sources") or []
    claims = trace.get("claims") or []
    outcomes = trace.get("outcomes") or []

    ticker = decision.get("ticker") or company.get("ticker", "?")
    trade_date = str(decision.get("trade_date", "?"))[:19]

    print("=" * 60)
    print(f"  Decision Trace: {ticker} @ {trade_date}")
    print("=" * 60)

    print(f"\n  Decision ID:      {decision.get('entity_id', '?')}")
    print(f"  Action:           {str(decision.get('action', '?')).upper()}")
    print(f"  Final Score:      {decision.get('final_score', '?')}")
    print(f"  Confidence:       {decision.get('confidence', '?')}")
    print(f"  Conflict Level:   {decision.get('conflict_level', '?')}")
    print(f"  Evidence Alignment: {decision.get('evidence_alignment', '?')}")

    supporting = decision.get("supporting_roles") or []
    opposing = decision.get("opposing_roles") or []
    print(f"  Supporting Roles: {', '.join(supporting) if supporting else '—'}")
    print(f"  Opposing Roles:   {', '.join(opposing) if opposing else '—'}")

    print(f"  Fresh Evidence:   {decision.get('fresh_evidence_count', '?')}")
    print(f"  Stale Evidence:   {decision.get('stale_evidence_count', '?')}")
    print(f"  Reason:           {str(decision.get('decision_reason', ''))[:300]}")

    # Score breakdown
    trace_json = decision.get("trace_json")
    if isinstance(trace_json, str):
        try:
            trace_json = json.loads(trace_json)
        except Exception:
            trace_json = None
    if isinstance(trace_json, dict) and trace_json:
        print(f"\n  {'─' * 40}")
        print("  Score Breakdown")
        print(f"  {'─' * 40}")
        for key, val in sorted(trace_json.items()):
            label = key.replace("_score", "").replace("_", " ").title()
            print(f"    {label}: {val}")

    # Evidence
    decision_evidence = trace.get("decision_evidence") or []
    if decision_evidence:
        print(f"\n  {'─' * 40}")
        print(f"  Evidence ({len(decision_evidence)} nodes)")
        print(f"  {'─' * 40}")
        for ev in decision_evidence[:10]:
            eid = ev.get("evidence_id") or ev.get("entity_id", "?")
            src = ev.get("source_name") or ev.get("title") or "?"
            conf = ev.get("confidence", "?")
            print(f"    {eid}  src={src}  conf={conf}")
        if len(decision_evidence) > 10:
            print(f"    ... +{len(decision_evidence) - 10} more")

    # Claims
    if claims:
        print(f"\n  {'─' * 40}")
        print(f"  Claims ({len(claims)})")
        print(f"  {'─' * 40}")
        for cl in claims[:10]:
            cid = cl.get("claim_id", "?")
            polarity = cl.get("polarity", "?")
            conf = cl.get("confidence", "?")
            text = str(cl.get("text", ""))[:80]
            print(f"    [{polarity}] {cid}  conf={conf}")
            print(f"      {text}")

    # Sources
    if sources:
        print(f"\n  {'─' * 40}")
        print(f"  Sources ({len(sources)})")
        print(f"  {'─' * 40}")
        for src in sources[:10]:
            name = src.get("source_name") or src.get("title") or "?"
            url = src.get("url", "")
            stype = src.get("source_type", "?")
            print(f"    [{stype}] {name}")
            if url:
                print(f"      {url}")

    # Agent assessments
    if assessments:
        print(f"\n  {'─' * 40}")
        print(f"  Agent Assessments ({len(assessments)})")
        print(f"  {'─' * 40}")
        for aa in assessments:
            role = aa.get("agent_role", "?")
            stance = aa.get("stance", "?")
            conf = aa.get("confidence", "?")
            score = aa.get("score", "?")
            supports = aa.get("supports_decision")
            opposes = aa.get("opposes_decision")
            position = "SUPPORTS" if supports else ("OPPOSES" if opposes else "NEUTRAL")
            print(f"    {role}: stance={stance} score={score} conf={conf} → {position}")
            summary = str(aa.get("summary", ""))[:120]
            if summary:
                print(f"      {summary}")

    # Outcomes
    if outcomes:
        print(f"\n  {'─' * 40}")
        print("  Backtest Outcome")
        print(f"  {'─' * 40}")
        for bo in outcomes:
            print(f"    Action:           {bo.get('action', '?')}")
            print(f"    Raw Return:       {bo.get('raw_return', '?')}")
            print(f"    Direction:        {bo.get('direction_outcome', '?')}")
            print(f"    Trade Executed:   {bo.get('trade_executed', '?')}")
            print(f"    Holding Days:     {bo.get('holding_days', '?')}")
    else:
        print(f"\n  {'─' * 40}")
        print("  Backtest Outcome: not connected")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a DecisionTrace by ID or ticker+date."
    )
    parser.add_argument(
        "--decision-id", default=None,
        help="e.g. decision:0700.HK:2025-06-01",
    )
    parser.add_argument("--ticker", default=None, help="e.g. 0700.HK")
    parser.add_argument("--date", default=None, help="e.g. 2025-06-01")
    args = parser.parse_args()

    load_dotenv()

    if args.decision_id:
        # Try Neo4j first
        trace = _try_neo4j_query(args.decision_id)
        if trace:
            _print_decision_from_neo4j(trace)
            return

        # Fallback: parse decision_id to find in trades
        parts = args.decision_id.split(":")
        if len(parts) >= 3:
            ticker = parts[1]
            date = parts[2]
            row = _find_decision_in_trades(ticker, date)
            if row:
                _print_decision_from_trades(row)
                return

        print(f"Decision '{args.decision_id}' not found in Neo4j or trades data.")
        sys.exit(1)

    elif args.ticker and args.date:
        # Try trades first (offline-friendly)
        row = _find_decision_in_trades(args.ticker, args.date)
        if row:
            _print_decision_from_trades(row)
            return

        # Try Neo4j
        decision_id = f"decision:{args.ticker}:{args.date}"
        trace = _try_neo4j_query(decision_id)
        if trace:
            _print_decision_from_neo4j(trace)
            return

        print(f"No decision found for {args.ticker} @ {args.date}")
        sys.exit(1)

    else:
        print("Please provide --decision-id OR --ticker + --date")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
