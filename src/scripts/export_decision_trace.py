"""Export a DecisionTrace + full provenance chain for dissertation case study.

Usage:
    python -m src.scripts.export_decision_trace \
        --decision-id decision:0700.HK:2025-03-01 \
        --output data/experiments/trace_0700_2025-03-01.json \
        --markdown-output data/experiments/trace_0700_2025-03-01.md

The Markdown output renders the full provenance chain:
    SourceDocument → Evidence → Claim → AgentAssessment → DecisionTrace → BacktestOutcome
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

from src.kg.query import KGQueryClient  # noqa: E402


# ── JSON export ────────────────────────────────────────────────────────

def export_trace_json(
    decision_id: str,
    output_path: str | Path,
    *,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str = "neo4j",
) -> dict:
    """Fetch decision trace and write raw JSON. Returns the trace dict."""

    uri = uri or os.environ["NEO4J_URI"]
    user = user or os.environ.get("NEO4J_USER", "neo4j")
    password = password or os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", database)

    client = KGQueryClient(uri, user, password, database)
    try:
        trace = client.get_decision_trace(decision_id)
    finally:
        client.close()

    if trace is None:
        print(f"⚠️  DecisionTrace '{decision_id}' not found in Neo4j.")
        sys.exit(1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)

    return trace


# ── Markdown export ────────────────────────────────────────────────────

def _fmt_date(val: Any) -> str:
    if val is None:
        return "—"
    return str(val)[:19]


def _truncate(text: str, max_len: int = 200) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def _safe_json_parse(val: Any) -> Any:
    """Parse a JSON string, or return the value if already parsed."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


def _render_markdown(trace: dict) -> str:
    """Render the full provenance chain as Markdown.

    Handles both fully-connected and partially-connected graphs gracefully.
    When the formal evidence chain is missing, falls back to extracting
    info from factors_json, trace_json, and evidence_refs.
    """

    lines: list[str] = []
    lines.append("# Decision Trace Case Study\n")

    decision = trace.get("decision") or {}
    company = trace.get("company") or {}
    assessments = trace.get("assessments") or []
    assessment_evidence = trace.get("assessment_evidence") or []
    decision_evidence = trace.get("decision_evidence") or []
    sources = trace.get("sources") or []
    claims = trace.get("claims") or []
    outcomes = trace.get("outcomes") or []

    ticker = decision.get("ticker") or company.get("ticker", "?")
    trade_date = _fmt_date(decision.get("trade_date"))

    # ── 1. Decision Overview ──
    lines.append("## 1. Decision Overview\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| **Decision ID** | `{decision.get('entity_id', '?')}` |")
    lines.append(f"| **Ticker** | {ticker} |")
    lines.append(f"| **Trade Date** | {trade_date} |")
    lines.append(f"| **Action** | **{decision.get('action', '?').upper()}** |")
    lines.append(f"| **Final Score** | {decision.get('final_score', '?')} |")
    lines.append(f"| **Confidence** | {decision.get('confidence', '?')} |")
    lines.append(f"| **Conflict Level** | {decision.get('conflict_level', '?')} |")
    lines.append(f"| **Evidence Alignment** | {decision.get('evidence_alignment', '?')} |")

    supporting = decision.get("supporting_roles") or []
    opposing = decision.get("opposing_roles") or []
    if supporting or opposing:
        lines.append(f"| **Supporting Roles** | {', '.join(supporting) or '—'} |")
        lines.append(f"| **Opposing Roles** | {', '.join(opposing) or '—'} |")

    lines.append(f"| **Fresh Evidence** | {decision.get('fresh_evidence_count', '?')} |")
    lines.append(f"| **Stale Evidence** | {decision.get('stale_evidence_count', '?')} |")
    lines.append(f"| **Reason** | {_truncate(decision.get('decision_reason', ''), 300)} |")
    lines.append("")

    # ── 2. Score Breakdown (from trace_json) ──
    trace_data = _safe_json_parse(decision.get("trace_json")) or {}
    if isinstance(trace_data, dict) and trace_data:
        lines.append("## 2. Score Breakdown\n")
        lines.append("| Agent | Score |")
        lines.append("|---|---|")
        for key in sorted(trace_data.keys()):
            val = trace_data[key]
            label = key.replace("_score", "").replace("_", " ").title()
            lines.append(f"| {label} | {val} |")
        lines.append("")

    # ── 3. Provenance Chain ──
    lines.append("## 3. Provenance Chain\n")
    lines.append("```")
    lines.append("SourceDocument → Evidence → Claim → AgentAssessment → DecisionTrace → BacktestOutcome")
    lines.append("```")

    has_full_chain = bool(sources or decision_evidence or claims)
    if not has_full_chain:
        lines.append("")
        lines.append("> **Note:** The formal evidence chain (SourceDocument → Evidence → Claim) "
                     "is not yet connected in Neo4j for this trace. "
                     "Evidence references below are signal IDs from the graph.\n")

    # ── 4. Source Documents ──
    if sources:
        lines.append("### 3.1 Source Documents\n")
        lines.append("| # | ID | Type | Name | Published |")
        lines.append("|---|---|---|---|---|")
        for i, src in enumerate(sources[:15], 1):
            lines.append(
                f"| {i} | `{src.get('entity_id', '?')[:40]}` "
                f"| {src.get('source_type', '?')} "
                f"| {src.get('source_name') or src.get('title') or '?'} "
                f"| {_fmt_date(src.get('published_at'))} |"
            )
        lines.append("")

    # ── 5. Evidence (from graph or from assessments) ──
    if decision_evidence:
        # Detect if evidence is IndicatorSignal type
        is_signal = any(ev.get("entity_type") == "IndicatorSignal" or ev.get("signal_type") for ev in decision_evidence[:1])
        if is_signal:
            lines.append("### 3.2 Evidence (IndicatorSignals)\n")
            lines.append("| # | Signal ID | Direction | Strength | Price | MA20 | %Diff | Date |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for i, ev in enumerate(decision_evidence[:20], 1):
                eid = ev.get("entity_id") or ev.get("evidence_id") or "?"
                direction = ev.get("direction", "?")
                strength = ev.get("strength", "?")
                price = ev.get("price", "?")
                ma20 = ev.get("ma20", "?")
                pct = ev.get("pct_diff", "?")
                date = _fmt_date(ev.get("as_of_date"))
                if isinstance(price, float): price = f"{price:.2f}"
                if isinstance(ma20, float): ma20 = f"{ma20:.2f}"
                if isinstance(pct, float): pct = f"{pct:+.4f}"
                if isinstance(strength, float): strength = f"{strength:.3f}"
                lines.append(
                    f"| {i} | `{eid}` "
                    f"| {direction} "
                    f"| {strength} "
                    f"| {price} "
                    f"| {ma20} "
                    f"| {pct} "
                    f"| {date} |"
                )
            if len(decision_evidence) > 20:
                lines.append(f"| … | *{len(decision_evidence) - 20} more* | | | | | | |")
            lines.append("")
        else:
            lines.append("### 3.2 Evidence\n")
            lines.append("| # | Evidence ID | Source | Confidence | Text |")
            lines.append("|---|---|---|---|---|")
            for i, ev in enumerate(decision_evidence[:15], 1):
                eid = ev.get("evidence_id") or ev.get("entity_id") or "?"
                src = ev.get("source_name") or ev.get("title") or ev.get("signal_type") or "?"
                conf = ev.get("confidence", "?")
                text = ev.get("extracted_text") or ev.get("description") or ev.get("summary") or ""
                lines.append(
                    f"| {i} | `{eid}` "
                    f"| {src} "
                    f"| {conf} "
                    f"| {_truncate(text, 100)} |"
                )
            lines.append("")
    else:
        # Fallback: collect unique evidence_refs from all assessments
        all_refs: list[str] = []
        seen: set[str] = set()
        for aa in assessments:
            for ref in (aa.get("evidence_refs") or []):
                if ref not in seen:
                    seen.add(ref)
                    all_refs.append(ref)
        if all_refs:
            lines.append("### 3.2 Evidence References (from Assessments)\n")
            lines.append(f"Total unique references: **{len(all_refs)}**\n")
            lines.append("| # | Signal ID |")
            lines.append("|---|---|")
            for i, ref in enumerate(all_refs[:20], 1):
                lines.append(f"| {i} | `{ref}` |")
            if len(all_refs) > 20:
                lines.append(f"| … | *{len(all_refs) - 20} more* |")
            lines.append("")

    # ── 6. Claims ──
    if claims:
        lines.append("### 3.3 Claims\n")
        lines.append("| # | Claim ID | Polarity | Confidence | Text |")
        lines.append("|---|---|---|---|---|")
        for i, cl in enumerate(claims[:15], 1):
            lines.append(
                f"| {i} | `{cl.get('claim_id', '?')}` "
                f"| {cl.get('polarity', '?')} "
                f"| {cl.get('confidence', '?')} "
                f"| {_truncate(cl.get('text', ''), 120)} |"
            )
        lines.append("")

    # ── 7. Agent Assessments ──
    if assessments:
        lines.append("### 3.4 Agent Assessments\n")
        for aa in assessments:
            role = aa.get("agent_role", "?")
            stance = aa.get("stance", "?")
            conf = aa.get("confidence", "?")
            score = aa.get("score", "?")
            summary = aa.get("summary", "")
            supports = aa.get("supports_decision")
            opposes = aa.get("opposes_decision")

            position = "SUPPORTS" if supports else ("OPPOSES" if opposes else "NEUTRAL")
            lines.append(f"#### {role.title()} Agent\n")
            lines.append(f"| Field | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| **Stance** | `{stance}` |")
            lines.append(f"| **Score** | {score} |")
            lines.append(f"| **Confidence** | {conf} |")
            lines.append(f"| **Position** | `{position}` |")
            lines.append(f"")
            lines.append(f"> {_truncate(summary, 300)}\n")

            # Factors from factors_json
            factors = _safe_json_parse(aa.get("factors_json")) or aa.get("factors") or []
            if isinstance(factors, list) and factors:
                lines.append("| Factor | Direction | Weight |")
                lines.append("|---|---|---|")
                for f in factors[:15]:
                    lines.append(
                        f"| {f.get('name', '?')} "
                        f"| {f.get('direction', '?')} "
                        f"| {f.get('weight', '?')} |"
                    )
                if len(factors) > 15:
                    lines.append(f"| … | *{len(factors) - 15} more* | |")
                lines.append("")

    # ── 8. Backtest Outcome ──
    if outcomes:
        lines.append("## 4. Backtest Outcome\n")
        for bo in outcomes:
            lines.append("| Field | Value |")
            lines.append("|---|---|")
            lines.append(f"| **Outcome ID** | `{bo.get('entity_id', '?')}` |")
            lines.append(f"| **Action** | {bo.get('action', '?')} |")
            lines.append(f"| **Raw Return** | {bo.get('raw_return', '?')} |")
            lines.append(f"| **Direction Outcome** | {bo.get('direction_outcome', '?')} |")
            lines.append(f"| **Trade Executed** | {bo.get('trade_executed', '?')} |")
            lines.append(f"| **Holding Days** | {bo.get('holding_days', '?')} |")
            lines.append(f"| **Evaluated At** | {_fmt_date(bo.get('evaluated_at'))} |")
            lines.append("")
    else:
        lines.append("## 4. Backtest Outcome\n")
        lines.append("> No BacktestOutcome connected for this trace.\n")

    # ── 9. Full JSON Reference ──
    lines.append("## 5. Raw JSON\n")
    lines.append("<details><summary>Click to expand full JSON</summary>\n")
    lines.append("```json")
    lines.append(json.dumps(trace, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    lines.append("\n</details>\n")

    return "\n".join(lines)


def export_trace_markdown(trace: dict, md_path: str | Path) -> None:
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(trace))


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DecisionTrace case study (JSON + optional Markdown)"
    )
    parser.add_argument(
        "--decision-id", required=True,
        help="e.g. decision:0700.HK:2025-03-01",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--markdown-output", default=None,
        help="Optional: also export a Markdown case study",
    )
    args = parser.parse_args()

    load_dotenv()

    trace = export_trace_json(args.decision_id, args.output)
    print(f"✅ JSON exported → {args.output}")

    if args.markdown_output:
        export_trace_markdown(trace, args.markdown_output)
        print(f"✅ Markdown exported → {args.markdown_output}")


if __name__ == "__main__":
    main()
