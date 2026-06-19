"""Report formatter following TradingAgents-style markdown output.

TradingAgents report structure:
1. Analyst Reports (prose with markdown tables)
   - Market Report (technical)
   - Sentiment Report (scored)
   - News Report
   - Fundamentals Report
2. Investment Debate (bull vs bear)
3. Research Manager → 5-tier rating + rationale + strategic actions
4. Trader → action + reasoning + entry/stop/sizing
5. Risk Debate (aggressive vs conservative vs neutral)
6. Portfolio Manager → final rating + executive summary + investment thesis

We adapt this to FinDynamicGraph's 4-agent output (news, technical, fundamental, risk)
and map to a 5-tier rating scale.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Rating mapping
# ---------------------------------------------------------------------------

# Map FinDynamicGraph 4-level action + score → TradingAgents 5-tier rating
def _score_to_rating(action: str, score: float) -> str:
    """Map action + score to 5-tier portfolio rating."""
    if action == "buy":
        return "Buy" if score >= 0.4 else "Overweight"
    elif action == "sell":
        return "Sell" if score <= -0.4 else "Underweight"
    elif action == "hold":
        return "Hold"
    else:  # abstain
        return "Hold"


def _rating_emoji(rating: str) -> str:
    return {
        "Buy": "🟢🟢", "Overweight": "🟢",
        "Hold": "🟡",
        "Underweight": "🔴", "Sell": "🔴🔴",
    }.get(rating, "⚪")


def _stance_emoji(stance: str) -> str:
    return {
        "bullish": "🟢", "bearish": "🔴",
        "neutral": "🟡", "uncertain": "⚪",
    }.get(stance, "❓")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(ticker: str, trade_date: str, rating: str) -> str:
    emoji = _rating_emoji(rating)
    return "\n".join([
        f"# 📊 {ticker} — Multi-Agent Analysis Report",
        "",
        f"**Date:** {trade_date}",
        f"**Rating:** {emoji} **{rating}**",
        "",
        "---",
        "",
    ])


def _render_analyst_reports(agent_reports: List[Dict[str, Any]]) -> str:
    """Render each agent's report as a named section, TradingAgents style."""
    if not agent_reports:
        return ""

    sections = []
    sections.append("## Analyst Reports\n")

    role_labels = {
        "news": ("📰 News Analyst", "Evaluates recent news and macro events affecting the stock."),
        "technical": ("📈 Technical Analyst", "Analyzes price action, moving averages, and technical indicators."),
        "fundamental": ("💹 Fundamental Analyst", "Assesses company financials, valuation metrics, and intrinsic value."),
        "risk": ("⚠️ Risk Analyst", "Identifies volatility, drawdown, leverage, and liquidity risks."),
    }

    for report in agent_reports:
        role = report.get("role", "unknown")
        stance = report.get("stance", "uncertain")
        confidence = report.get("confidence", 0.0)
        score = report.get("score", 0.0)
        summary = report.get("summary", "")
        factors = report.get("factors", [])
        evidence_refs = report.get("evidence_refs", [])

        label, description = role_labels.get(role, (f"🤖 {role.title()} Agent", ""))

        sections.append(f"### {label}")
        sections.append("")
        if description:
            sections.append(f"*{description}*")
            sections.append("")

        # Signal summary table
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Stance | {_stance_emoji(stance)} **{stance}** |")
        sections.append(f"| Score | {score:+.3f} |")
        sections.append(f"| Confidence | {confidence:.0%} |")
        sections.append("")

        # Narrative summary
        if summary:
            sections.append(f"**Assessment:** {summary}")
            sections.append("")

        # Factors table
        if factors:
            sections.append("**Key Factors:**")
            sections.append("")
            sections.append("| Factor | Direction | Weight |")
            sections.append("|--------|-----------|--------|")
            for f in factors[:10]:
                fname = f.get("name", "")
                fdir = f.get("direction", "")
                fwt = f.get("weight", 0.0)
                sections.append(f"| {fname} | {_stance_emoji(fdir)} {fdir} | {fwt:.3f} |")
            sections.append("")

        # Evidence references
        if evidence_refs:
            sections.append(f"**Evidence sources:** {len(evidence_refs)} references")
            sections.append("")

        sections.append("---")
        sections.append("")

    return "\n".join(sections)


def _render_investment_plan(decision: Dict[str, Any], rating: str) -> str:
    """Render the Research Manager / Portfolio Manager decision section."""
    final_score = decision.get("final_score", 0.0)
    confidence = decision.get("confidence", 0.0)
    conflict = decision.get("conflict_level", 0.0)
    reason = decision.get("decision_reason", "")
    supporting = decision.get("supporting_roles", [])
    opposing = decision.get("opposing_roles", [])
    evidence_alignment = decision.get("evidence_alignment", "unknown")
    trace = decision.get("trace", {})

    sections = []
    sections.append("## Investment Decision\n")

    # Rating with scale explanation
    sections.append(f"### Final Rating: {_rating_emoji(rating)} **{rating}**")
    sections.append("")
    sections.append("**Rating Scale:** Buy > Overweight > Hold > Underweight > Sell")
    sections.append("")

    # Decision metrics table
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Weighted Score | {final_score:+.4f} |")
    sections.append(f"| Confidence | {confidence:.0%} |")
    sections.append(f"| Conflict Level | {conflict:.0%} |")
    sections.append(f"| Evidence Alignment | {evidence_alignment} |")
    sections.append("")

    # Score breakdown
    if trace:
        sections.append("**Score Breakdown:**")
        sections.append("")
        sections.append("| Agent | Weighted Contribution |")
        sections.append("|-------|----------------------|")
        for key in ("technical_score", "news_score", "fundamental_score", "risk_score"):
            if key in trace:
                label = key.replace("_score", "").title()
                sections.append(f"| {label} | {trace[key]:+.4f} |")
        sections.append("")

    # Supporting / opposing
    if supporting:
        sections.append(f"**Supporting agents:** {', '.join(supporting)}")
    if opposing:
        sections.append(f"**Opposing agents:** {', '.join(opposing)}")
    if supporting or opposing:
        sections.append("")

    # Rationale
    sections.append("### Rationale")
    sections.append("")
    sections.append(reason)
    sections.append("")

    # Strategic actions (mapped from action)
    action = decision.get("action", "abstain")
    sections.append("### Strategic Actions")
    sections.append("")
    if action == "buy":
        sections.append("- Consider entering or adding to the position")
        sections.append("- Size position according to confidence level and risk tolerance")
        sections.append("- Monitor key risk signals for potential exit triggers")
    elif action == "sell":
        sections.append("- Consider exiting or reducing the position")
        sections.append("- Review stop-loss levels and downside protection")
        sections.append("- Reassess if conflicting signals resolve in bull's favor")
    elif action == "hold":
        sections.append("- Maintain current position, no new action required")
        sections.append("- Watch for signal convergence before adjusting exposure")
    else:
        sections.append("- Insufficient evidence for a directional trade")
        sections.append("- Wait for stronger signals or additional data")
    sections.append("")

    return "\n".join(sections)


def _render_risk_assessment(decision: Dict[str, Any]) -> str:
    """Render a dedicated risk section from the risk agent data."""
    agent_reports = decision.get("agent_reports", [])
    risk_report = next((r for r in agent_reports if r.get("role") == "risk"), None)

    if not risk_report:
        return ""

    sections = []
    sections.append("## Risk Assessment\n")

    stance = risk_report.get("stance", "uncertain")
    summary = risk_report.get("summary", "")
    factors = risk_report.get("factors", [])

    sections.append(f"**Overall Risk Stance:** {_stance_emoji(stance)} **{stance}**")
    sections.append("")

    if summary:
        sections.append(summary)
        sections.append("")

    if factors:
        sections.append("| Risk Factor | Direction | Severity |")
        sections.append("|-------------|-----------|----------|")
        for f in factors[:10]:
            fname = f.get("name", "")
            fdir = f.get("direction", "")
            fwt = f.get("weight", 0.0)
            severity = "High" if fwt >= 0.6 else "Medium" if fwt >= 0.3 else "Low"
            sections.append(f"| {fname} | {_stance_emoji(fdir)} {fdir} | {severity} ({fwt:.2f}) |")
        sections.append("")

    sections.append("---")
    sections.append("")
    return "\n".join(sections)


def _render_backtest(backtest: Dict[str, Any]) -> str:
    """Render backtest outcome section."""
    if not backtest or not backtest.get("trade_executed"):
        return ""

    raw_ret = backtest.get("raw_return", 0.0)
    entry = backtest.get("entry_price", "?")
    exit_ = backtest.get("exit_price", "?")
    days = backtest.get("holding_days", 0)
    entry_date = backtest.get("entry_date", "?")
    exit_date = backtest.get("exit_date", "?")

    outcome_emoji = "🟢" if raw_ret > 0 else "🔴" if raw_ret < 0 else "🟡"

    sections = []
    sections.append("## Backtest Reference\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Return | {outcome_emoji} {raw_ret:+.2%} |")
    sections.append(f"| Entry | {entry} ({entry_date}) |")
    sections.append(f"| Exit | {exit_} ({exit_date}) |")
    sections.append(f"| Holding Period | {days} days |")
    sections.append("")
    sections.append("---")
    sections.append("")
    return "\n".join(sections)


def _render_data_summary(collect_summary: Dict[str, Any]) -> str:
    """Render data collection summary."""
    if not collect_summary:
        return ""

    ticker = collect_summary.get("ticker", "")
    ohlcv = collect_summary.get("ohlcv_rows", 0)
    news = collect_summary.get("news_rows", 0)
    fund = collect_summary.get("fundamental_rows", 0)

    sections = []
    sections.append("## Data Sources\n")
    sections.append(f"| Source | Records |")
    sections.append(f"|--------|---------|")
    sections.append(f"| OHLCV (price) | {ohlcv} |")
    sections.append(f"| News articles | {news} |")
    sections.append(f"| Fundamental metrics | {fund} |")
    sections.append("")
    sections.append("---")
    sections.append("")
    return "\n".join(sections)


def _render_footer() -> str:
    return "\n".join([
        "---",
        "",
        "*⚠️ This analysis is generated by a multi-agent AI system and is for research "
        "purposes only. It does not constitute financial, investment, or trading advice. "
        "Always conduct your own due diligence before making investment decisions.*",
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def format_report(
    decision: Dict[str, Any],
    collect_summary: Dict[str, Any] | None = None,
    backtest: Dict[str, Any] | None = None,
) -> str:
    """Format decision output as a TradingAgents-style markdown report.

    Sections:
    1. Header (ticker, date, final rating)
    2. Data Sources (collection summary)
    3. Analyst Reports (each agent's section with tables)
    4. Risk Assessment (dedicated risk section)
    5. Investment Decision (rating, rationale, strategic actions)
    6. Backtest Reference (optional)
    7. Footer (disclaimer)
    """
    ticker = decision.get("ticker", "???")
    trade_date = decision.get("trade_date", "???")
    action = decision.get("action", "abstain")
    final_score = decision.get("final_score", 0.0)
    rating = _score_to_rating(action, final_score)

    parts = [
        _render_header(ticker, trade_date, rating),
        _render_data_summary(collect_summary or {}),
        _render_analyst_reports(decision.get("agent_reports", [])),
        _render_risk_assessment(decision),
        _render_investment_plan(decision, rating),
        _render_backtest(backtest or {}),
        _render_footer(),
    ]

    return "\n".join(p for p in parts if p)
