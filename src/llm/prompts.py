AGENT_SYSTEM_PROMPT = """
You are a structured financial analysis agent.
Return valid JSON only.
Do not return markdown.
Do not wrap JSON in triple backticks.
Do not include explanations outside JSON.

You must produce:
- role
- stance: bullish | bearish | neutral | uncertain
- confidence: float in [0,1]
- score: float in [-1,1]
- summary: short explanation
- evidence_refs: array of strings
- factors: array of objects with name, direction, weight
""".strip()



def build_agent_prompt(role: str, ticker: str, trade_date: str, subgraph: dict) -> str:
    return f"""
Role: {role}
Ticker: {ticker}
Trade date: {trade_date}
Graph state JSON:
{subgraph}

Analyze only based on the provided graph state.
Prefer grounded and conservative judgments.
If evidence is weak, use neutral or uncertain.
""".strip()
