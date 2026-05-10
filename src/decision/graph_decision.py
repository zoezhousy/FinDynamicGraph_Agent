# graph decision engine to generate buy / sell / hold / abstain decisions based on the graph signals
from src.kg.schema import TradingDecision


class GraphDecisionEngine:
    def decide(self, ticker: str, signals: list[dict]) -> TradingDecision:
        bullish_score = 0.0
        bearish_score = 0.0
        neutral_score = 0.0
        evidence_ids = []

        for signal in signals:
            direction = signal.get("direction")
            strength = float(signal.get("strength", 0.0))
            evidence_id = signal.get("evidence_id")

            if evidence_id:
                evidence_ids.append(evidence_id)

            if direction == "bullish":
                bullish_score += strength
            elif direction == "bearish":
                bearish_score += strength
            else:
                neutral_score += strength

        if not signals:
            action = "abstain"
            reason = "No graph evidence is available."
        elif bullish_score - bearish_score > 0.5:
            action = "buy"
            reason = "Bullish graph evidence is stronger than bearish evidence."
        elif bearish_score - bullish_score > 0.5:
            action = "sell"
            reason = "Bearish graph evidence is stronger than bullish evidence."
        elif abs(bullish_score - bearish_score) <= 0.3:
            action = "hold"
            reason = "Bullish and bearish graph evidence are balanced."
        else:
            action = "abstain"
            reason = "Graph evidence is conflicting or insufficient."

        return TradingDecision(
            ticker=ticker,
            action=action,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            evidence_ids=evidence_ids,
            reason=reason,
        )