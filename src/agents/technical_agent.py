# technical agent to generate technical signals for a given ticker

from datetime import datetime, timezone
from uuid import uuid4

from src.agents.base_agent import BaseAgent
from src.kg.schema import Evidence, FinancialSignal, GraphUpdate


class TechnicalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("technical_agent")

    def run(self, input_data: dict) -> GraphUpdate:
        ticker = input_data["ticker"]
        rsi = input_data.get("rsi")
        ma_short = input_data.get("ma_short")
        ma_long = input_data.get("ma_long")

        direction = "neutral"
        strength = 0.5

        if rsi is not None and rsi < 30:
            direction = "bullish"
            strength = 0.7
        elif rsi is not None and rsi > 70:
            direction = "bearish"
            strength = 0.7

        if ma_short is not None and ma_long is not None:
            if ma_short > ma_long:
                direction = "bullish"
                strength = max(strength, 0.65)
            elif ma_short < ma_long:
                direction = "bearish"
                strength = max(strength, 0.65)

        now = datetime.now(timezone.utc)

        evidence = Evidence(
            evidence_id=f"ev_{uuid4().hex}",
            source_type="technical_indicator",
            source_name="computed_indicators",
            extracted_text=f"RSI={rsi}, short_MA={ma_short}, long_MA={ma_long}",
            confidence=1.0,
            published_at=now,
        )

        signal = FinancialSignal(
            signal_id=f"sig_{uuid4().hex}",
            ticker=ticker,
            signal_type="technical",
            direction=direction,
            strength=strength,
            valid_from=now,
            evidence_id=evidence.evidence_id,
            description=(
                f"Technical signal generated from RSI={rsi}, "
                f"short_MA={ma_short}, long_MA={ma_long}."
            ),
        )

        return GraphUpdate(
            ticker=ticker,
            signal=signal,
            evidence=evidence,
        )