# technical agent to generate technical signals for a given ticker

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

import pandas as pd

from src.agents.base_agent import BaseAgent
from src.kg.schema import Entity, Relation


class TechnicalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("technical_agent")

    def build_signal_entities_from_ohlcv(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        entities: List[Entity] = []
        relations: List[Relation] = []

        if "date" not in ohlcv.columns or "close" not in ohlcv.columns:
            raise ValueError("OHLCV frame must contain at least 'date' and 'close' columns.")

        frame = ohlcv.sort_values("date").reset_index(drop=True).copy()
        frame["close"] = frame["close"].astype(float)
        frame["ma20"] = frame["close"].rolling(window=20, min_periods=20).mean()

        for _, row in frame.iterrows():
            if pd.isna(row["ma20"]):
                continue

            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])
            ma20 = float(row["ma20"])

            signal_name = None
            direction = None

            if price > ma20 * 1.01:
                signal_name = "price_above_ma20"
                direction = "bullish"
            elif price < ma20 * 0.99:
                signal_name = "price_below_ma20"
                direction = "bearish"

            if signal_name is None:
                continue

            signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"

            entities.append(
                Entity(
                    entity_id=signal_id,
                    type="IndicatorSignal",
                    properties={
                        "ticker": ticker,
                        "name": signal_name,
                        "signal_type": "technical",
                        "direction": direction,
                        "strength": 0.7,
                        "price": price,
                        "ma20": ma20,
                        "as_of_date": date.isoformat(),
                    },
                )
            )

            relations.append(
                Relation(
                    start_id=f"company:{ticker}",
                    end_id=signal_id,
                    type="HAS_SIGNAL",
                    as_of_date=date,
                    confidence=0.7,
                    direction=direction,
                    valid_from=date,
                    valid_to=None,
                    evidence_ids=None,
                )
            )

        return entities, relations
