# technical agent to generate technical signals for a given ticker

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.kg.schema import Entity, Relation


class TechnicalAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("technical_agent")

    def run(self, input_data: dict) -> Tuple[List[Entity], List[Relation]]:
        ticker = input_data["ticker"]
        ohlcv = input_data["ohlcv"]
        return self.build_signal_entities_from_ohlcv(ticker, ohlcv)

    def build_signal_entities_from_ohlcv(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        if "date" not in ohlcv.columns or "close" not in ohlcv.columns:
            raise ValueError("OHLCV frame must contain at least 'date' and 'close' columns.")

        frame = ohlcv.sort_values("date").reset_index(drop=True).copy()
        frame["close"] = frame["close"].astype(float)

        # Ensure we have high/low for ATR and Bollinger
        for col in ("high", "low"):
            if col not in frame.columns:
                frame[col] = frame["close"]

        # ── Compute all indicators ──
        self._compute_indicators(frame)

        # ── Generate signals ──
        entities: List[Entity] = []
        relations: List[Relation] = []

        generators = [
            self._gen_ma_crossover,       # MA20, MA50, MA200 crossovers
            self._gen_ema_crossover,      # EMA10 crossover
            self._gen_macd_signals,       # MACD line/signal crossover
            self._gen_rsi_signals,        # RSI overbought/oversold
            self._gen_bollinger_signals,  # Bollinger Band touch/pierce
            self._gen_atr_signals,        # ATR volatility regime
        ]

        for gen in generators:
            ents, rels = gen(ticker, frame)
            entities.extend(ents)
            relations.extend(rels)

        return entities, relations

    # ── Indicator computation ──

    def _compute_indicators(self, frame: pd.DataFrame) -> None:
        """Compute all technical indicators in-place on the frame."""
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]

        # Moving Averages
        frame["ma20"] = close.rolling(window=20, min_periods=20).mean()
        frame["ma50"] = close.rolling(window=50, min_periods=50).mean()
        frame["ma200"] = close.rolling(window=200, min_periods=200).mean()

        # EMA10
        frame["ema10"] = close.ewm(span=10, adjust=False).mean()

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        frame["macd_line"] = ema12 - ema26
        frame["macd_signal"] = frame["macd_line"].ewm(span=9, adjust=False).mean()
        frame["macd_hist"] = frame["macd_line"] - frame["macd_signal"]

        # RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        frame["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20, 2)
        frame["boll_mid"] = close.rolling(window=20, min_periods=20).mean()
        boll_std = close.rolling(window=20, min_periods=20).std()
        frame["boll_ub"] = frame["boll_mid"] + 2 * boll_std
        frame["boll_lb"] = frame["boll_mid"] - 2 * boll_std

        # ATR (14)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        frame["atr"] = true_range.rolling(window=14, min_periods=14).mean()

        # ATR as percentage of price (normalized)
        frame["atr_pct"] = frame["atr"] / close * 100

    # ── Signal generators ──

    def _gen_ma_crossover(
        self, ticker: str, frame: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Generate signals for MA20 / MA50 / MA200 crossovers."""
        entities: List[Entity] = []
        relations: List[Relation] = []

        ma_configs = [
            ("ma20", 20, 0.01),   # 1% threshold
            ("ma50", 50, 0.015),  # 1.5% threshold
            ("ma200", 200, 0.02), # 2% threshold
        ]

        for _, row in frame.iterrows():
            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])

            for ma_col, period, threshold in ma_configs:
                ma_val = row.get(ma_col)
                if pd.isna(ma_val):
                    continue
                ma_val = float(ma_val)

                signal_name = None
                direction = None
                strength = 0.0

                pct_diff = (price - ma_val) / ma_val
                if pct_diff > threshold:
                    signal_name = f"price_above_ma{period}"
                    direction = "bullish"
                    strength = min(0.9, 0.6 + abs(pct_diff) * 5)
                elif pct_diff < -threshold:
                    signal_name = f"price_below_ma{period}"
                    direction = "bearish"
                    strength = min(0.9, 0.6 + abs(pct_diff) * 5)
                else:
                    continue

                signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"
                entities.append(Entity(
                    entity_id=signal_id,
                    type="IndicatorSignal",
                    properties={
                        "ticker": ticker,
                        "name": signal_name,
                        "signal_type": "technical",
                        "direction": direction,
                        "strength": round(strength, 3),
                        "price": price,
                        f"ma{period}": round(ma_val, 4),
                        "pct_diff": round(pct_diff, 4),
                        "as_of_date": date.isoformat(),
                    },
                ))
                relations.append(Relation(
                    start_id=f"company:{ticker}",
                    end_id=signal_id,
                    type="HAS_SIGNAL",
                    as_of_date=date,
                    confidence=round(strength, 3),
                    direction=direction,
                    valid_from=date,
                    valid_to=None,
                    evidence_ids=None,
                ))

        return entities, relations

    def _gen_ema_crossover(
        self, ticker: str, frame: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Generate signals for EMA10 (short-term momentum)."""
        entities: List[Entity] = []
        relations: List[Relation] = []

        for _, row in frame.iterrows():
            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])
            ema10 = row.get("ema10")
            if pd.isna(ema10):
                continue
            ema10 = float(ema10)

            pct_diff = (price - ema10) / ema10
            signal_name = None
            direction = None

            if pct_diff > 0.02:
                signal_name = "price_above_ema10"
                direction = "bullish"
            elif pct_diff < -0.02:
                signal_name = "price_below_ema10"
                direction = "bearish"
            else:
                continue

            strength = min(0.85, 0.55 + abs(pct_diff) * 5)
            signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"

            entities.append(Entity(
                entity_id=signal_id,
                type="IndicatorSignal",
                properties={
                    "ticker": ticker,
                    "name": signal_name,
                    "signal_type": "technical",
                    "direction": direction,
                    "strength": round(strength, 3),
                    "price": price,
                    "ema10": round(ema10, 4),
                    "pct_diff": round(pct_diff, 4),
                    "as_of_date": date.isoformat(),
                },
            ))
            relations.append(Relation(
                start_id=f"company:{ticker}",
                end_id=signal_id,
                type="HAS_SIGNAL",
                as_of_date=date,
                confidence=round(strength, 3),
                direction=direction,
                valid_from=date,
                valid_to=None,
                evidence_ids=None,
            ))

        return entities, relations

    def _gen_macd_signals(
        self, ticker: str, frame: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Generate MACD crossover signals (line crosses signal line)."""
        entities: List[Entity] = []
        relations: List[Relation] = []

        prev_macd = None
        prev_signal = None

        for _, row in frame.iterrows():
            macd_val = row.get("macd_line")
            signal_val = row.get("macd_signal")
            if pd.isna(macd_val) or pd.isna(signal_val):
                prev_macd = macd_val
                prev_signal = signal_val
                continue

            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])

            signal_name = None
            direction = None

            # Bullish crossover: MACD crosses above signal line
            if prev_macd is not None and not pd.isna(prev_macd):
                if prev_macd <= prev_signal and macd_val > signal_val:
                    signal_name = "macd_bullish_crossover"
                    direction = "bullish"
                elif prev_macd >= prev_signal and macd_val < signal_val:
                    signal_name = "macd_bearish_crossover"
                    direction = "bearish"

            if signal_name:
                strength = min(0.85, 0.6 + abs(float(macd_val) - float(signal_val)) / price * 100)
                signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"

                entities.append(Entity(
                    entity_id=signal_id,
                    type="IndicatorSignal",
                    properties={
                        "ticker": ticker,
                        "name": signal_name,
                        "signal_type": "technical",
                        "direction": direction,
                        "strength": round(strength, 3),
                        "price": price,
                        "macd_line": round(float(macd_val), 4),
                        "macd_signal": round(float(signal_val), 4),
                        "macd_hist": round(float(row.get("macd_hist", 0)), 4),
                        "as_of_date": date.isoformat(),
                    },
                ))
                relations.append(Relation(
                    start_id=f"company:{ticker}",
                    end_id=signal_id,
                    type="HAS_SIGNAL",
                    as_of_date=date,
                    confidence=round(strength, 3),
                    direction=direction,
                    valid_from=date,
                    valid_to=None,
                    evidence_ids=None,
                ))

            prev_macd = macd_val
            prev_signal = signal_val

        return entities, relations

    def _gen_rsi_signals(
        self, ticker: str, frame: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Generate RSI overbought (>70) / oversold (<30) signals."""
        entities: List[Entity] = []
        relations: List[Relation] = []

        for _, row in frame.iterrows():
            rsi_val = row.get("rsi")
            if pd.isna(rsi_val):
                continue

            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])
            rsi_val = float(rsi_val)

            signal_name = None
            direction = None
            strength = 0.0

            if rsi_val >= 70:
                signal_name = "rsi_overbought"
                direction = "bearish"  # overbought → bearish signal
                strength = min(0.9, 0.6 + (rsi_val - 70) / 100)
            elif rsi_val <= 30:
                signal_name = "rsi_oversold"
                direction = "bullish"  # oversold → bullish signal
                strength = min(0.9, 0.6 + (30 - rsi_val) / 100)
            elif rsi_val >= 60:
                signal_name = "rsi_approaching_overbought"
                direction = "bearish"
                strength = 0.5
            elif rsi_val <= 40:
                signal_name = "rsi_approaching_oversold"
                direction = "bullish"
                strength = 0.5
            else:
                continue

            signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"
            entities.append(Entity(
                entity_id=signal_id,
                type="IndicatorSignal",
                properties={
                    "ticker": ticker,
                    "name": signal_name,
                    "signal_type": "technical",
                    "direction": direction,
                    "strength": round(strength, 3),
                    "price": price,
                    "rsi": round(rsi_val, 2),
                    "as_of_date": date.isoformat(),
                },
            ))
            relations.append(Relation(
                start_id=f"company:{ticker}",
                end_id=signal_id,
                type="HAS_SIGNAL",
                as_of_date=date,
                confidence=round(strength, 3),
                direction=direction,
                valid_from=date,
                valid_to=None,
                evidence_ids=None,
            ))

        return entities, relations

    def _gen_bollinger_signals(
        self, ticker: str, frame: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Generate Bollinger Band touch/pierce signals."""
        entities: List[Entity] = []
        relations: List[Relation] = []

        for _, row in frame.iterrows():
            ub = row.get("boll_ub")
            lb = row.get("boll_lb")
            if pd.isna(ub) or pd.isna(lb):
                continue

            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])
            ub = float(ub)
            lb = float(lb)

            signal_name = None
            direction = None
            strength = 0.0

            # Price at or above upper band → overbought / bearish
            if price >= ub:
                signal_name = "boll_upper_pierce"
                direction = "bearish"
                strength = min(0.85, 0.6 + (price - ub) / ub * 10)
            # Price at or below lower band → oversold / bullish
            elif price <= lb:
                signal_name = "boll_lower_pierce"
                direction = "bullish"
                strength = min(0.85, 0.6 + (lb - price) / lb * 10)
            # Near upper band (within 1%)
            elif price >= ub * 0.99:
                signal_name = "boll_upper_near"
                direction = "bearish"
                strength = 0.55
            # Near lower band (within 1%)
            elif price <= lb * 1.01:
                signal_name = "boll_lower_near"
                direction = "bullish"
                strength = 0.55
            else:
                continue

            signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"
            entities.append(Entity(
                entity_id=signal_id,
                type="IndicatorSignal",
                properties={
                    "ticker": ticker,
                    "name": signal_name,
                    "signal_type": "technical",
                    "direction": direction,
                    "strength": round(strength, 3),
                    "price": price,
                    "boll_ub": round(ub, 4),
                    "boll_lb": round(lb, 4),
                    "boll_mid": round(float(row.get("boll_mid", 0)), 4),
                    "as_of_date": date.isoformat(),
                },
            ))
            relations.append(Relation(
                start_id=f"company:{ticker}",
                end_id=signal_id,
                type="HAS_SIGNAL",
                as_of_date=date,
                confidence=round(strength, 3),
                direction=direction,
                valid_from=date,
                valid_to=None,
                evidence_ids=None,
            ))

        return entities, relations

    def _gen_atr_signals(
        self, ticker: str, frame: pd.DataFrame,
    ) -> Tuple[List[Entity], List[Relation]]:
        """Generate ATR volatility regime signals."""
        entities: List[Entity] = []
        relations: List[Relation] = []

        # Compute rolling average ATR for regime detection
        atr_pct_series = frame["atr_pct"].dropna()
        if len(atr_pct_series) < 50:
            return entities, relations

        rolling_mean_atr = atr_pct_series.rolling(window=60, min_periods=30).mean()
        rolling_std_atr = atr_pct_series.rolling(window=60, min_periods=30).std()

        for idx, row in frame.iterrows():
            atr_pct = row.get("atr_pct")
            if pd.isna(atr_pct):
                continue

            date = pd.to_datetime(row["date"]).to_pydatetime()
            price = float(row["close"])
            atr_pct = float(atr_pct)

            mean_atr = rolling_mean_atr.get(idx)
            std_atr = rolling_std_atr.get(idx)
            if pd.isna(mean_atr) or pd.isna(std_atr) or std_atr == 0:
                continue

            z_score = (atr_pct - float(mean_atr)) / float(std_atr)

            signal_name = None
            direction = None
            strength = 0.0

            if z_score > 2.0:
                signal_name = "high_volatility_regime"
                direction = "bearish"  # high vol generally bearish
                strength = min(0.85, 0.6 + z_score * 0.05)
            elif z_score < -1.5:
                signal_name = "low_volatility_regime"
                direction = "neutral"  # low vol → breakout pending
                strength = 0.45
            else:
                continue

            signal_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"
            entities.append(Entity(
                entity_id=signal_id,
                type="IndicatorSignal",
                properties={
                    "ticker": ticker,
                    "name": signal_name,
                    "signal_type": "technical",
                    "direction": direction,
                    "strength": round(strength, 3),
                    "price": price,
                    "atr": round(float(row.get("atr", 0)), 4),
                    "atr_pct": round(atr_pct, 2),
                    "atr_z_score": round(z_score, 2),
                    "as_of_date": date.isoformat(),
                },
            ))
            relations.append(Relation(
                start_id=f"company:{ticker}",
                end_id=signal_id,
                type="HAS_SIGNAL",
                as_of_date=date,
                confidence=round(strength, 3),
                direction=direction,
                valid_from=date,
                valid_to=None,
                evidence_ids=None,
            ))

        return entities, relations
