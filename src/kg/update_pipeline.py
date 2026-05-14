from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from typing import List, Tuple

import pandas as pd

from src.kg.schema import Entity, Evidence, Relation


@dataclass
class KGBatch:
    entities: List[Entity]
    evidences: List[Evidence]
    relations: List[Relation]

# build company entity from ticker and name
def build_company_entity(ticker: str, name: str | None = None) -> Entity:
    properties = {"ticker": ticker}
    if name:
        properties["name"] = name
    return Entity(entity_id=f"company:{ticker}", type="Company", properties=properties)

# build technical indicator signal entities from ohlcv frame
def build_indicator_entities_from_ohlcv(
    ticker: str, ohlcv: pd.DataFrame
) -> Tuple[List[Entity], List[Relation]]:
    entities: List[Entity] = []
    relations: List[Relation] = []

    if "date" not in ohlcv.columns or "close" not in ohlcv.columns:
        raise ValueError("OHLCV frame must contain at least 'date' and 'close' columns.")

    # For simplicity, we only generate one type of technical signal based on 20-day moving average crossover.
    # TODO: in the future, expand to more technical indicators and patterns in the future.
    ohlcv = ohlcv.sort_values("date").reset_index(drop=True)
    close = ohlcv["close"].astype(float)
    ma20 = close.rolling(window=20, min_periods=20).mean()

    for idx, row in ohlcv.iterrows():
        date = _to_dt(row["date"])
        if pd.isna(ma20.iloc[idx]):
            continue

        price = float(row["close"])
        ma = float(ma20.iloc[idx])

        if price > ma * 1.01:
            signal_name = "price_above_ma20"
            direction = "bullish"
        elif price < ma * 0.99:
            signal_name = "price_below_ma20"
            direction = "bearish"
        else:
            continue

        sig_id = f"signal:{ticker}:{signal_name}:{date.date().isoformat()}"

        entities.append(
            Entity(
                entity_id=sig_id,
                type="IndicatorSignal",
                properties={
                    "ticker": ticker,
                    "name": signal_name,
                    "signal_type": "technical",
                    "direction": direction,
                    "strength": 0.7,
                    "price": price,
                    "ma20": ma,
                    "as_of_date": date.isoformat(),
                },
            )
        )

        relations.append(
            Relation(
                start_id=f"company:{ticker}",
                end_id=sig_id,
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

# build news from news frame
def build_news_from_frame(
    ticker: str, news: pd.DataFrame
) -> Tuple[List[Entity], List[Evidence], List[Relation]]:
    """
    Build a richer evidence-grounded news chain:

    SourceDocument
        -[:CONTAINS_EVIDENCE]->
    Evidence
        -[:SUPPORTS_CLAIM]->
    Claim
        -[:CLAIM_USED_BY]->
    NewsEvent

    Company
        -[:MENTIONED_IN]->
    NewsEvent
    """

    entities: List[Entity] = []
    evidences: List[Evidence] = []
    relations: List[Relation] = []

    if "url" not in news.columns:
        raise ValueError("News frame must include 'url' column.")

    seen_urls = set()

    for _, row in news.iterrows():
        url = row.get("url")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        pub_time_raw = row.get("published_time")
        published_at = _parse_dt(pub_time_raw)
        as_of_date = published_at or datetime.utcnow()

        title = row.get("title")
        source_name = row.get("source")
        content = row.get("content") or title or ""
        confidence = float(row.get("score") or 0.6)

        content_hash = sha1(f"{url}|{title}|{content}".encode("utf-8")).hexdigest()[:16]

        source_id = f"source:news:{ticker}:{content_hash}"
        ev_id = f"news:{ticker}:{content_hash}"
        claim_id = f"claim:news:{ticker}:{content_hash}"
        news_ent_id = f"news_event:{ticker}:{content_hash}"

        # 1. SourceDocument node
        entities.append(
            Entity(
                entity_id=source_id,
                type="SourceDocument",
                properties={
                    "source_id": source_id,
                    "source_type": "news",
                    "source_name": source_name,
                    "url": url,
                    "title": title,
                    "published_at": published_at.isoformat() if published_at else None,
                    "retrieved_at": datetime.utcnow().isoformat(),
                    "content_hash": content_hash,
                    "raw_text_preview": str(content)[:500],
                },
            )
        )

        # 2. Evidence node
        evidences.append(
            Evidence(
                evidence_id=ev_id,
                source_type="news",
                source_name=source_name,
                url=url,
                title=title,
                published_at=published_at,
                extracted_text=str(content)[:2000],
                confidence=confidence,
            )
        )

        # 3. Claim node
        claim_text = _build_news_claim_text(ticker=ticker, title=title, content=content)
        entities.append(
            Entity(
                entity_id=claim_id,
                type="Claim",
                properties={
                    "claim_id": claim_id,
                    "ticker": ticker,
                    "claim_type": "news",
                    "text": claim_text,
                    "polarity": "unknown",
                    "confidence": confidence,
                    "as_of_date": as_of_date.isoformat(),
                    "valid_from": as_of_date.isoformat(),
                    "valid_to": None,
                    "evidence_ids": [ev_id],
                },
            )
        )

        # 4. NewsEvent node
        entities.append(
            Entity(
                entity_id=news_ent_id,
                type="NewsEvent",
                properties={
                    "ticker": ticker,
                    "title": title,
                    "source": source_name,
                    "url": url,
                    "published_at": published_at.isoformat() if published_at else None,
                    "claim_id": claim_id,
                    "evidence_id": ev_id,
                    "source_id": source_id,
                },
            )
        )

        # Company -> NewsEvent
        relations.append(
            Relation(
                start_id=f"company:{ticker}",
                end_id=news_ent_id,
                type="MENTIONED_IN",
                as_of_date=as_of_date,
                confidence=confidence,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[ev_id],
            )
        )

        # SourceDocument -> Evidence
        relations.append(
            Relation(
                start_id=source_id,
                end_id=ev_id,
                type="CONTAINS_EVIDENCE",
                as_of_date=as_of_date,
                confidence=confidence,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[ev_id],
            )
        )

        # Evidence -> Claim
        relations.append(
            Relation(
                start_id=ev_id,
                end_id=claim_id,
                type="SUPPORTS_CLAIM",
                as_of_date=as_of_date,
                confidence=confidence,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[ev_id],
            )
        )

        # Claim -> NewsEvent
        relations.append(
            Relation(
                start_id=claim_id,
                end_id=news_ent_id,
                type="CLAIM_USED_BY",
                as_of_date=as_of_date,
                confidence=confidence,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[ev_id],
            )
        )

        # Backward-compatible relation: NewsEvent -> Evidence
        relations.append(
            Relation(
                start_id=news_ent_id,
                end_id=ev_id,
                type="SUPPORTED_BY",
                as_of_date=as_of_date,
                confidence=confidence,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[ev_id],
            )
        )

    return entities, evidences, relations

# add fundamental signal entities builder from fundamental frame
def build_fundamentals_from_frame(
    ticker: str,
    fundamentals: pd.DataFrame,
) -> Tuple[List[Entity], List[Evidence], List[Relation]]:
    """Build FundamentalSignal nodes from collected fundamental metrics.

    Output structure:

    Company
        -[:HAS_SIGNAL]->
    FundamentalSignal
        -[:SUPPORTED_BY]->
    Evidence

    Evidence
        -[:SUPPORTS_CLAIM]->
    Claim
        -[:CLAIM_USED_BY]->
    FundamentalSignal
    """

    entities: List[Entity] = []
    evidences: List[Evidence] = []
    relations: List[Relation] = []

    # The input frame is expected to have columns: metric, value, as_of_date, source (optional)
    required_cols = {"metric", "value", "as_of_date"}
    missing = required_cols - set(fundamentals.columns)
    if missing:
        raise ValueError(f"Fundamentals frame missing columns: {sorted(missing)}")

    # for each fundamental metric, create a fundamental signal entity & an evidence node & claim node & link them
    # Claim text: generated based on metrics
    for _, row in fundamentals.iterrows():
        metric = str(row.get("metric") or "").strip()
        if not metric:
            continue

        raw_value = row.get("value")
        numeric_value = row.get("numeric_value")

        value_for_interpretation = numeric_value
        if pd.isna(value_for_interpretation):
            value_for_interpretation = raw_value

        as_of_date = _parse_dt(row.get("as_of_date")) or datetime.utcnow()
        source_name = str(row.get("source") or "yfinance.info")

        stable_hash = sha1(
            f"{ticker}|{metric}|{raw_value}|{numeric_value}|{as_of_date.date()}".encode("utf-8")
        ).hexdigest()[:16]

        signal_id = f"fundamental:{ticker}:{metric}:{stable_hash}"
        evidence_id = f"fundamental_evidence:{ticker}:{metric}:{stable_hash}"
        claim_id = f"claim:fundamental:{ticker}:{metric}:{stable_hash}"

        direction, strength, description = _interpret_fundamental_metric(
            metric,
            value_for_interpretation,
        )

        evidence_text = (
            f"{ticker} fundamental metric {metric} = {raw_value}, "
            f"numeric_value = {numeric_value}, collected from {source_name}."
        )

        evidences.append(
            Evidence(
                evidence_id=evidence_id,
                source_type="fundamental",
                source_name=source_name,
                url=None,
                title=f"{ticker} {metric}",
                published_at=as_of_date,
                extracted_text=evidence_text,
                confidence=0.75,
            )
        )

        entities.append(
            Entity(
                entity_id=signal_id,
                type="FundamentalSignal",
                properties={
                    "ticker": ticker,
                    "name": metric,
                    "signal_type": "fundamental",
                    "direction": direction,
                    "strength": strength,
                    "metric": metric,
                    "value": _json_safe_value(raw_value),
                    "numeric_value": None if pd.isna(numeric_value) else _json_safe_value(numeric_value),
                    "as_of_date": as_of_date.isoformat(),
                    "description": description,
                    "evidence_id": evidence_id,
                    "claim_id": claim_id,
                },
            )
        )

        entities.append(
            Entity(
                entity_id=claim_id,
                type="Claim",
                properties={
                    "claim_id": claim_id,
                    "ticker": ticker,
                    "claim_type": "fundamental",
                    "text": description,
                    "polarity": _direction_to_claim_polarity(direction),
                    "confidence": 0.75,
                    "as_of_date": as_of_date.isoformat(),
                    "valid_from": as_of_date.isoformat(),
                    "valid_to": None,
                    "evidence_ids": [evidence_id],
                },
            )
        )

        relations.append(
            Relation(
                start_id=f"company:{ticker}",
                end_id=signal_id,
                type="HAS_SIGNAL",
                as_of_date=as_of_date,
                confidence=0.75,
                direction=direction,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

        relations.append(
            Relation(
                start_id=signal_id,
                end_id=evidence_id,
                type="SUPPORTED_BY",
                as_of_date=as_of_date,
                confidence=0.75,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

        relations.append(
            Relation(
                start_id=evidence_id,
                end_id=claim_id,
                type="SUPPORTS_CLAIM",
                as_of_date=as_of_date,
                confidence=0.75,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

        relations.append(
            Relation(
                start_id=claim_id,
                end_id=signal_id,
                type="CLAIM_USED_BY",
                as_of_date=as_of_date,
                confidence=0.75,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

    return entities, evidences, relations


def build_kg_batch_for_ticker(
    ticker: str, ohlcv: pd.DataFrame, news: pd.DataFrame
) -> KGBatch:
    company = build_company_entity(ticker)
    sig_entities, sig_relations = build_indicator_entities_from_ohlcv(ticker, ohlcv)
    news_entities, evidences, news_relations = build_news_from_frame(ticker, news)

    entities = [company, *sig_entities, *news_entities]
    relations = [*sig_relations, *news_relations]

    return KGBatch(entities=entities, evidences=evidences, relations=relations)

# update risk events based on OHLCV and fundamental metrics
def build_risk_events_from_frame(
    ticker: str,
    ohlcv: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
) -> Tuple[List[Entity], List[Evidence], List[Relation]]:
    """Build explicit KG risk events from OHLCV and fundamental metrics.

    Risk types:
    - volatility risk
    - drawdown risk
    - beta risk
    - leverage risk
    - liquidity risk
    """

    entities: List[Entity] = []
    evidences: List[Evidence] = []
    relations: List[Relation] = []

    if "date" not in ohlcv.columns or "close" not in ohlcv.columns:
        raise ValueError("OHLCV frame must contain at least 'date' and 'close' columns.")

    frame = ohlcv.sort_values("date").reset_index(drop=True).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = frame["close"].astype(float)

    latest_date = _to_dt(frame["date"].iloc[-1])
    close = frame["close"]

    daily_ret = close.pct_change()
    vol_20 = float(daily_ret.tail(20).std() or 0.0)
    drawdown_60 = float((close.iloc[-1] / close.tail(60).max()) - 1.0)

    risk_specs: list[dict[str, object]] = []

    if vol_20 >= 0.025:
        risk_specs.append(
            {
                "name": "high_20d_volatility",
                "severity": min(1.0, vol_20 / 0.06),
                "direction": "bearish",
                "description": f"{ticker} has elevated 20-day daily volatility of {vol_20:.4f}.",
                "metric": "vol_20",
                "value": vol_20,
            }
        )

    if drawdown_60 <= -0.12:
        risk_specs.append(
            {
                "name": "large_60d_drawdown",
                "severity": min(1.0, abs(drawdown_60) / 0.35),
                "direction": "bearish",
                "description": f"{ticker} is in a 60-day drawdown of {drawdown_60:.4f}.",
                "metric": "drawdown_60",
                "value": drawdown_60,
            }
        )

    if fundamentals is not None and not fundamentals.empty:
        metric_map = {
            str(row.get("metric")): row
            for _, row in fundamentals.iterrows()
            if row.get("metric") is not None
        }

        beta = _get_fundamental_numeric(metric_map, "beta")
        if beta is not None and beta >= 1.4:
            risk_specs.append(
                {
                    "name": "high_beta_risk",
                    "severity": min(1.0, beta / 2.5),
                    "direction": "bearish",
                    "description": f"{ticker} has high beta of {beta:.4f}, indicating market sensitivity risk.",
                    "metric": "beta",
                    "value": beta,
                }
            )

        debt_to_equity = _get_fundamental_numeric(metric_map, "debtToEquity")
        if debt_to_equity is not None and debt_to_equity >= 200:
            risk_specs.append(
                {
                    "name": "high_leverage_risk",
                    "severity": min(1.0, debt_to_equity / 400.0),
                    "direction": "bearish",
                    "description": f"{ticker} has high debt-to-equity of {debt_to_equity:.4f}.",
                    "metric": "debtToEquity",
                    "value": debt_to_equity,
                }
            )

        current_ratio = _get_fundamental_numeric(metric_map, "currentRatio")
        if current_ratio is not None and current_ratio < 1.0:
            risk_specs.append(
                {
                    "name": "weak_liquidity_risk",
                    "severity": min(1.0, (1.0 - current_ratio) + 0.5),
                    "direction": "bearish",
                    "description": f"{ticker} has weak current ratio of {current_ratio:.4f}.",
                    "metric": "currentRatio",
                    "value": current_ratio,
                }
            )

    if not risk_specs:
        risk_specs.append(
            {
                "name": "no_major_rule_based_risk_detected",
                "severity": 0.2,
                "direction": "neutral",
                "description": f"No major rule-based volatility, drawdown, leverage, beta, or liquidity risk detected for {ticker}.",
                "metric": "risk_screen",
                "value": 0.0,
            }
        )

    for spec in risk_specs:
        risk_name = str(spec["name"])
        description = str(spec["description"])
        direction = str(spec["direction"])
        severity = float(spec["severity"])
        metric = str(spec["metric"])
        value = spec["value"]

        stable_hash = sha1(
            f"{ticker}|{risk_name}|{metric}|{value}|{latest_date.date()}".encode("utf-8")
        ).hexdigest()[:16]

        risk_id = f"risk:{ticker}:{risk_name}:{stable_hash}"
        evidence_id = f"risk_evidence:{ticker}:{risk_name}:{stable_hash}"
        claim_id = f"claim:risk:{ticker}:{risk_name}:{stable_hash}"

        evidences.append(
            Evidence(
                evidence_id=evidence_id,
                source_type="risk",
                source_name="rule_based_risk_builder",
                url=None,
                title=f"{ticker} {risk_name}",
                published_at=latest_date,
                extracted_text=description,
                confidence=0.75,
            )
        )

        entities.append(
            Entity(
                entity_id=risk_id,
                type="RiskEvent",
                properties={
                    "ticker": ticker,
                    "name": risk_name,
                    "risk_type": metric,
                    "direction": direction,
                    "severity": severity,
                    "strength": severity,
                    "value": _json_safe_value(value),
                    "as_of_date": latest_date.isoformat(),
                    "description": description,
                    "evidence_id": evidence_id,
                    "claim_id": claim_id,
                },
            )
        )

        entities.append(
            Entity(
                entity_id=claim_id,
                type="Claim",
                properties={
                    "claim_id": claim_id,
                    "ticker": ticker,
                    "claim_type": "risk",
                    "text": description,
                    "polarity": _direction_to_claim_polarity(direction),
                    "confidence": 0.75,
                    "as_of_date": latest_date.isoformat(),
                    "valid_from": latest_date.isoformat(),
                    "valid_to": None,
                    "evidence_ids": [evidence_id],
                },
            )
        )

        relations.append(
            Relation(
                start_id=f"company:{ticker}",
                end_id=risk_id,
                type="HAS_RISK",
                as_of_date=latest_date,
                confidence=0.75,
                direction=direction,
                valid_from=latest_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

        relations.append(
            Relation(
                start_id=risk_id,
                end_id=evidence_id,
                type="SUPPORTED_BY",
                as_of_date=latest_date,
                confidence=0.75,
                direction=None,
                valid_from=latest_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

        relations.append(
            Relation(
                start_id=evidence_id,
                end_id=claim_id,
                type="SUPPORTS_CLAIM",
                as_of_date=latest_date,
                confidence=0.75,
                direction=None,
                valid_from=latest_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

        relations.append(
            Relation(
                start_id=claim_id,
                end_id=risk_id,
                type="CLAIM_USED_BY",
                as_of_date=latest_date,
                confidence=0.75,
                direction=None,
                valid_from=latest_date,
                valid_to=None,
                evidence_ids=[evidence_id],
            )
        )

    return entities, evidences, relations

def _build_news_claim_text(ticker: str, title: object, content: object) -> str:
    title_text = str(title or "").strip()
    content_text = str(content or "").strip()

    if title_text:
        return f"News about {ticker}: {title_text}"

    if content_text:
        return f"News evidence about {ticker}: {content_text[:240]}"

    return f"News evidence about {ticker} was collected but no textual content was available."


def _to_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_dt(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    

def _interpret_fundamental_metric(metric: str, value: object) -> tuple[str, float, str]:
    """Simple rule-based interpretation for fundamental fields.

    This is intentionally conservative. It gives the FundamentalAgent structured
    signals while avoiding overclaiming from one isolated metric.
    """

    numeric_value = _to_float(value)

    if numeric_value is None:
        return (
            "neutral",
            0.4,
            f"Fundamental metric {metric} is available with non-numeric value: {value}.",
        )

    metric_lower = metric.lower()

    if metric_lower in {"profitmargins", "operatingmargins", "grossmargins"}:
        if numeric_value >= 0.20:
            direction = "bullish"
            strength = 0.75
        elif numeric_value < 0:
            direction = "bearish"
            strength = 0.75
        else:
            direction = "neutral"
            strength = 0.5
        return direction, strength, f"{metric} is {numeric_value:.4f}, indicating {direction} profitability signal."

    if metric_lower in {"returnonequity", "returnonassets"}:
        if numeric_value >= 0.15:
            direction = "bullish"
            strength = 0.75
        elif numeric_value < 0:
            direction = "bearish"
            strength = 0.75
        else:
            direction = "neutral"
            strength = 0.5
        return direction, strength, f"{metric} is {numeric_value:.4f}, indicating {direction} return efficiency signal."

    if metric_lower in {"revenuegrowth", "earningsgrowth"}:
        if numeric_value >= 0.10:
            direction = "bullish"
            strength = 0.75
        elif numeric_value < 0:
            direction = "bearish"
            strength = 0.75
        else:
            direction = "neutral"
            strength = 0.5
        return direction, strength, f"{metric} is {numeric_value:.4f}, indicating {direction} growth signal."

    if metric_lower == "debttoequity":
        if numeric_value <= 80:
            direction = "bullish"
            strength = 0.65
        elif numeric_value >= 200:
            direction = "bearish"
            strength = 0.75
        else:
            direction = "neutral"
            strength = 0.5
        return direction, strength, f"Debt-to-equity is {numeric_value:.4f}, indicating {direction} leverage signal."

    if metric_lower in {"currentratio", "quickratio"}:
        if numeric_value >= 1.5:
            direction = "bullish"
            strength = 0.65
        elif numeric_value < 1.0:
            direction = "bearish"
            strength = 0.7
        else:
            direction = "neutral"
            strength = 0.5
        return direction, strength, f"{metric} is {numeric_value:.4f}, indicating {direction} liquidity signal."

    if metric_lower in {"trailingpe", "forwardpe"}:
        if numeric_value <= 0:
            direction = "bearish"
            strength = 0.6
        elif numeric_value < 12:
            direction = "bullish"
            strength = 0.55
        elif numeric_value > 40:
            direction = "bearish"
            strength = 0.55
        else:
            direction = "neutral"
            strength = 0.45
        return direction, strength, f"{metric} is {numeric_value:.4f}, indicating {direction} valuation signal."

    if metric_lower == "pricetobook":
        if numeric_value < 1.0:
            direction = "bullish"
            strength = 0.55
        elif numeric_value > 8.0:
            direction = "bearish"
            strength = 0.55
        else:
            direction = "neutral"
            strength = 0.45
        return direction, strength, f"Price-to-book is {numeric_value:.4f}, indicating {direction} valuation signal."

    if metric_lower in {"freecashflow", "operatingcashflow", "ebitda", "totalrevenue", "grossprofits"}:
        if numeric_value > 0:
            direction = "bullish"
            strength = 0.55
        elif numeric_value < 0:
            direction = "bearish"
            strength = 0.65
        else:
            direction = "neutral"
            strength = 0.4
        return direction, strength, f"{metric} is {numeric_value:.4f}, indicating {direction} scale/cashflow signal."

    return (
        "neutral",
        0.35,
        f"Fundamental metric {metric} is {numeric_value:.4f}; no strong directional rule is applied.",
    )


def _direction_to_claim_polarity(direction: str) -> str:
    if direction == "bullish":
        return "supports"
    if direction == "bearish":
        return "contradicts"
    if direction == "neutral":
        return "neutral"
    return "unknown"


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _json_safe_value(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

# helper to extract numeric value from fundamental metric rows
def _get_fundamental_numeric(metric_map: dict[str, object], metric: str) -> float | None:
    row = metric_map.get(metric)
    if row is None:
        return None

    try:
        numeric_value = row.get("numeric_value")
        if numeric_value is not None and not pd.isna(numeric_value):
            return float(numeric_value)
    except Exception:
        pass

    try:
        value = row.get("value")
        if value is not None and not pd.isna(value):
            return float(value)
    except Exception:
        return None

    return None