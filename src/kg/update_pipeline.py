from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from src.kg.schema import Entity, Evidence, Relation


@dataclass
class KGBatch:
    entities: List[Entity]
    evidences: List[Evidence]
    relations: List[Relation]


def build_company_entity(ticker: str, name: str | None = None) -> Entity:
    return Entity(
        entity_id=f"company:{ticker}",
        type="Company",
        properties={"ticker": ticker, "name": name} if name else {"ticker": ticker},
    )


def build_indicator_entities_from_ohlcv(
    ticker: str, ohlcv: pd.DataFrame
) -> Tuple[List[Entity], List[Relation]]:
    entities: List[Entity] = []
    relations: List[Relation] = []

    if "date" not in ohlcv.columns or "close" not in ohlcv.columns:
        raise ValueError("OHLCV frame must contain at least 'date' and 'close' columns.")

    ohlcv = ohlcv.sort_values("date")
    close = ohlcv["close"].astype(float)
    ma20 = close.rolling(window=20, min_periods=20).mean()

    for idx, row in ohlcv.iterrows():
        date = row["date"]
        if pd.isna(ma20.loc[idx]):
            continue

        price = float(row["close"])
        ma = float(ma20.loc[idx])
        if price > ma * 1.01:
            signal = "price_above_ma20"
            direction = "bullish"
        elif price < ma * 0.99:
            signal = "price_below_ma20"
            direction = "bearish"
        else:
            continue

        sig_id = f"signal:{ticker}:{signal}:{date}"
        entities.append(
            Entity(
                entity_id=sig_id,
                type="IndicatorSignal",
                properties={
                    "ticker": ticker,
                    "name": signal,
                    "price": price,
                    "ma20": ma,
                    "as_of_date": str(date),
                },
            )
        )
        relations.append(
            Relation(
                start_id=f"company:{ticker}",
                end_id=sig_id,
                type="HAS_SIGNAL",
                as_of_date=_to_dt(date),
                confidence=0.7,
                direction=direction,
                valid_from=_to_dt(date),
                valid_to=None,
                evidence_ids=None,
            )
        )

    return entities, relations


def build_news_from_frame(ticker: str, news: pd.DataFrame) -> Tuple[List[Entity], List[Evidence], List[Relation]]:
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
        published_time = _parse_dt(pub_time_raw)

        ev_id = f"news:{ticker}:{hash(url)}"
        evidences.append(
            Evidence(
                evidence_id=ev_id,
                source_type="news",
                url=url,
                published_time=published_time,
                snippet=(row.get("title") or "")[:512],
            )
        )

        news_ent_id = f"news_event:{ticker}:{hash(url)}"
        entities.append(
            Entity(
                entity_id=news_ent_id,
                type="NewsEvent",
                properties={
                    "ticker": ticker,
                    "title": row.get("title"),
                    "source": row.get("source"),
                    "url": url,
                },
            )
        )

        as_of_date = published_time or datetime.utcnow()
        relations.append(
            Relation(
                start_id=f"company:{ticker}",
                end_id=news_ent_id,
                type="MENTIONED_IN",
                as_of_date=as_of_date,
                confidence=0.6,
                direction=None,
                valid_from=as_of_date,
                valid_to=None,
                evidence_ids=[ev_id],
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


def _to_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_dt(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None

