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


def build_company_entity(ticker: str, name: str | None = None) -> Entity:
    properties = {"ticker": ticker}
    if name:
        properties["name"] = name
    return Entity(entity_id=f"company:{ticker}", type="Company", properties=properties)


def build_indicator_entities_from_ohlcv(
    ticker: str, ohlcv: pd.DataFrame
) -> Tuple[List[Entity], List[Relation]]:
    entities: List[Entity] = []
    relations: List[Relation] = []

    if "date" not in ohlcv.columns or "close" not in ohlcv.columns:
        raise ValueError("OHLCV frame must contain at least 'date' and 'close' columns.")

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


def build_news_from_frame(
    ticker: str, news: pd.DataFrame
) -> Tuple[List[Entity], List[Evidence], List[Relation]]:
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
        stable_hash = sha1(url.encode("utf-8")).hexdigest()[:16]

        ev_id = f"news:{ticker}:{stable_hash}"
        title = row.get("title")
        content = row.get("content") or title or ""
        evidences.append(
            Evidence(
                evidence_id=ev_id,
                source_type="news",
                source_name=row.get("source"),
                url=url,
                title=title,
                published_at=published_at,
                extracted_text=str(content)[:2000],
                confidence=float(row.get("score") or 0.6),
            )
        )

        news_ent_id = f"news_event:{ticker}:{stable_hash}"
        entities.append(
            Entity(
                entity_id=news_ent_id,
                type="NewsEvent",
                properties={
                    "ticker": ticker,
                    "title": title,
                    "source": row.get("source"),
                    "url": url,
                    "published_at": published_at.isoformat() if published_at else None,
                },
            )
        )

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

        # fix bugs: warining on evidence_support relationship(connect the NewsEvent and Evidence)
        relations.append(
            Relation(
                start_id=news_ent_id,
                end_id=ev_id,
                type="SUPPORTED_BY",
                as_of_date=as_of_date,
                confidence=float(row.get("score") or 0.6),
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
