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


def build_kg_batch_for_ticker(
    ticker: str, ohlcv: pd.DataFrame, news: pd.DataFrame
) -> KGBatch:
    company = build_company_entity(ticker)
    sig_entities, sig_relations = build_indicator_entities_from_ohlcv(ticker, ohlcv)
    news_entities, evidences, news_relations = build_news_from_frame(ticker, news)

    entities = [company, *sig_entities, *news_entities]
    relations = [*sig_relations, *news_relations]

    return KGBatch(entities=entities, evidences=evidences, relations=relations)


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