"""Strictly time-scoped sources for the dissertation protocol."""
from datetime import datetime
from hashlib import sha256
import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing import Literal

from src.eval.protocol import available_time
from src.eval.records import digest
from src.eval.research_safety import public_text


def _source_time(value):
    if value is None or len(value) < 16 or value[10] not in "T ":
        return None
    try:
        return available_time(datetime.fromisoformat(value))
    except ValueError:
        return None


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(pattern=r"^[0-9]{4,5}\.HK$")
    source_type: Literal["hkexnews", "local_news", "southbound_holding", "ah_premium",
                         "technical", "fundamental"]
    source: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    language: Literal["zh-Hant", "en", "bilingual"]
    published_at: str | None
    valid_to: str | None = None
    text: str = Field(min_length=1)
    period: str | None = None
    value: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = None
    a_share: str | None = Field(default=None, pattern=r"^[0-9]{6}\.(SS|SZ)$")

    @model_validator(mode="after")
    def validate_observation(self):
        published = _source_time(self.published_at)
        expiry = _source_time(self.valid_to)
        if published is None or (self.valid_to is not None and (expiry is None or expiry <= published)):
            raise ValueError("Invalid source publication or expiry")
        if not self.text.strip() or any(not token.strip() for token in (self.source, self.document_id, self.version)):
            raise ValueError("Source metadata and excerpt must contain text")
        if self.source_type in {"southbound_holding", "ah_premium"}:
            if not self.period or self.value is None or not self.unit:
                raise ValueError("Market relationship requires a period, value and unit")
        if self.source_type == "ah_premium" and not self.a_share:
            raise ValueError("A/H premium requires the confirmed mainland share code")
        if self.a_share and self.source_type != "ah_premium":
            raise ValueError("Mainland share code belongs to an A/H premium record")
        return self


def read_json_input(path):
    data = path.read_bytes()
    return json.loads(data.decode("utf-8-sig")), sha256(data).hexdigest()


class LocalEvidence:
    def __init__(self, payload):
        if not isinstance(payload, list) or not payload:
            raise ValueError("Provide a non-empty list of source records")
        self.by_ticker = {}
        self.rejected = []
        seen = set()
        versions = set()
        for index, item in enumerate(payload):
            try:
                source = SourceRecord.model_validate(item)
            except ValidationError as error:
                self.rejected.append({"index": index, "field_errors": [
                    {"field": list(issue["loc"]), "type": issue["type"]} for issue in error.errors()
                ]})
                continue
            published = _source_time(source.published_at)
            key = (source.ticker, source.source_type, source.source, source.document_id)
            if (key, published) in seen or (key, source.version) in versions:
                raise ValueError("Source revisions need distinct versions and publication instants")
            seen.add((key, published))
            versions.add((key, source.version))
            fields = source.model_dump()
            fields.update(published_at=published.isoformat(),
                          valid_to=_source_time(source.valid_to).isoformat() if source.valid_to else None,
                          source=public_text(source.source), document_id=public_text(source.document_id),
                          version=public_text(source.version), extracted_text=public_text(source.text.strip()))
            content_hash = digest(fields.pop("text"))
            fields.update(source_fingerprint=digest([source.model_dump(exclude={"text"}), content_hash]))
            fields["evidence_id"] = "evidence:" + fields["source_fingerprint"]
            self.by_ticker.setdefault(source.ticker, []).append((key, fields))

    def snapshot(self, ticker, cutoff):
        if not re.fullmatch(r"[0-9]{4,5}\.HK", ticker) or ticker not in self.by_ticker:
            raise ValueError("Missing source coverage for ticker")
        latest = {}
        for key, row in sorted(self.by_ticker[ticker], key=lambda item: item[1]["published_at"]):
            if available_time(row["published_at"]) < cutoff:
                latest[key] = row
        return {"evidences": [row for row in latest.values()
                              if row["valid_to"] is None or available_time(row["valid_to"]) > cutoff]}
