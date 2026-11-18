"""Small, explicitly selected issuer corpus; not an exhaustive news archive."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import re
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from src.eval.records import write_json
from src.eval.thesis_sources import SourceRecord, LocalEvidence

SEEDS = [
    ("0005.HK", "https://www.hsbc.com/news-and-views/news/hsbc-news-archive/hsbc-buys-silicon-valley-bank-uk"),
    ("0005.HK", "https://www.hsbc.com/news-and-views/news/hsbc-news-archive/canada-sale-completes"),
    ("0700.HK", "https://www.tencent.com/tencent-unveils-new-ai-upgrades-proprietary-innovations-and-global-solutions/"),
    ("0700.HK", "https://www.tencent.com/zh-hk/mobily-tencent-to-spearhead-global-digital-ecosystem-in-saudi-arabia/"),
]
HOSTS = {"www.hsbc.com": "0005.HK", "www.tencent.com": "0700.HK"}


def parse_issuer_html(ticker, url, html):
    """Use article-local visible date/body, excluding navigation/recommendations."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or HOSTS.get(parsed.hostname) != ticker or parsed.query or parsed.fragment:
        raise ValueError("Expected a clean, matching official issuer HTTPS URL")
    soup = BeautifulSoup(html, "html.parser")
    if parsed.hostname == "www.hsbc.com":
        body = soup.select_one("main .article-sublayout")
        stamp = soup.select_one("main .link-to-author-page__date")
        formats = ("%d %b %Y",)
    else:
        article = soup.select_one("main article")
        body = article.select_one(".entry-content") if article else None
        stamp = article.select_one("header .entry-date") if article else None
        formats = ("%B %d, %Y", "%Y年%m月%d日")
    if body is None or stamp is None:
        raise ValueError("Missing article body or publication date")
    date_text = stamp.get_text(" ", strip=True)
    day = None
    for fmt in formats:
        try:
            day = datetime.strptime(date_text, fmt).date()
            break
        except ValueError:
            pass
    if day is None or not 2023 <= day.year <= 2025:
        raise ValueError("Unknown or out-of-protocol publication date")
    # UTC next-day midnight is conservative even for these publishers' local days.
    available = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    for tag in body.select("script,style,nav,aside,form"):
        tag.decompose()
    # Normalize HTML layout whitespace before model input and content hashing.
    # Raw HTML remains saved independently for provenance checks.
    text = " ".join(body.get_text(" ", strip=True).split())
    if len(text) < 100:
        raise ValueError("Article body is empty or truncated")
    title = soup.select_one("h1")
    title_text = title.get_text(" ", strip=True) if title else ""
    record = SourceRecord(ticker=ticker, source_type="local_news", source=url,
                          document_id=url, version=sha256(text.encode()).hexdigest(),
                          language="zh-Hant" if re.search(r"[\u4e00-\u9fff]", title_text) else "en",
                          published_at=available.isoformat(), text=title_text + "\n" + text)
    return record.model_dump(), {"url": url, "ticker": ticker, "title": title_text,
        "publication_date": day.isoformat(), "availability_policy": "next_day_00_UTC_date_only",
        "body_sha256": sha256(text.encode()).hexdigest(), "text_normalization": "html_whitespace_v1"}


def collect_public_disclosures(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    accepted, rejected, provenance = [], [], []
    with requests.Session() as session:
        for ticker, url in SEEDS:
            try:
                response = session.get(url, timeout=30, allow_redirects=False)
                response.raise_for_status()
                if response.status_code != 200:
                    raise ValueError("Redirect or non-document response")
                record, audit = parse_issuer_html(ticker, url, response.content)
                raw_path = output / (sha256(url.encode()).hexdigest() + ".html")
                raw_path.write_bytes(response.content)
                audit.update(raw_file=raw_path.name, raw_sha256=sha256(response.content).hexdigest(),
                             retrieved_at=datetime.now(timezone.utc).isoformat())
                accepted.append(record)
                provenance.append(audit)
            except Exception as exc:
                rejected.append({"ticker": ticker, "url": url, "error_type": type(exc).__name__})
    if accepted:
        parsed = LocalEvidence(accepted)
        if parsed.rejected:
            raise ValueError("Collected records failed source validation")
    write_json(output / "evidence.json", accepted)
    report = {"n_sources": len(accepted), "rejected": rejected, "sources": provenance,
        "selection": "manually_selected_official_issuer_seed_corpus_not_exhaustive",
        "limitations": ["Two issuers only; no full-universe or monthly coverage claim.",
                        "Current historical pages; original archived versions are not independently verified.",
                        "Relation extraction and independent verification are required before graph admission."]}
    write_json(output / "collection_report.json", report)
    return report
