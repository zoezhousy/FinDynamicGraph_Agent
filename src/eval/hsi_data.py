"""Explicit HSI source preparation and point-in-time technical evidence."""
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path

import pandas as pd
import requests

from src.eval.records import digest, write_json

UNIVERSE_SOURCE = "https://www.hsi.com.hk/static/uploads/contents/en/dl_centre/blog/20240709T000000.pdf"
CONSTITUENTS = [
    ("0700.HK", "Tencent", 8.3), ("0005.HK", "HSBC", 8.2),
    ("9988.HK", "Alibaba", 7.5), ("3690.HK", "Meituan", 6.4),
    ("1299.HK", "AIA", 5.3), ("0939.HK", "China Construction Bank", 5.2),
    ("0941.HK", "China Mobile", 4.3), ("1398.HK", "ICBC", 3.3),
    ("0883.HK", "CNOOC", 3.3), ("3988.HK", "Bank of China", 2.9),
]


def technical_source(ticker, day, tape):
    rows = [(session, tape.prices.get((ticker, session), {}).get("close"))
            for session in tape.sessions if session <= day]
    current = tape.prices.get((ticker, day), {}).get("close")
    if current is None:
        return None
    recent = rows[-20:]
    closes = [value for _, value in recent if value is not None]
    prior = rows[-2][1] if len(rows) >= 2 else None
    values = {"date": day, "close": current, "prior_session_close": prior,
              "one_session_return": current / prior - 1 if prior else None,
              "ma20": sum(closes) / 20 if len(recent) == 20 and len(closes) == 20 else None}
    import json
    return {"ticker": ticker, "source_type": "technical", "source": "derived_daily_prices",
            "document_id": f"technical-{ticker}-{day}", "version": "1", "language": "en",
            "published_at": day + "T16:10:00+08:00", "period": day,
            "text": json.dumps(values, sort_keys=True), "valid_to": None}


def prepare_hsi(output_dir, *, news_files=()):
    """Download prices into a new directory; never substitute recent news for archives."""
    import exchange_calendars as calendars
    import yfinance as yf
    from src.eval.thesis_sources import LocalEvidence, read_json_input

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    calendar = calendars.get_calendar("XHKG", start="2023-01-01", end="2025-12-31")
    sessions = [day.strftime("%Y-%m-%d") for day in calendar.sessions]
    prices, coverage, errors = [], [], []
    for ticker, name, weight in CONSTITUENTS:
        try:
            frame = yf.Ticker(ticker).history(start="2023-01-01", end="2026-01-01", auto_adjust=False,
                                             actions=True, timeout=30, raise_errors=True)
            seen = set()
            for stamp, row in frame.iterrows():
                day = stamp.strftime("%Y-%m-%d")
                if day not in sessions:
                    continue
                def mark(field):
                    value = row.get(field)
                    return float(value) if pd.notna(value) and float(value) > 0 else None
                prices.append({"ticker": ticker, "date": day, "open": mark("Open"),
                               "close": mark("Close"), "adjusted_close": mark("Adj Close"),
                               "volume": float(row["Volume"]) if pd.notna(row.get("Volume")) else None})
                seen.add(day)
            coverage.append({"ticker": ticker, "n_sessions": len(sessions), "n_bars": len(seen),
                             "missing_sessions": sorted(set(sessions) - seen)})
        except Exception as exc:
            errors.append({"ticker": ticker, "stage": "yfinance", "error_type": type(exc).__name__})
    evidence, news_hashes = [], []
    for path in news_files:
        payload, fingerprint = read_json_input(Path(path))
        LocalEvidence(payload)
        evidence.extend(payload)
        news_hashes.append(fingerprint)
    write_json(output / "universe.json", {"published_at": "2024-07-09", "weights_as_of": "2024-07-05",
               "selection": "official_top10_before_test", "source": UNIVERSE_SOURCE,
               "constituents": [{"ticker": t, "name": n, "weight": w} for t, n, w in CONSTITUENTS]})
    market = {"sessions": sessions, "prices": prices}
    write_json(output / "market.json", market)
    write_json(output / "evidence.json", evidence)
    report = {"calendar": "XHKG", "price_source": "yfinance", "price_basis": "unadjusted_close",
              "calendar_package_version": importlib.metadata.version("exchange_calendars"),
              "retrieved_at": datetime.now(timezone.utc).isoformat(), "market_digest": digest(market),
              "news_input_hashes": news_hashes, "news_rows": len(evidence),
              "coverage": coverage, "errors": errors,
              "limitations": ["Historical Yahoo price revisions and corporate actions need independent checks.",
                              "No imported news means news coverage is missing, not neutral sentiment.",
                              "Selection uses July 2024 weights, not December 2024 weights."]}
    write_json(output / "preparation_report.json", report)
    return report


def collect_tushare_news(output, start, end, *, sources=("sina",)):
    """Split full result windows; credentials remain in process environment."""
    from src.eval.protocol import available_time
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("TUSHARE_TOKEN is required for authorized historical news")
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    aliases = {"0700.HK": ["腾讯", "Tencent"], "0005.HK": ["汇丰", "滙豐", "HSBC"],
               "9988.HK": ["阿里巴巴", "Alibaba"], "3690.HK": ["美团", "美團", "Meituan"],
               "1299.HK": ["友邦", "AIA"], "0939.HK": ["建设银行", "建設銀行"],
               "0941.HK": ["中国移动", "中國移動"], "1398.HK": ["工商银行", "工商銀行"],
               "0883.HK": ["中国海洋石油", "中國海洋石油", "中海油", "CNOOC"],
               "3988.HK": ["中国银行", "中國銀行", "Bank of China"]}
    result, seen = [], set()
    with requests.Session() as session:
        def window(src, left, right):
            response = session.post("https://api.tushare.pro", json={"api_name": "news", "token": token,
                                    "params": {"src": src, "start_date": left.strftime("%Y-%m-%d %H:%M:%S"),
                                               "end_date": right.strftime("%Y-%m-%d %H:%M:%S")}}, timeout=60)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                raise RuntimeError("Tushare news query failed; check account permission")
            fields, items = data["data"]["fields"], data["data"]["items"]
            if len(items) >= 1500:
                if (right - left).total_seconds() <= 1:
                    raise ValueError("News window remains truncated at one second")
                middle = left + (right - left) // 2
                window(src, left, middle)
                window(src, middle, right)
                return
            for values in items:
                row = dict(zip(fields, values))
                text = str(row.get("title") or "") + "\n" + str(row.get("content") or "")
                published = available_time(row.get("datetime"))
                if published is None or not left <= pd.Timestamp(published).tz_convert("Asia/Hong_Kong").tz_localize(None) <= right:
                    continue
                for ticker, names in aliases.items():
                    if not any(name.casefold() in text.casefold() for name in names):
                        continue
                    key = digest([ticker, src, published.isoformat(), text])
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append({"ticker": ticker, "source_type": "local_news", "source": "tushare_" + src,
                                   "document_id": key, "version": "1", "language": "bilingual",
                                   "published_at": published.isoformat(), "text": text})
        for src in sources:
            for day in pd.date_range(start, end, freq="D"):
                window(src, day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    write_json(path, result)
    return {"news_rows": len(result), "company_matching": "explicit_alias_candidates_require_audit"}
