"""Frozen text-RAG input under the explicit Hong Kong cutoff policy."""
from pathlib import Path
import pandas as pd
from src.eval.protocol import eligible_record, available_time, HK

def load_text_input(ticker, cutoff, data_root, ohlcv):
    rows = ohlcv.sort_values("date").copy()
    dates = pd.to_datetime(rows["date"], errors="coerce")
    # A daily bar becomes available after the Hong Kong market close.
    times = dates.map(lambda d: None if pd.isna(d) else available_time(str(d.date()) + "T16:10:00+08:00"))
    rows = rows.loc[times.map(lambda d: d is not None and d <= cutoff)].tail(20)
    items = []
    lines = []
    for _, row in rows.iterrows():
        date = pd.Timestamp(row["date"]).date().isoformat()
        items.append({"evidence_id": f"ohlcv:{ticker}:{date}", "published_at": available_time(date + "T16:10:00+08:00").isoformat(), "source_type": "ohlcv"})
        lines.append(f"  {date}: O={row.get('open', 0):.2f} H={row.get('high', 0):.2f} L={row.get('low', 0):.2f} C={row.get('close', 0):.2f} V={row.get('volume', 0)}")
    headlines = []
    selected = None
    for name in ("news_latest.parquet", "news_combined_latest.parquet"):
        path = Path(data_root) / ticker / name
        if path.exists():
            selected = str(path)
            frame = pd.read_parquet(path)
            candidates = []
            for index, row in enumerate(frame.to_dict("records")):
                field = next((k for k in ("published_time", "published_at", "date") if row.get(k) is not None), None)
                row["published_at"] = available_time(row.get(field)) if field else None
                if eligible_record(row, cutoff):
                    candidates.append((row["published_at"], index, row))
            candidates.sort(key=lambda value: value[0], reverse=True)
            for published, index, row in candidates[:10]:
                eid = str(row.get("evidence_id") or row.get("url") or row.get("title") or index)
                items.append({"evidence_id": f"text:{ticker}:{eid}", "published_at": published.isoformat(), "source_type": "news", "record": row})
                if row.get("title"):
                    headlines.append(f"  - [{row.get('source') or ''}] {row['title']} ({published.isoformat()})")
            break
    return ("\n".join(lines) or "No OHLCV data available.", "\n".join(headlines) or "No news available."), items, selected
