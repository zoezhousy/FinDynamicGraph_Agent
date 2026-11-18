"""Run the small four-system temporal-KG preliminary experiment.

Default scope: one ticker, five event-rich dates, and the canonical systems
``direct_llm``, ``text_rag``, ``static_kg``, and ``kg_dynamic``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from dotenv import load_dotenv

from src.agents.kg_tools import KGAgentContext
from src.agents.orchestrator import KGBasedOrchestrator, OrchestratorConfig
from src.eval.baselines import (
    _filter_news_by_trade_date,
    baseline_evidence_no_kg,
    baseline_no_kg_no_evidence,
    baseline_static_kg,
)
from src.eval.temporal_audit import (
    TemporalAuditResult,
    assert_no_future,
    audit_items,
    audit_subgraph,
    snapshot_counts,
    snapshot_fingerprint,
)
from src.kg.query import KGQueryClient
from src.sim.backtest import BacktestConfig, compute_trade_return


logger = logging.getLogger(__name__)

SYSTEMS = ("direct_llm", "text_rag", "static_kg", "kg_dynamic")

RESULT_COLUMNS = (
    "ticker",
    "trade_date",
    "system",
    "action",
    "confidence",
    "final_score",
    "decision_reason",
    "conflict_level",
    "evidence_refs",
    "claim_refs",
    "retrieved_evidence_ids",
    "retrieved_claim_ids",
    "retrieved_count",
    "valid_count",
    "stale_count",
    "future_count",
    "temporal_precision",
    "stale_evidence_rate",
    "future_evidence_rate",
    "evidence_coverage",
    "retrieval_latency",
    "total_latency",
    "snapshot_hash",
    "snapshot_counts",
    "snapshot_changed_from_previous",
    "trade_executed",
    "raw_return",
    "error",
)


def run_preliminary(
    *,
    ticker: str,
    trade_dates: list[datetime],
    data_root: Path,
    output_dir: Path,
    static_cutoff: datetime,
    database: str = "neo4j",
    staleness_days: int = 30,
    experiment_config=None, query_client=None, model_client=None, llm=None,
) -> pd.DataFrame:
    """Execute and persist the preliminary comparison."""
    if not trade_dates:
        raise ValueError("At least one trade date is required")

    if experiment_config is not None and experiment_config.protocol_version == "pit_v1":
        return _run_preliminary_pit(ticker=ticker, trade_dates=trade_dates,
            data_root=data_root, output_dir=output_dir, static_cutoff=static_cutoff,
            database=database, staleness_days=staleness_days,
            experiment_config=experiment_config, query_client=query_client,
            model_client=model_client, llm=llm)

    with ExitStack() as resources:
        client = query_client or _connect_query_client(database)
        if query_client is None:
            resources.callback(client.close)
        context = KGAgentContext(client)
        orchestrator = KGBasedOrchestrator(
            context,
            OrchestratorConfig(persist_decision_trace=False),
            model_client=model_client, llm=llm,
        )
        resources.callback(orchestrator.close)
        ohlcv = _load_ohlcv(data_root, ticker)
        backtest_config = BacktestConfig()

        rows: list[dict[str, Any]] = []
        previous_dynamic_hash: str | None = None
        dynamic_hashes: list[str] = []
        for trade_date in sorted(trade_dates):
            date_text = trade_date.date().isoformat()
            logger.info("Running %s at %s", ticker, date_text)

            direct = _timed_direct_llm(ticker, trade_date)
            rows.append(
                _normalise_result(
                    direct[0], ticker, trade_date, "direct_llm",
                    _empty_audit(), direct[1], direct[1], None, None,
                    ohlcv, backtest_config,
                )
            )

            text_result, text_items, retrieval_seconds, total_seconds = _timed_text_rag(
                ticker, trade_date, data_root
            )
            text_audit = audit_items(
                text_items, trade_date, staleness_days=staleness_days
            )
            assert_no_future(text_audit, context=f"text_rag {ticker} {date_text}")
            rows.append(
                _normalise_result(
                    text_result, ticker, trade_date, "text_rag", text_audit,
                    retrieval_seconds, total_seconds, None, None,
                    ohlcv, backtest_config,
                )
            )

            static_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            static_snapshot = context.load_subgraph(ticker, static_cutoff)
            static_retrieval = time.perf_counter() - retrieval_start
            static_audit = audit_subgraph(
                static_snapshot, trade_date, staleness_days=staleness_days
            )
            assert_no_future(static_audit, context=f"static_kg {ticker} {date_text}")
            static_result = baseline_static_kg(
                ticker,
                trade_date,
                kg_context=context,
                static_cutoff=static_cutoff,
            )
            static_total = time.perf_counter() - static_start
            rows.append(
                _normalise_result(
                    static_result, ticker, trade_date, "static_kg", static_audit,
                    static_retrieval, static_total,
                    snapshot_fingerprint(static_snapshot), snapshot_counts(static_snapshot),
                    ohlcv, backtest_config,
                )
            )

            dynamic_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            dynamic_snapshot = context.load_subgraph(ticker, trade_date)
            dynamic_retrieval = time.perf_counter() - retrieval_start
            dynamic_audit = audit_subgraph(
                dynamic_snapshot, trade_date, staleness_days=staleness_days
            )
            assert_no_future(dynamic_audit, context=f"kg_dynamic {ticker} {date_text}")
            dynamic_hash = snapshot_fingerprint(dynamic_snapshot)
            changed = None if previous_dynamic_hash is None else dynamic_hash != previous_dynamic_hash
            dynamic_result = orchestrator.run_for_ticker(ticker, trade_date)
            dynamic_total = time.perf_counter() - dynamic_start
            rows.append(
                _normalise_result(
                    dynamic_result, ticker, trade_date, "kg_dynamic", dynamic_audit,
                    dynamic_retrieval, dynamic_total, dynamic_hash,
                    snapshot_counts(dynamic_snapshot), ohlcv, backtest_config,
                    snapshot_changed=changed,
                )
            )
            dynamic_hashes.append(dynamic_hash)
            previous_dynamic_hash = dynamic_hash

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    _write_results(df, output_dir)

    if len(dynamic_hashes) > 1 and len(set(dynamic_hashes)) == 1:
        raise AssertionError(
            "Dynamic KG snapshots did not change across the selected dates. "
            "Inspect ingestion coverage or choose different event dates."
        )
    return df



def _run_preliminary_pit(*, ticker, trade_dates, data_root, output_dir, static_cutoff,
    database, staleness_days, experiment_config, query_client=None, model_client=None, llm=None):
    from dataclasses import replace
    from src.config import CollectionConfig
    from src.eval.runner import run_grid
    from src.eval.protocol import decision_cutoff
    from src.eval.records import digest
    mapping = {"no_kg_no_evidence": "direct_llm", "evidence_no_kg": "text_rag", "static_kg": "static_kg", "kg_dynamic": "kg_dynamic"}
    cfg = replace(experiment_config, output_dir=output_dir)
    df = run_grid([ticker], [d.date().isoformat() for d in trade_dates],
        CollectionConfig(tickers=[ticker], output_root=data_root, neo4j_database=database),
        static_cutoff.isoformat(), experiment_config=cfg, query_client=query_client,
        model_client=model_client, llm=llm, persist_outcomes=False)
    run_dir = Path(df.attrs["run_dir"])
    records = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "tasks").glob("*.json")]
    order = {"no_kg_no_evidence": 0, "evidence_no_kg": 1, "static_kg": 2, "kg_dynamic": 3}
    records.sort(key=lambda record: (record["trade_date"], order[record["system"]]))
    rows, hashes = [], []
    previous = None
    for record in records:
        inputs = record["inputs"]
        graph = inputs.get("snapshot")
        cutoff = decision_cutoff(record["trade_date"])
        audit = audit_subgraph(graph, cutoff, staleness_days=staleness_days) if graph is not None else audit_items(inputs.get("text_items", []), cutoff, staleness_days=staleness_days)
        assert_no_future(audit, context=record["system"])
        fingerprint = digest(graph) if graph is not None else None
        changed = None
        if record["system"] == "kg_dynamic" and record["status"] != "failed":
            changed = None if previous is None else fingerprint != previous
            previous = fingerprint
            hashes.append(fingerprint)
        timing = record["timings"]
        row = _normalise_result(record["decision"] or {"action": "abstain"}, ticker,
            datetime.fromisoformat(record["trade_date"]), mapping[record["system"]], audit,
            timing.get("retrieval_latency", 0), timing.get("total_latency", 0),
            fingerprint, snapshot_counts(graph) if graph is not None else None,
            pd.DataFrame(inputs.get("ohlcv", []), columns=["date", "open", "close"]), BacktestConfig(**inputs["backtest_config"]), snapshot_changed=changed)
        if record["status"] != "success":
            row["error"] = record["issues"]
        if record.get("backtest") is not None:
            row["raw_return"] = record["backtest"]["raw_return"]
        rows.append(row)
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result.attrs.update(df.attrs)
    _write_results(result, run_dir)
    if len(hashes) > 1 and len(set(hashes)) == 1:
        raise AssertionError("Dynamic KG snapshots did not change across selected dates")
    return result


def select_event_rich_dates(
    ticker: str,
    data_root: Path,
    *,
    count: int = 5,
) -> list[datetime]:
    """Choose the highest-news-volume dates that also have market data."""
    ticker_dir = data_root / ticker
    news_frames: list[pd.DataFrame] = []
    for filename in (
        "news_history.parquet",
        "news_combined_latest.parquet",
        "news_latest.parquet",
    ):
        path = ticker_dir / filename
        if path.exists():
            news_frames.append(pd.read_parquet(path))
    if not news_frames:
        raise FileNotFoundError(f"No news parquet found under {ticker_dir}")

    news = pd.concat(news_frames, ignore_index=True)
    time_col = next(
        (column for column in ("published_time", "published_at", "date") if column in news.columns),
        None,
    )
    if time_col is None:
        raise ValueError("News data has no published_time/published_at/date column")
    dedupe_columns = [
        column for column in (time_col, "url", "title") if column in news.columns
    ]
    news = news.drop_duplicates(subset=dedupe_columns)
    published = pd.to_datetime(news[time_col], errors="coerce", utc=True)
    news = news.loc[published.notna()].copy()
    news["_event_date"] = published.loc[published.notna()].dt.tz_convert(None).dt.normalize()

    ohlcv = _load_ohlcv(data_root, ticker)
    market_dates = set(pd.to_datetime(ohlcv["date"]).dt.normalize())
    counts = news.groupby("_event_date").size().sort_values(ascending=False)
    selected = [date for date in counts.index if date in market_dates][:count]
    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} event-rich market dates found for {ticker}; requested {count}"
        )
    return [value.to_pydatetime() for value in sorted(selected)]


def _timed_direct_llm(
    ticker: str, trade_date: datetime
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = baseline_no_kg_no_evidence(ticker, trade_date)
    return result, time.perf_counter() - started


def _timed_text_rag(
    ticker: str,
    trade_date: datetime,
    data_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float]:
    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    items = _load_text_evidence_items(ticker, trade_date, data_root)
    retrieval_seconds = time.perf_counter() - retrieval_started
    result = baseline_evidence_no_kg(ticker, trade_date, data_root=data_root)
    total_seconds = time.perf_counter() - started
    result["retrieved_evidence_ids"] = [str(item["evidence_id"]) for item in items]
    return result, items, retrieval_seconds, total_seconds


def _load_text_evidence_items(
    ticker: str,
    trade_date: datetime,
    data_root: Path,
) -> list[dict[str, Any]]:
    """Return the exact dated records available to the text-RAG baseline."""
    ticker_dir = data_root / ticker
    items: list[dict[str, Any]] = []

    ohlcv = _load_ohlcv(data_root, ticker)
    dates = pd.to_datetime(ohlcv["date"], errors="coerce")
    recent = ohlcv.loc[dates <= pd.Timestamp(trade_date)].tail(20)
    for index, (_, row) in enumerate(recent.iterrows()):
        date_value = pd.Timestamp(row["date"]).to_pydatetime()
        items.append(
            {
                "evidence_id": f"ohlcv:{ticker}:{date_value.date().isoformat()}:{index}",
                "as_of_date": date_value,
                "valid_from": date_value,
                "source_type": "ohlcv",
            }
        )

    news_path = ticker_dir / "news_latest.parquet"
    if not news_path.exists():
        news_path = ticker_dir / "news_combined_latest.parquet"
    if news_path.exists():
        news = _filter_news_by_trade_date(pd.read_parquet(news_path), trade_date)
        for index, (_, row) in enumerate(news.iterrows()):
            published = row.get("published_time") or row.get("published_at") or row.get("date")
            raw_id = row.get("evidence_id") or row.get("url") or row.get("title") or index
            items.append(
                {
                    "evidence_id": f"text:{ticker}:{raw_id}",
                    "published_at": published,
                    "valid_from": published,
                    "source_type": "news",
                }
            )
    return items


def _normalise_result(
    result: Mapping[str, Any],
    ticker: str,
    trade_date: datetime,
    system: str,
    audit: TemporalAuditResult,
    retrieval_latency: float,
    total_latency: float,
    snapshot_hash: str | None,
    counts: dict[str, int] | None,
    ohlcv: pd.DataFrame,
    backtest_config: BacktestConfig,
    *,
    snapshot_changed: bool | None = None,
) -> dict[str, Any]:
    action = str(result.get("action") or "abstain")
    backtest = compute_trade_return(
        ohlcv,
        trade_date.date().isoformat(),
        action,
        backtest_config,
    )
    evidence_refs = _as_list(result.get("evidence_refs"))
    retrieved_ids = _as_list(result.get("retrieved_evidence_ids"))
    coverage = 1.0 if evidence_refs or retrieved_ids else 0.0
    audit_fields = audit.to_dict()
    return {
        "ticker": ticker,
        "trade_date": trade_date.date().isoformat(),
        "system": system,
        "action": action,
        "confidence": float(result.get("confidence") or 0.0),
        "final_score": float(result.get("final_score") or 0.0),
        "decision_reason": str(result.get("decision_reason") or ""),
        "conflict_level": float(result.get("conflict_level") or 0.0),
        "evidence_refs": evidence_refs,
        "claim_refs": _as_list(result.get("claim_refs")),
        "retrieved_evidence_ids": retrieved_ids,
        "retrieved_claim_ids": _as_list(result.get("retrieved_claim_ids")),
        "retrieved_count": audit_fields["retrieved_count"],
        "valid_count": audit_fields["valid_count"],
        "stale_count": audit_fields["stale_count"],
        "future_count": audit_fields["future_count"],
        "temporal_precision": audit_fields["temporal_precision"],
        "stale_evidence_rate": audit_fields["stale_evidence_rate"],
        "future_evidence_rate": audit_fields["future_evidence_rate"],
        "evidence_coverage": coverage,
        "retrieval_latency": round(retrieval_latency, 6),
        "total_latency": round(total_latency, 6),
        "snapshot_hash": snapshot_hash,
        "snapshot_counts": counts or {},
        "snapshot_changed_from_previous": snapshot_changed,
        "trade_executed": bool(backtest.get("trade_executed")),
        "raw_return": float(backtest.get("raw_return") or 0.0),
        "error": None,
    }


def _connect_query_client(database: str) -> KGQueryClient:
    required = ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    client = KGQueryClient(
        os.environ["NEO4J_URI"],
        os.environ["NEO4J_USER"],
        os.environ["NEO4J_PASSWORD"],
        database=database,
    )
    try:
        client._driver.verify_connectivity()
    except Exception:
        client.close()
        raise RuntimeError(
            "Neo4j is configured but not reachable. Start Neo4j before running the preliminary experiment."
        ) from None
    return client


def _load_ohlcv(data_root: Path, ticker: str) -> pd.DataFrame:
    for filename in ("ohlcv_2021_now.parquet", "ohlcv_2021_2025.parquet"):
        path = data_root / ticker / filename
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No OHLCV parquet found for {ticker} under {data_root}")


def _write_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "preliminary_results.jsonl"
    csv_path = output_dir / "preliminary_results.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in df.to_dict("records"):
            handle.write(json.dumps(row, default=_json_default, ensure_ascii=False) + "\n")

    csv_df = df.copy()
    for column in csv_df.columns:
        if csv_df[column].map(lambda value: isinstance(value, (list, dict))).any():
            csv_df[column] = csv_df[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict)) else value
            )
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s and %s", jsonl_path, csv_path)


def _empty_audit() -> TemporalAuditResult:
    return audit_items([], datetime.min)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parse_dates(values: Iterable[str]) -> list[datetime]:
    return [datetime.fromisoformat(value.strip()) for value in values if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("legacy", "pit_v1"), default="legacy")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ticker", default="0700.HK")
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Four to six ISO dates. Omit to auto-select event-rich dates.",
    )
    parser.add_argument("--date-count", type=int, choices=(4, 5, 6), default=5)
    parser.add_argument("--static-cutoff", default="2025-01-01")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/market_news"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--staleness-days", type=int, default=30)
    parser.add_argument("--database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    load_dotenv()
    args = build_parser().parse_args()
    trade_dates = (
        _parse_dates(args.dates)
        if args.dates
        else select_event_rich_dates(args.ticker, args.data_root, count=args.date_count)
    )
    if not 4 <= len(trade_dates) <= 6:
        raise ValueError("Preliminary scope must contain 4 to 6 dates")
    logger.info("Selected dates: %s", [value.date().isoformat() for value in trade_dates])
    from src.config import ExperimentConfig
    df = run_preliminary(
        ticker=args.ticker,
        trade_dates=trade_dates,
        data_root=args.data_root,
        output_dir=args.output_dir,
        static_cutoff=datetime.fromisoformat(args.static_cutoff),
        database=args.database,
        staleness_days=args.staleness_days,
        experiment_config=ExperimentConfig(protocol_version=args.protocol, output_dir=args.output_dir, run_id=args.run_id, resume=args.resume),
    )
    print(
        df.groupby("system")[[
            "temporal_precision",
            "stale_evidence_rate",
            "future_evidence_rate",
            "evidence_coverage",
            "retrieval_latency",
            "total_latency",
        ]].mean().round(4).to_string()
    )


if __name__ == "__main__":
    main()
