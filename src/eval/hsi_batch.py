"""Bounded stock-level parallel execution of the frozen HSI direction protocol."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from src.eval.hsi_daily import run_hsi, replay_hsi, validate_inputs
from src.eval.prediction_review import FrozenForecast, classification_metrics, labels_for
from src.eval.records import RunRecorder, code_identity, digest, model_identity, write_json
from src.eval.research_safety import public_value


def batch_summary(forecasts, market, expected):
    parsed = [FrozenForecast.model_validate(item) for item in forecasts]
    labels = labels_for(parsed, market)
    rows = [{**item.prediction(), **labels[item.ticker, item.date]} for item in parsed]
    return {"n_expected": expected, "n_success": len(rows), "n_failed": expected - len(rows),
            "n_unlabeled": sum(row["label"] is None for row in rows),
            **classification_metrics([row for row in rows if row["label"] is not None])}


def run_batch(universe, market, evidence, *, client_factory, output_dir, run_id, workers=4, resume=False):
    tickers, tape, _ = validate_inputs(universe, market, evidence)
    if not 1 <= workers <= 10:
        raise ValueError("Workers must be between one and ten")
    dates = [day for day in tape.sessions if day.startswith("2025-")]
    if not dates:
        raise ValueError("No 2025 sessions")
    evidence = public_value(evidence)
    identity = {"version": "hsi_batch_v1", "universe_digest": digest(universe),
                "market_digest": digest(market), "evidence_digest": digest(evidence),
                "tickers": tickers, "dates": dates, "variant": "full", "retrieval": "trace",
                "budget": 8192, "model": model_identity(client_factory()), "code": code_identity()}
    recorder = RunRecorder(output_dir, identity, run_id=run_id, resume=resume)
    write_json(recorder.path / "batch_inputs.json", {"universe": universe, "market": market, "evidence": evidence})
    results, errors = {}, {}

    def progress(status):
        write_json(recorder.path / "batch_progress.json", {"status": status, "n_expected": len(tickers) * len(dates),
                   "completed_stocks": sorted(results), "failed_stocks": errors,
                   "stock_summaries": results, "workers": workers})

    def stock_run(ticker):
        stock_id = ticker.replace(".", "_")
        shard_root = recorder.path / "shards"
        exists = (shard_root / "runs" / stock_id / "manifest.json").exists()
        result = run_hsi(universe, market, evidence, client=client_factory(), output_dir=shard_root,
                         run_id=stock_id, resume=exists, tickers=[ticker], dates=dates,
                         variants=["full"], retrieval="trace", budget=8192)
        # One pass retries only failed tasks; prior failed attempts remain archived.
        if result["summary"]["full"]["n_success"] < len(dates):
            result = run_hsi(universe, market, evidence, client=client_factory(), output_dir=shard_root,
                             run_id=stock_id, resume=True, tickers=[ticker], dates=dates,
                             variants=["full"], retrieval="trace", budget=8192)
        replay_hsi(result["run_dir"])
        return result["summary"]["full"]

    progress("running")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(stock_run, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:
                errors[ticker] = type(exc).__name__
            progress("running")
    forecasts = []
    for ticker in tickers:
        path = recorder.path / "shards/runs" / ticker.replace(".", "_") / "full_forecasts.json"
        if ticker in results:
            forecasts.extend(json.loads(path.read_text(encoding="utf-8")))
    summary = batch_summary(forecasts, market, len(tickers) * len(dates))
    summary.update(stock_summaries=results, shard_errors=errors)
    write_json(recorder.path / "full_forecasts.json", forecasts)
    write_json(recorder.path / "batch_summary.json", summary)
    progress("complete" if not errors else "completed_with_shard_errors")
    return {"run_dir": str(recorder.path), **summary}


def replay_batch(run_dir):
    root = Path(run_dir)
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    manifest, data = read("manifest.json"), read("batch_inputs.json")
    identity = manifest["identity"]
    if identity.get("version") != "hsi_batch_v1" or digest(identity) != manifest["identity_digest"]:
        raise ValueError("Invalid batch manifest")
    for field in ("universe", "market", "evidence"):
        if digest(data[field]) != identity[field + "_digest"]:
            raise ValueError("Batch inputs changed")
    summary = read("batch_summary.json")
    forecasts = []
    for ticker in identity["tickers"]:
        if ticker in summary["stock_summaries"]:
            shard = root / "shards/runs" / ticker.replace(".", "_")
            result = replay_hsi(shard)
            if result["full"] != summary["stock_summaries"][ticker]:
                raise ValueError("Batch stock summary differs")
            forecasts.extend(json.loads((shard / "full_forecasts.json").read_text(encoding="utf-8")))
    if forecasts != read("full_forecasts.json"):
        raise ValueError("Batch forecast export differs")
    calculated = batch_summary(forecasts, data["market"], len(identity["tickers"]) * len(identity["dates"]))
    if any(summary[key] != value for key, value in calculated.items()):
        raise ValueError("Batch aggregate summary differs")
    return summary
