"""Read-only source adapter for explicitly authorized historical task records."""
from copy import deepcopy
from pathlib import Path
import json
import pandas as pd
from src.eval.protocol import decision_cutoff, eligible_record
from src.eval.records import digest
from src.eval.research import RESEARCH_VERSION


class ResearchSnapshotSource:
    """Read an exact successful full-graph task from an authorized research run."""

    def __init__(self, run_dir, output_dir):
        self.path = Path(run_dir).resolve()
        if Path(output_dir).resolve().is_relative_to(self.path):
            raise ValueError("Diagnostic output must be outside the source run")
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.identity = self.manifest["identity"]
        if (digest(self.identity) != self.manifest["identity_digest"]
                or self.identity.get("research_version") != RESEARCH_VERSION):
            raise ValueError("Invalid research run manifest")

    def load_task(self, ticker, date):
        if ticker not in self.identity["tickers"] or date not in self.identity["trade_dates"]:
            raise ValueError("Ticker/date is outside the source run")
        task_id = digest([ticker, date, "full_graph"])
        record = json.loads((self.path / "tasks" / (task_id + ".json")).read_text(encoding="utf-8"))
        inputs, snapshot = record["inputs"], record["decision_snapshot"]
        when = decision_cutoff(date)
        if (record["status"] != "success" or record["run_id"] != self.manifest["run_id"]
                or (record["ticker"], record["trade_date"], record["system"]) != (ticker, date, "full_graph")
                or inputs.get("research_version") != RESEARCH_VERSION or inputs["decision_time"] != when.isoformat()
                or digest(inputs) != record["input_digest"]
                or digest(snapshot) != record["decision_snapshot_digest"]
                or digest(snapshot) != inputs["snapshot_digest"]):
            raise ValueError("Case requires an intact successful full_graph record")
        if any(row.get("ticker") != ticker or not eligible_record(row, when)
               for field in ("evidences", "claims") for row in snapshot[field]):
            raise ValueError("Snapshot contains unavailable or out-of-scope evidence or claims")
        return record


class SavedRunSource:
    def __init__(self, run_dir):
        self.records = []
        for path in sorted((Path(run_dir) / "tasks").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if digest(record["inputs"]) != record["input_digest"]:
                raise ValueError("Source input integrity mismatch")
            snapshot = record.get("decision_snapshot")
            if snapshot is not None and digest(snapshot) != record.get("decision_snapshot_digest"):
                raise ValueError("Source snapshot integrity mismatch")
            if record["status"] == "success" and record["system"] in {"kg_dynamic", "static_kg"}:
                self.records.append(record)
        if not self.records:
            raise ValueError("No successful saved KG source records")

    def load_subgraph(self, ticker, cutoff):
        exact = [r for r in self.records if r["ticker"] == ticker and r["system"] == "kg_dynamic"
                 and decision_cutoff(r["trade_date"]) == cutoff]
        if not exact:
            exact = [r for r in self.records if r["ticker"] == ticker and r["system"] == "static_kg"
                     and decision_cutoff(r["inputs"].get("effective_cutoff", "1900-01-01")) == cutoff]
        if not exact:
            raise ValueError("No successful saved source for the requested ticker/cutoff")
        snapshots = [r.get("decision_snapshot") or r["inputs"].get("snapshot") for r in exact]
        if any(s is None for s in snapshots) or len({digest(s) for s in snapshots}) != 1:
            raise ValueError("Source snapshots are missing or disagree")
        snapshot = deepcopy(snapshots[0])
        for key in ("signals", "fundamentals", "risks", "news", "evidences", "claims"):
            if any(not eligible_record(row, cutoff) for row in snapshot.get(key, [])):
                raise ValueError("Saved source contains unavailable or unknown-time evidence")
        return snapshot

    def load_ohlcv(self, ticker):
        frames = [r["inputs"]["ohlcv"] for r in self.records if r["ticker"] == ticker and r["system"] == "kg_dynamic"]
        if not frames or len({digest(frame) for frame in frames}) != 1:
            raise ValueError("Saved market frames are missing or disagree")
        return pd.DataFrame(deepcopy(frames[0]))
