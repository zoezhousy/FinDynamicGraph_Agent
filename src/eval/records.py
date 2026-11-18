"""Isolated, atomic experiment records. No credentials or environment dump."""
from datetime import date, datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import re
import subprocess
import uuid

SCHEMA_VERSION = 1

def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")

def digest(value):
    payload = json.dumps(json_value(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def code_identity():
    root = Path(__file__).resolve().parents[2]
    hasher = hashlib.sha256()
    paths = sorted((root / "src").rglob("*.py"))
    for extra in ("requirements.txt",):
        if (root / extra).exists():
            paths.append(root / extra)
    for path in paths:
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")), "source_digest": hasher.hexdigest()}

def model_identity(client=None, llm=None):
    result = {key: os.getenv(key) for key in ("LLM_MODEL", "AGENT_BACKEND", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "LLM_TIMEOUT_SECONDS")}
    config = getattr(client, "config", None)
    if config is not None:
        result["direct_config"] = {key: getattr(config, key, None) for key in ("model", "temperature", "max_tokens", "timeout_seconds")}
        thinking = getattr(config, "deepseek_thinking", None)
        if thinking is not None:
            result["direct_config"]["deepseek_thinking"] = thinking
    if llm is not None:
        result["graph_config"] = {key: getattr(llm, key, None) for key in ("model_name", "temperature", "max_tokens")}
    return result

class RunRecorder:
    def __init__(self, output_dir, manifest, *, run_id=None, resume=False):
        self.run_id = run_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", self.run_id):
            raise ValueError("Invalid run_id")
        self.path = Path(output_dir) / "runs" / self.run_id
        self.manifest_path = self.path / "manifest.json"
        identity = json_value(manifest)
        if resume:
            if not self.manifest_path.exists():
                raise FileNotFoundError("Cannot resume a missing run")
            saved = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if saved["identity_digest"] != digest(identity):
                raise ValueError("Run configuration, model or source identity changed; start a new run")
        else:
            self.path.mkdir(parents=True, exist_ok=False)
            write_json(self.manifest_path, {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "created_at": datetime.now(timezone.utc), "identity": identity, "identity_digest": digest(identity)})

    def task_path(self, ticker, trade_date, system):
        return self.path / "tasks" / (digest([ticker, trade_date, system]) + ".json")

    def existing(self, ticker, trade_date, system):
        path = self.task_path(ticker, trade_date, system)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def previous(self, ticker, trade_date, system, inputs):
        record = self.existing(ticker, trade_date, system)
        if record is None:
            return None
        if record["input_digest"] != digest(inputs):
            old = record["inputs"]
            extended_failed_input = record["status"] == "failed" and "snapshot" not in old and all(k in inputs and digest(v) == digest(inputs[k]) for k, v in old.items())
            if not extended_failed_input:
                raise ValueError("Task input snapshot changed; start a new run")
        return record

    def save(self, ticker, trade_date, system, inputs, *, status, row=None, decision=None, calls=(), issues=(), backtest=None, audit=None, previous=None, timings=None, decision_snapshot=None):
        record = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "ticker": ticker, "trade_date": trade_date, "system": system, "status": status, "attempt": (previous or {}).get("attempt", 0) + 1, "inputs": inputs, "input_digest": digest(inputs), "decision": decision, "row": row, "calls": list(calls), "issues": list(issues), "backtest": backtest, "audit": audit, "timings": timings or {}}
        record["decision_snapshot"] = json_value(decision_snapshot)
        record["decision_snapshot_digest"] = digest(decision_snapshot) if decision_snapshot is not None else None
        if previous is not None:
            write_json(self.path / "attempts" / (digest([ticker, trade_date, system]) + f"_{previous['attempt']}.json"), previous)
        write_json(self.task_path(ticker, trade_date, system), record)
        return record

    def finish(self, records):
        counts = {status: sum(r["status"] == status for r in records) for status in ("success", "degraded", "failed")}
        write_json(self.path / "run_summary.json", {"run_id": self.run_id, "expected_tasks": len(records), "status_counts": counts, "technical_failure_rate": (counts["failed"] + counts["degraded"]) / len(records) if records else 0.0, "tasks": [{k: r[k] for k in ("ticker", "trade_date", "system", "status", "attempt")} for r in records]})
