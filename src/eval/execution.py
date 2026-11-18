"""Request-scoped diagnostics; never alters decision payloads."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ExecutionCapture:
    calls: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    decision_snapshot: dict[str, Any] | None = None

    @property
    def status(self):
        return "degraded" if self.issues else "success"

_current = ContextVar("experiment_capture", default=None)

@contextmanager
def capture_execution():
    capture = ExecutionCapture()
    token = _current.set(capture)
    try:
        yield capture
    finally:
        _current.reset(token)

def current_capture():
    return _current.get()

def note_issue(stage, error=None, *, reason=None):
    capture = _current.get()
    if capture is not None:
        capture.issues.append({"stage": stage, "error_type": type(error).__name__ if error is not None else None, "reason": reason})

def record_call(record):
    capture = _current.get()
    if capture is not None:
        capture.calls.append(record)

def record_snapshot(snapshot):
    from copy import deepcopy
    capture = _current.get()
    if capture is not None:
        capture.decision_snapshot = deepcopy(snapshot)


def token_usage(payload):
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {key: value for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                                   "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
            if type(value := usage.get(key)) is int and value >= 0}
