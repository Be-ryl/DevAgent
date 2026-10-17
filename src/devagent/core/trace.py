"""Uniform, behavior-neutral tracing for SingleAgent and MultiAgent runs."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from devagent.core.state import RuntimeState
from devagent.execution import RunMetrics


TRACE_ROOT = Path(".devagent") / "traces"


class TraceRecorder:
    """Write JSONL events, a JSON summary, and a Markdown timeline."""

    def __init__(
        self,
        runtime: RuntimeState,
        metrics: RunMetrics,
        task: str,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.trace_id = f"run-{stamp}-{uuid4().hex[:8]}"
        self.runtime = runtime
        self.metrics = metrics
        self.task = task
        self.root = runtime.workspace / TRACE_ROOT / self.trace_id
        self.events_path = self.root / "events.jsonl"
        self.summary_path = self.root / "summary.json"
        self.timeline_path = self.root / "timeline.md"
        self.started = perf_counter()
        self.sequence = 0
        self.timeline: list[str] = []
        self.errors: list[str] = []
        self.enabled = True
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.enabled = False
            self.errors.append(f"{type(error).__name__}: {error}")
        self.metrics.trace_id = self.trace_id

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        """Best-effort event recording that never changes agent behavior."""
        if not self.enabled:
            return
        try:
            self.sequence += 1
            event = {
                "sequence": self.sequence,
                "timestamp": _utc_now(),
                "elapsed_seconds": round(perf_counter() - self.started, 6),
                "type": event_type,
                "payload": _compact(payload),
            }
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.timeline.append(
                f"- {event['elapsed_seconds']:.6f}s `{event_type}`"
            )
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")

    def start(self, *, execution_mode: str, resumed: bool) -> None:
        self.record(
            "run_start",
            {
                "task": self.task,
                "execution_mode": execution_mode,
                "workspace": str(self.runtime.workspace),
                "resumed": resumed,
            },
        )

    def finish(self, *, status: str, state: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.record(
            "run_end",
            {
                "status": status,
                "verified": state.get("verified"),
                "attempts": state.get("attempts"),
                "error": self.metrics.error,
            },
        )
        summary = {
            "trace_id": self.trace_id,
            "status": status,
            "task": self.task,
            "workspace": str(self.runtime.workspace),
            "execution_mode": self.metrics.execution_mode.value,
            "event_count": self.sequence,
            "metrics": _json_safe(asdict(self.metrics)),
            "errors": list(self.errors),
            "events_file": str(self.events_path),
            "timeline_file": str(self.timeline_path),
        }
        try:
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.timeline_path.write_text(
                "\n".join(
                    [
                        "# DevAgent Trace Timeline",
                        "",
                        f"- trace_id: {self.trace_id}",
                        f"- status: {status}",
                        f"- execution_mode: {self.metrics.execution_mode.value}",
                        "",
                        "## Events",
                        "",
                        *(self.timeline or ["- (none)"]),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")


def _compact(value: Any, limit: int = 1600) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, str):
        return safe if len(safe) <= limit else safe[: limit - 3] + "..."
    if isinstance(safe, dict):
        return {key: _compact(item, limit) for key, item in safe.items()}
    if isinstance(safe, list):
        return [_compact(item, limit) for item in safe[:100]]
    return safe


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
