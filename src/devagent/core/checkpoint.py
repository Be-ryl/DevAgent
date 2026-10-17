"""Lightweight JSON checkpoint and human-readable recovery support."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from devagent.core.state import RuntimeState
from devagent.execution import RunMetrics


CHECKPOINT_ROOT = Path(".devagent") / "checkpoints"
CHECKPOINT_FILE = "checkpoint.json"
RECOVERY_FILE = "RECOVERY.md"

CHECKPOINT_FIELDS = (
    "task",
    "execution_mode",
    "todos",
    "plan_summary",
    "acceptance_criteria",
    "verification_commands",
    "sources",
    "handoffs",
    "last_error",
    "verifier_feedback",
    "changed_files",
    "context_summary",
    "history_summary",
    "attempts",
    "max_attempts",
    "research_notes",
    "verified",
)


class CheckpointManager:
    """Persist only serializable recovery data inside the workspace."""

    def __init__(
        self,
        runtime: RuntimeState,
        metrics: RunMetrics,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.metrics = metrics
        self.event_sink = event_sink
        self.root = runtime.workspace / CHECKPOINT_ROOT
        self.checkpoint_path = self.root / CHECKPOINT_FILE
        self.recovery_path = self.root / RECOVERY_FILE

    def save(
        self,
        state: dict[str, Any],
        *,
        status: str,
        latest_node: str,
    ) -> dict[str, Any]:
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "latest_node": latest_node,
            "workspace": str(self.runtime.workspace),
            "state": {
                key: _json_safe(state.get(key))
                for key in CHECKPOINT_FIELDS
                if key in state
            },
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.recovery_path.write_text(
            _recovery_markdown(payload),
            encoding="utf-8",
        )
        self.metrics.checkpoints += 1
        event = {
            "status": status,
            "latest_node": latest_node,
            "checkpoint_file": str(self.checkpoint_path),
            "recovery_file": str(self.recovery_path),
        }
        if self.event_sink is not None:
            self.event_sink("checkpoint", event)
        return event

    def load(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.checkpoint_path}"
            )
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("checkpoint state is not a JSON object")
        return dict(state)


def load_checkpoint_state(workspace: Path | str) -> dict[str, Any]:
    """Read checkpoint state without constructing runtime-only objects."""
    checkpoint_path = Path(workspace) / CHECKPOINT_ROOT / CHECKPOINT_FILE
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state is not a JSON object")
    return dict(state)


def _recovery_markdown(payload: dict[str, Any]) -> str:
    state = payload.get("state", {})
    todos = state.get("todos", []) or []
    sources = state.get("sources", []) or []
    handoffs = state.get("handoffs", []) or []
    changed_files = state.get("changed_files", []) or []
    lines = [
        "# DevAgent Recovery",
        "",
        f"- status: {payload.get('status', '')}",
        f"- updated_at: {payload.get('updated_at', '')}",
        f"- latest_node: {payload.get('latest_node', '')}",
        f"- execution_mode: {state.get('execution_mode', '')}",
        f"- workspace: {payload.get('workspace', '')}",
        "",
        "## Task",
        "",
        str(state.get("task", "")),
        "",
        "## Plan",
        "",
        str(state.get("plan_summary", "")),
        "",
        "## Todos",
        "",
    ]
    lines.extend(
        f"- [{item.get('status', 'pending')}] {item.get('content', '')}"
        for item in todos
    )
    lines.extend(["", "## Acceptance Criteria", ""])
    lines.extend(f"- {item}" for item in state.get("acceptance_criteria", []) or [])
    lines.extend(["", "## Verification Commands", ""])
    lines.extend(
        f"- `{json.dumps(command, ensure_ascii=False)}`"
        for command in state.get("verification_commands", []) or []
    )
    lines.extend(["", "## Sources", ""])
    lines.extend(
        f"- {item.get('title', '')}: {item.get('url', '')}"
        for item in sources
    )
    lines.extend(["", "## Handoffs", ""])
    lines.extend(
        f"- {item.get('from_agent', '')} → {item.get('to_agent', '')}: {item.get('instruction', '')}"
        for item in handoffs
    )
    lines.extend(["", "## Changed Files", ""])
    lines.extend(f"- {item}" for item in changed_files)
    lines.extend(
        [
            "",
            "## Last Error",
            "",
            str(state.get("last_error") or state.get("verifier_feedback") or ""),
            "",
            "## Context Summary",
            "",
            str(state.get("context_summary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
