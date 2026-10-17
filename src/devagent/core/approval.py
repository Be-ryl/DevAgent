"""Injectable command approval policy shared by all command executors."""

from dataclasses import dataclass
import re
import shlex
from typing import Any, Callable
from uuid import uuid4

from devagent.execution import RunMetrics


@dataclass(frozen=True)
class ApprovalRequest:
    """A high-risk command awaiting a policy decision."""

    id: str
    command: list[str]
    display_command: str
    risk_reason: str
    actor: str


@dataclass(frozen=True)
class ApprovalDecision:
    """Normalized approval callback response."""

    approved: bool
    reason: str = ""


ApprovalCallback = Callable[[ApprovalRequest], bool | ApprovalDecision]


RISK_PATTERNS = (
    (r"\b(?:pip|pip3)\s+install(?:\s|$)", "dependency installation"),
    (r"\bpython(?:3)?\s+-m\s+pip\s+install(?:\s|$)", "dependency installation"),
    (r"\b(?:npm|pnpm)\s+(?:install|add)(?:\s|$)", "dependency installation"),
    (r"\byarn\s+(?:install|add)(?:\s|$)", "dependency installation"),
    (r"\b(?:curl|wget)\b", "network access"),
    (r"\bgit\s+(?:clone|fetch|pull|push)\b", "network access"),
    (r"\b(?:uvicorn|gunicorn)\b", "background service"),
    (r"python(?:3)?\s+-m\s+http\.server\b", "background service"),
    (r"\b(?:rm\s+-rf|shutdown|reboot|mkfs|format)\b", "destructive command"),
    (r"git\s+(?:reset\s+--hard|clean\s+-[a-z]*f)", "destructive command"),
)


class ApprovalPolicy:
    """Classify high-risk commands and resolve them by configured mode."""

    def __init__(
        self,
        mode: str,
        callback: ApprovalCallback | None = None,
        metrics: RunMetrics | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if mode not in {"inline", "deny", "auto"}:
            raise ValueError("invalid approval mode")
        self.mode = mode
        self.callback = callback
        self.metrics = metrics
        self.event_sink = event_sink

    def review(self, command: list[str], *, actor: str) -> dict[str, Any]:
        """Return approval metadata; safe commands bypass approval."""
        display = shlex.join(command)
        risk_reason = classify_command_risk(display)
        if risk_reason is None:
            return {"requires_approval": False, "approved": True}

        request = ApprovalRequest(
            id=f"approval-{uuid4().hex[:8]}",
            command=list(command),
            display_command=display,
            risk_reason=risk_reason,
            actor=actor,
        )
        if self.metrics is not None:
            self.metrics.approval_requests += 1

        decision = self._decide(request)
        if not decision.approved and self.metrics is not None:
            self.metrics.approval_denials += 1
        payload = {
            "approval_id": request.id,
            "actor": actor,
            "command": display,
            "risk_reason": risk_reason,
            "mode": self.mode,
            "approved": decision.approved,
            "reason": decision.reason,
        }
        if self.event_sink is not None:
            self.event_sink("approval", payload)
        return {"requires_approval": True, **payload}

    def _decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if self.mode == "auto":
            return ApprovalDecision(True, "automatically approved")
        if self.mode == "deny":
            return ApprovalDecision(False, "high-risk commands are denied")
        if self.callback is None:
            return ApprovalDecision(False, "inline approval callback is unavailable")
        raw_decision = self.callback(request)
        if isinstance(raw_decision, ApprovalDecision):
            return raw_decision
        return ApprovalDecision(bool(raw_decision))


def classify_command_risk(command: str) -> str | None:
    """Return a human-readable high-risk category when one matches."""
    normalized = command.strip()
    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return reason
    return None
