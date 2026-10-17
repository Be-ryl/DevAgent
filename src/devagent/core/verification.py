"""Execute only Planner-approved verification commands outside Verifier."""

from typing import Any, Callable

from devagent.core.approval import ApprovalPolicy
from devagent.core.state import RuntimeState
from devagent.execution import RunMetrics
from devagent.tools.bash_tool import run_command


class VerificationExecutor:
    """Run an immutable allowlist of Planner-generated commands."""

    def __init__(
        self,
        runtime: RuntimeState,
        planned_commands: list[list[str]],
        *,
        timeout_seconds: float,
        approval_policy: ApprovalPolicy,
        metrics: RunMetrics,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.planned_commands = [list(command) for command in planned_commands]
        self.timeout_seconds = timeout_seconds
        self.approval_policy = approval_policy
        self.metrics = metrics
        self.event_sink = event_sink

    def execute_planned(self) -> list[dict[str, Any]]:
        """Execute each planned command once in the declared order."""
        return [self.execute(command) for command in self.planned_commands]

    def execute(self, command: list[str]) -> dict[str, Any]:
        """Reject any command not present in the Planner allowlist."""
        if list(command) not in self.planned_commands:
            result = {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "changed_files": [],
                "error": {
                    "code": "unplanned_verification_command",
                    "message": "command is not in verification_commands",
                },
            }
            self._record(command, result)
            return result

        self.metrics.tool_calls += 1
        result = run_command(
            self.runtime,
            list(command),
            timeout=self.timeout_seconds,
            approval_policy=self.approval_policy,
            actor="VerificationExecutor",
        )
        self._record(command, result)
        return result

    def _record(self, command: list[str], result: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        self.event_sink(
            "tool_call",
            {"actor": "VerificationExecutor", "name": "run_command", "command": command},
        )
        self.event_sink(
            "tool_result",
            {"actor": "VerificationExecutor", "name": "run_command", "result": result},
        )
