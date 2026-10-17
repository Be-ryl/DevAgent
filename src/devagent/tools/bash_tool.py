"""Minimal subprocess command execution for a runtime workspace."""

import subprocess
from typing import Any

from devagent.core.approval import ApprovalPolicy
from devagent.core.state import RuntimeState
from devagent.core.workspace import changed_paths, snapshot_workspace


def _output_as_text(output: str | bytes | None) -> str:
    """Normalize captured subprocess output to text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def run_command(
    state: RuntimeState,
    command: list[str],
    timeout: float = 30.0,
    approval_policy: ApprovalPolicy | None = None,
    actor: str = "agent",
) -> dict[str, Any]:
    """Run a command directly inside the runtime workspace."""
    approval = (
        approval_policy.review(command, actor=actor)
        if approval_policy is not None
        else {"requires_approval": False, "approved": True}
    )
    if not approval.get("approved"):
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "changed_files": [],
            "approval": approval,
            "error": {
                "code": "approval_denied",
                "message": approval.get("reason") or "command was not approved",
            },
        }

    before = snapshot_workspace(state.workspace)
    try:
        completed = subprocess.run(
            command,
            cwd=state.workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        after = snapshot_workspace(state.workspace)
        return {
            "ok": False,
            "stdout": _output_as_text(error.stdout),
            "stderr": _output_as_text(error.stderr),
            "returncode": None,
            "changed_files": changed_paths(before, after),
            "approval": approval,
            "error": {
                "code": "timeout",
                "message": "command timed out",
                "timeout": timeout,
            },
        }
    except OSError:
        after = snapshot_workspace(state.workspace)
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "changed_files": changed_paths(before, after),
            "approval": approval,
            "error": {
                "code": "execution_error",
                "message": "unable to start command",
            },
        }

    after = snapshot_workspace(state.workspace)
    return {
        "ok": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
        "changed_files": changed_paths(before, after),
        "approval": approval,
    }

