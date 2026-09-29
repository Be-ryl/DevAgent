"""Minimal subprocess command execution for a runtime workspace."""

import subprocess
from typing import Any

from devagent.core.state import RuntimeState


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
) -> dict[str, Any]:
    """Run a command directly inside the runtime workspace."""
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
        return {
            "ok": False,
            "stdout": _output_as_text(error.stdout),
            "stderr": _output_as_text(error.stderr),
            "returncode": None,
            "error": {
                "code": "timeout",
                "message": "command timed out",
                "timeout": timeout,
            },
        }
    except OSError:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": {
                "code": "execution_error",
                "message": "unable to start command",
            },
        }

    return {
        "ok": True,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }

