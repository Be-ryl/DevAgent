"""Tests for minimal workspace command execution."""

from pathlib import Path
import sys
import tempfile
import time
import unittest

from devagent.core.state import RuntimeState
from devagent.tools.bash_tool import run_command


class RunCommandTest(unittest.TestCase):
    """Tests for commands executed without a shell."""

    def test_command_runs_in_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            script = (
                "from pathlib import Path; "
                "Path('marker.txt').write_text('created', encoding='utf-8'); "
                "print(Path.cwd())"
            )

            result = run_command(
                state,
                [sys.executable, "-c", script],
                timeout=2.0,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["stdout"].strip(), str(state.workspace))
            self.assertEqual(result["stderr"], "")
            self.assertEqual(
                (state.workspace / "marker.txt").read_text(encoding="utf-8"),
                "created",
            )

    def test_timeout_terminates_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            started_at = time.monotonic()

            result = run_command(
                state,
                [sys.executable, "-c", "import time; time.sleep(5)"],
                timeout=0.1,
            )

            elapsed = time.monotonic() - started_at
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], "timeout")
            self.assertIsNone(result["returncode"])
            self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
