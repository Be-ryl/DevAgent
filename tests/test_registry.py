"""Tests for the runtime-bound tool registry."""

from pathlib import Path
import tempfile
import unittest

from devagent.core.state import RuntimeState
from devagent.tools.registry import build_tools


class ToolRegistryTest(unittest.TestCase):
    """Tests for registered tool lookup and invocation."""

    def test_registered_write_file_writes_without_explicit_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            tools = build_tools(state)

            write_tool = tools["write_file"]
            result = write_tool["function"]("notes/result.txt", "created\n")

            self.assertEqual(
                set(tools),
                {
                    "read_file",
                    "write_file",
                    "edit_file",
                    "search_text",
                    "run_command",
                },
            )
            self.assertEqual(write_tool["name"], "write_file")
            self.assertTrue(write_tool["description"])
            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "notes/result.txt")
            self.assertEqual(
                (state.workspace / "notes" / "result.txt").read_text(
                    encoding="utf-8"
                ),
                "created\n",
            )


if __name__ == "__main__":
    unittest.main()
