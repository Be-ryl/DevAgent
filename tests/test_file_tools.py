"""Tests for the workspace-constrained file read tool."""

from pathlib import Path
import tempfile
import unittest

from devagent.core.state import RuntimeState
from devagent.tools.file_tools import edit_file, read_file, write_file


class ReadFileTest(unittest.TestCase):
    """Tests for reading text files from a runtime workspace."""

    def test_reads_requested_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            file_path = state.workspace / "notes.txt"
            file_path.write_text("zero\none\ntwo\nthree\n", encoding="utf-8")

            result = read_file(state, "notes.txt", offset=1, limit=2)

            self.assertEqual(
                result,
                {
                    "ok": True,
                    "path": "notes.txt",
                    "content": "one\ntwo\n",
                    "offset": 1,
                    "lines_returned": 2,
                },
            )

    def test_path_outside_workspace_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            result = read_file(state, "../secret.txt")

            self.assertFalse(result["ok"])
            self.assertIsNone(result["path"])
            self.assertEqual(result["error"]["code"], "invalid_path")

    def test_missing_file_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            result = read_file(state, "missing.txt")

            self.assertFalse(result["ok"])
            self.assertEqual(result["path"], "missing.txt")
            self.assertEqual(result["error"]["code"], "file_not_found")

    def test_directory_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            (state.workspace / "notes").mkdir()

            result = read_file(state, "notes")

            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], "not_a_file")


class WriteFileTest(unittest.TestCase):
    """Tests for creating UTF-8 files in a runtime workspace."""

    def test_creates_file_and_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            result = write_file(state, "notes/result.txt", "hello\n")

            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "notes/result.txt")
            self.assertEqual(
                (state.workspace / "notes" / "result.txt").read_text(
                    encoding="utf-8"
                ),
                "hello\n",
            )

    def test_path_outside_workspace_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            result = write_file(state, "../outside.txt", "not written")

            self.assertFalse(result["ok"])
            self.assertIsNone(result["path"])
            self.assertEqual(result["error"]["code"], "invalid_path")
            self.assertFalse(Path(temporary_directory, "outside.txt").exists())


class EditFileTest(unittest.TestCase):
    """Tests for exact text replacement in existing files."""

    def test_replaces_a_single_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            file_path = state.workspace / "message.txt"
            file_path.write_text("hello world\n", encoding="utf-8")

            result = edit_file(state, "message.txt", "world", "DevAgent")

            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], "message.txt")
            self.assertEqual(
                file_path.read_text(encoding="utf-8"),
                "hello DevAgent\n",
            )

    def test_zero_or_multiple_matches_return_error(self) -> None:
        cases = (("alpha\n", "missing", 0), ("one one\n", "one", 2))

        for content, old_text, expected_matches in cases:
            with self.subTest(expected_matches=expected_matches):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    state = RuntimeState(Path(temporary_directory) / "workspace")
                    file_path = state.workspace / "message.txt"
                    file_path.write_text(content, encoding="utf-8")

                    result = edit_file(
                        state,
                        "message.txt",
                        old_text,
                        "replacement",
                    )

                    self.assertFalse(result["ok"])
                    self.assertEqual(result["path"], "message.txt")
                    self.assertEqual(
                        result["error"]["code"],
                        "match_count_mismatch",
                    )
                    self.assertEqual(
                        result["error"]["matches"],
                        expected_matches,
                    )
                    self.assertEqual(
                        file_path.read_text(encoding="utf-8"),
                        content,
                    )


if __name__ == "__main__":
    unittest.main()
