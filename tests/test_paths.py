"""Tests for runtime workspace path handling."""

import os
from pathlib import Path
import tempfile
import unittest

from devagent.core.paths import resolve_workspace_path
from devagent.core.state import RuntimeState


class RuntimeStateTest(unittest.TestCase):
    """Tests for workspace initialization."""

    def test_workspace_is_normalized_to_an_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            original_directory = Path.cwd()
            os.chdir(temporary_directory)
            try:
                state = RuntimeState(Path("parent") / ".." / "workspace")

                self.assertTrue(state.workspace.is_absolute())
                self.assertEqual(
                    state.workspace,
                    (Path(temporary_directory) / "workspace").resolve(),
                )
            finally:
                os.chdir(original_directory)

    def test_missing_workspace_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / "new" / "workspace"

            state = RuntimeState(workspace)

            self.assertTrue(state.workspace.is_dir())


class WorkspacePathTest(unittest.TestCase):
    """Tests for paths constrained to a workspace."""

    def test_nested_relative_path_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            result = state.resolve_path("notes/result.txt")

            self.assertEqual(result, state.workspace / "notes" / "result.txt")
            self.assertTrue(result.is_absolute())

    def test_internal_parent_component_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            result = state.resolve_path("subdir/../inside.txt")

            self.assertEqual(result, state.workspace / "inside.txt")

    def test_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            with self.assertRaises(ValueError):
                state.resolve_path("../secret.txt")

    def test_multi_level_parent_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")

            with self.assertRaises(ValueError):
                state.resolve_path("subdir/../../outside.txt")

    def test_absolute_path_outside_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            outside_path = Path(temporary_directory) / "outside.txt"

            with self.assertRaises(ValueError):
                state.resolve_path(outside_path.resolve())

    def test_absolute_path_inside_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            inside_path = state.workspace / "inside.txt"

            with self.assertRaises(ValueError):
                state.resolve_path(inside_path)

    def test_symlink_to_outside_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            outside = root / "outside"
            workspace.mkdir()
            outside.mkdir()
            link = workspace / "escape"

            try:
                link.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"symbolic links are not supported: {error}")

            with self.assertRaises(ValueError):
                resolve_workspace_path(workspace, "escape/secret.txt")


if __name__ == "__main__":
    unittest.main()
