"""Tests for the one-request DevAgent command-line interface."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, sentinel

from devagent.cli.app import main


class CliTest(unittest.TestCase):
    """Tests for CLI assembly without model or tool execution."""

    @patch("devagent.cli.app.run_agent", return_value="final answer")
    @patch("devagent.cli.app.build_tools", return_value=sentinel.registry)
    @patch("devagent.cli.app.RuntimeState", return_value=sentinel.state)
    def test_cli_builds_runtime_and_prints_answer(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
    ) -> None:
        workspace = Path("/tmp/devagent-cli-workspace")
        stdout = StringIO()

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ), redirect_stdout(stdout):
            status = main(
                [
                    "--workspace",
                    str(workspace),
                    "read the project file",
                ]
            )

        self.assertEqual(status, 0)
        mock_runtime_state.assert_called_once_with(workspace)
        mock_build_tools.assert_called_once_with(sentinel.state)
        mock_run_agent.assert_called_once_with(
            "read the project file",
            sentinel.registry,
        )
        self.assertEqual(stdout.getvalue(), "final answer\n")

    @patch("devagent.cli.app.run_agent")
    @patch("devagent.cli.app.build_tools")
    @patch("devagent.cli.app.RuntimeState")
    def test_missing_api_key_returns_clear_error(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
    ) -> None:
        stderr = StringIO()

        with patch.dict(os.environ, {}, clear=True), patch(
            "devagent.cli.app.load_dotenv"
        ), redirect_stderr(stderr):
            status = main(["hello"])

        self.assertEqual(status, 2)
        self.assertIn("DEEPSEEK_API_KEY is not set", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        mock_runtime_state.assert_not_called()
        mock_build_tools.assert_not_called()
        mock_run_agent.assert_not_called()

    @patch("devagent.cli.app.run_agent", return_value="loaded from dotenv")
    @patch("devagent.cli.app.build_tools", return_value=sentinel.registry)
    @patch("devagent.cli.app.RuntimeState", return_value=sentinel.state)
    def test_cli_loads_api_key_from_gitignored_dotenv(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
    ) -> None:
        stdout = StringIO()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_directory = Path(temporary_directory)
            (project_directory / ".env").write_text(
                'DEEPSEEK_API_KEY="dotenv-test-key"\n',
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True), patch(
                "devagent.cli.app.Path.cwd",
                return_value=project_directory,
            ), redirect_stdout(stdout):
                status = main(["hello"])
                loaded_key = os.environ.get("DEEPSEEK_API_KEY")

        self.assertEqual(status, 0)
        self.assertEqual(loaded_key, "dotenv-test-key")
        mock_runtime_state.assert_called_once_with(project_directory)
        mock_build_tools.assert_called_once_with(sentinel.state)
        mock_run_agent.assert_called_once_with("hello", sentinel.registry)
        self.assertEqual(stdout.getvalue(), "loaded from dotenv\n")

    @patch("devagent.__main__.cli_main", return_value=0)
    def test_module_entry_point_forwards_to_cli(self, mock_cli_main) -> None:
        from devagent.__main__ import main as module_main

        arguments = ["--workspace", "/tmp/workspace", "hello"]

        status = module_main(arguments)

        self.assertEqual(status, 0)
        mock_cli_main.assert_called_once_with(arguments)


if __name__ == "__main__":
    unittest.main()
