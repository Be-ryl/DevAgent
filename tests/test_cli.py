"""Tests for the DevAgent command-line execution mode selection."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from devagent.cli.app import main
from devagent.execution import ExecutionMode, RunMetrics, RunResult


def successful_result(mode: ExecutionMode, answer: str) -> RunResult:
    return RunResult(
        final_answer=answer,
        metrics=RunMetrics(execution_mode=mode, completed=True),
    )


class CliTest(unittest.TestCase):
    """Tests for CLI assembly without model or tool execution."""

    @patch("devagent.cli.app.RunnerFactory")
    def test_cli_defaults_to_single_mode_and_prints_answer(
        self,
        mock_factory_class,
    ) -> None:
        workspace = Path("/tmp/devagent-cli-workspace")
        runner = mock_factory_class.return_value.create.return_value
        runner.run.return_value = successful_result(
            ExecutionMode.SINGLE,
            "final answer",
        )
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
        mock_factory_class.return_value.create.assert_called_once_with(
            ExecutionMode.SINGLE
        )
        config = runner.run.call_args.args[0]
        self.assertEqual(config.task, "read the project file")
        self.assertEqual(config.workspace, workspace)
        self.assertEqual(config.mode, ExecutionMode.SINGLE)
        self.assertEqual(stdout.getvalue(), "final answer\n")

    @patch("devagent.cli.app.RunnerFactory")
    def test_cli_mode_option_switches_runner(self, mock_factory_class) -> None:
        runner = mock_factory_class.return_value.create.return_value
        runner.run.return_value = successful_result(
            ExecutionMode.MULTI,
            "multi result",
        )

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ), redirect_stdout(StringIO()):
            status = main(["--mode", "multi", "implement task"])

        self.assertEqual(status, 0)
        mock_factory_class.return_value.create.assert_called_once_with(
            ExecutionMode.MULTI
        )
        config = runner.run.call_args.args[0]
        self.assertEqual(config.mode, ExecutionMode.MULTI)

    @patch("devagent.cli.app.RunnerFactory")
    def test_cli_forwards_approval_mode(self, mock_factory_class) -> None:
        runner = mock_factory_class.return_value.create.return_value
        runner.run.return_value = successful_result(
            ExecutionMode.SINGLE,
            "approval result",
        )

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ), redirect_stdout(StringIO()):
            status = main(
                ["--approval-mode", "deny", "run a safe coding task"]
            )

        self.assertEqual(status, 0)
        config = runner.run.call_args.args[0]
        self.assertEqual(config.approval_mode, "deny")
        self.assertIsNone(config.approval_callback)

    @patch(
        "devagent.cli.app.load_checkpoint_state",
        return_value={"execution_mode": "multi", "task": "resume task"},
    )
    @patch("devagent.cli.app.RunnerFactory")
    def test_cli_resume_infers_mode_from_checkpoint(
        self,
        mock_factory_class,
        mock_load_checkpoint,
    ) -> None:
        workspace = Path("/tmp/devagent-resume-workspace")
        runner = mock_factory_class.return_value.create.return_value
        runner.run.return_value = successful_result(
            ExecutionMode.MULTI,
            "resumed result",
        )

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ), redirect_stdout(StringIO()):
            status = main(["--resume", str(workspace)])

        self.assertEqual(status, 0)
        mock_load_checkpoint.assert_called_once_with(workspace)
        mock_factory_class.return_value.create.assert_called_once_with(
            ExecutionMode.MULTI
        )
        config = runner.run.call_args.args[0]
        self.assertTrue(config.resume)
        self.assertEqual(config.workspace, workspace)
        self.assertEqual(config.mode, ExecutionMode.MULTI)

    @patch("devagent.cli.app.RunnerFactory")
    def test_missing_api_key_returns_clear_error(
        self,
        mock_factory_class,
    ) -> None:
        stderr = StringIO()

        with patch.dict(os.environ, {}, clear=True), patch(
            "devagent.cli.app.load_dotenv"
        ), redirect_stderr(stderr):
            status = main(["hello"])

        self.assertEqual(status, 2)
        self.assertIn("DEEPSEEK_API_KEY is not set", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        mock_factory_class.assert_not_called()

    @patch("devagent.cli.app.RunnerFactory")
    def test_cli_loads_api_key_from_gitignored_dotenv(
        self,
        mock_factory_class,
    ) -> None:
        runner = mock_factory_class.return_value.create.return_value
        runner.run.return_value = successful_result(
            ExecutionMode.SINGLE,
            "loaded from dotenv",
        )
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
        config = runner.run.call_args.args[0]
        self.assertEqual(config.workspace, project_directory)
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
