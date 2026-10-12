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

    @patch(
        "devagent.cli.app.BM25RagService.from_directory",
        return_value=sentinel.rag_service,
    )
    @patch("devagent.cli.app.run_agent", return_value="knowledge answer")
    @patch("devagent.cli.app.build_tools", return_value=sentinel.registry)
    @patch("devagent.cli.app.RuntimeState", return_value=sentinel.state)
    def test_cli_builds_one_rag_service_when_knowledge_directory_is_given(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
        mock_from_directory,
    ) -> None:
        stdout = StringIO()

        with tempfile.TemporaryDirectory() as temporary_directory:
            knowledge_directory = Path(temporary_directory)
            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
            ), redirect_stdout(stdout):
                status = main(
                    [
                        "--knowledge-dir",
                        str(knowledge_directory),
                        "search policies",
                    ]
                )

        self.assertEqual(status, 0)
        mock_runtime_state.assert_called_once_with(Path.cwd())
        mock_from_directory.assert_called_once_with(knowledge_directory)
        mock_build_tools.assert_called_once_with(
            sentinel.state,
            sentinel.rag_service,
        )
        mock_run_agent.assert_called_once_with(
            "search policies",
            sentinel.registry,
        )
        self.assertEqual(stdout.getvalue(), "knowledge answer\n")

    @patch("devagent.cli.app.BM25RagService.from_directory")
    @patch("devagent.cli.app.run_agent")
    @patch("devagent.cli.app.build_tools")
    @patch("devagent.cli.app.RuntimeState")
    def test_cli_rejects_invalid_knowledge_directory(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
        mock_from_directory,
    ) -> None:
        stderr = StringIO()
        missing_directory = Path("/path/that/does/not/exist")

        with patch.dict(os.environ, {}, clear=True), patch(
            "devagent.cli.app.load_dotenv"
        ), redirect_stderr(stderr):
            status = main(
                [
                    "--knowledge-dir",
                    str(missing_directory),
                    "search policies",
                ]
            )

        self.assertEqual(status, 2)
        self.assertIn("knowledge directory is not a directory", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        mock_runtime_state.assert_not_called()
        mock_from_directory.assert_not_called()
        mock_build_tools.assert_not_called()
        mock_run_agent.assert_not_called()

    @patch(
        "devagent.rag.dense_index.DenseRagService.from_index",
        return_value=sentinel.dense_service,
    )
    @patch("devagent.cli.app.run_agent", return_value="dense answer")
    @patch("devagent.cli.app.build_tools", return_value=sentinel.registry)
    @patch("devagent.cli.app.RuntimeState", return_value=sentinel.state)
    def test_cli_loads_dense_index_and_builds_knowledge_tool(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
        mock_from_index,
    ) -> None:
        stdout = StringIO()
        index_directory = Path("/tmp/devagent-dense-index")

        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ), redirect_stdout(stdout):
            status = main(
                [
                    "--retrieval-mode",
                    "dense",
                    "--dense-index-dir",
                    str(index_directory),
                    "--embedding-model",
                    "fake-model",
                    "search policies",
                ]
            )

        self.assertEqual(status, 0)
        mock_runtime_state.assert_called_once_with(Path.cwd())
        mock_from_index.assert_called_once_with(
            index_directory,
            model_name="fake-model",
            knowledge_directory=None,
        )
        mock_build_tools.assert_called_once_with(
            sentinel.state,
            sentinel.dense_service,
        )
        mock_run_agent.assert_called_once_with(
            "search policies",
            sentinel.registry,
        )
        self.assertEqual(stdout.getvalue(), "dense answer\n")

    @patch("devagent.cli.app.run_agent")
    @patch("devagent.cli.app.build_tools", return_value=sentinel.registry)
    @patch("devagent.cli.app.RuntimeState", return_value=sentinel.state)
    def test_trace_flag_prints_runtime_events_to_stderr(
        self,
        mock_runtime_state,
        mock_build_tools,
        mock_run_agent,
    ) -> None:
        stdout = StringIO()
        stderr = StringIO()

        def fake_run_agent(user_message, registry, trace_fn):
            self.assertEqual(user_message, "search policies")
            self.assertIs(registry, sentinel.registry)
            trace_fn(
                '[tool_call] knowledge_search query="行李收费" top_k=3'
            )
            trace_fn(
                "[retrieval_hit] rank=1 source=baggage_policy.md "
                "chunk_id=baggage-001 score=3.250000 "
                'content_preview="超额行李费为每公斤 60 元。"'
            )
            trace_fn("[tool_result] returned_to_model=true")
            return "final answer"

        mock_run_agent.side_effect = fake_run_agent
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key-not-a-real-secret"},
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(["--trace", "search policies"])

        self.assertEqual(status, 0)
        mock_runtime_state.assert_called_once_with(Path.cwd())
        mock_build_tools.assert_called_once_with(sentinel.state)
        self.assertEqual(stdout.getvalue(), "final answer\n")
        self.assertEqual(
            stderr.getvalue().splitlines(),
            [
                '[tool_call] knowledge_search query="行李收费" top_k=3',
                "[retrieval_hit] rank=1 source=baggage_policy.md "
                "chunk_id=baggage-001 score=3.250000 "
                'content_preview="超额行李费为每公斤 60 元。"',
                "[tool_result] returned_to_model=true",
            ],
        )

    @patch("devagent.__main__.cli_main", return_value=0)
    def test_module_entry_point_forwards_to_cli(self, mock_cli_main) -> None:
        from devagent.__main__ import main as module_main

        arguments = ["--workspace", "/tmp/workspace", "hello"]

        status = module_main(arguments)

        self.assertEqual(status, 0)
        mock_cli_main.assert_called_once_with(arguments)


if __name__ == "__main__":
    unittest.main()
