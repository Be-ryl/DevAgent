"""Offline tests for context engineering and the execution harness."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from devagent.agents.verifier import Verifier
from devagent.core.approval import ApprovalDecision, ApprovalPolicy
from devagent.core.context import LayeredMemory, prepare_context
from devagent.core.state import RuntimeState
from devagent.core.verification import VerificationExecutor
from devagent.execution import ExecutionMode, RunConfig, RunMetrics, RunResult
from devagent.runners import RunnerFactory
from devagent.tools.bash_tool import run_command


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


class QueueModel:
    def __init__(self, provider: "QueueProvider", role: str) -> None:
        self.provider = provider
        self.role = role

    def bind_tools(self, tools):
        self.provider.bound_tools.setdefault(self.role, []).append(
            {item["function"]["name"] for item in tools}
        )
        return self

    def invoke(self, messages):
        self.provider.seen_messages.setdefault(self.role, []).append(list(messages))
        response = self.provider.responses[self.role].pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class QueueProvider:
    def __init__(self, **responses) -> None:
        self.responses = {role: list(items) for role, items in responses.items()}
        self.bound_tools: dict[str, list[set[str]]] = {}
        self.seen_messages: dict[str, list[list]] = {}

    def create_model(self, role: str, timeout_seconds: float):
        del timeout_seconds
        return QueueModel(self, role)


class ContextEngineeringTest(unittest.TestCase):
    def test_context_below_limit_is_not_compressed(self) -> None:
        messages = [HumanMessage(content="short task")]
        state = {"task": "short task", "todos": []}
        metrics = RunMetrics()
        config = RunConfig(task="short task", context_token_limit=1000)

        result = prepare_context(messages, state, config, metrics)

        self.assertIs(result, messages)
        self.assertEqual(metrics.context_compressions, 0)
        self.assertNotIn("context_summary", state)

    def test_context_over_limit_replaces_messages_and_preserves_state(self) -> None:
        messages = [
            HumanMessage(content="old request " * 100),
            AIMessage(content="old response " * 100),
        ]
        state = {
            "task": "critical task",
            "plan_summary": "critical plan",
            "todos": [{"id": "todo-1", "content": "finish", "status": "pending"}],
            "acceptance_criteria": ["tests pass"],
            "verification_commands": [["python", "-m", "unittest"]],
            "sources": [{"url": "https://example.com"}],
            "handoffs": [{"to_agent": "CodeAgent"}],
            "changed_files": ["app.py"],
            "last_error": "previous failure",
        }
        metrics = RunMetrics()
        config = RunConfig(task="critical task", context_token_limit=20)

        result = prepare_context(messages, state, config, metrics)

        self.assertEqual(len(result), 2)
        self.assertNotIn(messages[0], result)
        self.assertNotIn(messages[1], result)
        self.assertEqual(metrics.context_compressions, 1)
        summary = state["context_summary"]
        for expected in (
            "critical task",
            "finish",
            "tests pass",
            "unittest",
            "https://example.com",
            "app.py",
            "previous failure",
        ):
            self.assertIn(expected, summary)
        memory = LayeredMemory().build(state)
        self.assertEqual(set(memory), {"rules", "working_memory", "history_summary"})

    def test_single_and_multi_read_the_same_context_limit(self) -> None:
        with patch.dict(
            os.environ,
            {"DEVAGENT_CONTEXT_TOKEN_LIMIT": "321"},
        ):
            single = RunConfig(task="x", mode=ExecutionMode.SINGLE)
            multi = RunConfig(task="x", mode=ExecutionMode.MULTI)

        self.assertEqual(single.context_token_limit, 321)
        self.assertEqual(multi.context_token_limit, 321)


class ApprovalAndVerificationTest(unittest.TestCase):
    def test_dangerous_command_is_denied(self) -> None:
        metrics = RunMetrics()
        policy = ApprovalPolicy("deny", metrics=metrics)
        with tempfile.TemporaryDirectory() as directory:
            result = run_command(
                RuntimeState(directory),
                ["python", "-m", "pip", "install", "not-a-real-package"],
                approval_policy=policy,
                actor="SingleAgent",
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "approval_denied")
        self.assertEqual(metrics.approval_requests, 1)
        self.assertEqual(metrics.approval_denials, 1)

    def test_inline_mode_invokes_injected_callback(self) -> None:
        requests = []

        def fake_approval(request):
            requests.append(request)
            return ApprovalDecision(False, "test rejection")

        metrics = RunMetrics()
        policy = ApprovalPolicy(
            "inline",
            callback=fake_approval,
            metrics=metrics,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_command(
                RuntimeState(directory),
                ["curl", "https://example.com"],
                approval_policy=policy,
                actor="CodeAgent",
            )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].actor, "CodeAgent")
        self.assertFalse(result["approval"]["approved"])

    def test_verifier_cannot_execute_python_shell_or_other_commands(self) -> None:
        provider = QueueProvider(
            verifier=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "run_command",
                            {
                                "command": [
                                    sys.executable,
                                    "-c",
                                    "open('python.txt','w').write('bad')",
                                ]
                            },
                            "python-write",
                        ),
                        tool_call(
                            "run_command",
                            {"command": ["sh", "-c", "printf bad > shell.txt"]},
                            "shell-write",
                        ),
                        tool_call(
                            "run_command",
                            {"command": ["touch", "other.txt"]},
                            "other-write",
                        ),
                    ],
                ),
                AIMessage(
                    content='{"verified": false, "reason": "commands unavailable", "next_steps": []}'
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            Verifier(
                provider,
                RunConfig(task="verify", workspace=workspace),
                RunMetrics(),
            ).run(RuntimeState(workspace), "verify", [], [], [])

            self.assertFalse((workspace / "python.txt").exists())
            self.assertFalse((workspace / "shell.txt").exists())
            self.assertFalse((workspace / "other.txt").exists())

        self.assertEqual(
            provider.bound_tools["verifier"][0],
            {"read_file", "search_text"},
        )

    def test_verification_executor_runs_only_planned_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = RuntimeState(workspace)
            planned = [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('planned.txt').write_text('ok')",
            ]
            unplanned = [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('unplanned.txt').write_text('bad')",
            ]
            executor = VerificationExecutor(
                runtime,
                [planned],
                timeout_seconds=5,
                approval_policy=ApprovalPolicy("auto"),
                metrics=RunMetrics(),
            )

            planned_result = executor.execute_planned()[0]
            rejected_result = executor.execute(unplanned)

            self.assertTrue(planned_result["ok"])
            self.assertIn("planned.txt", planned_result["changed_files"])
            self.assertEqual(
                rejected_result["error"]["code"],
                "unplanned_verification_command",
            )
            self.assertTrue((workspace / "planned.txt").exists())
            self.assertFalse((workspace / "unplanned.txt").exists())

    def test_bash_snapshot_records_created_and_modified_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "existing.txt").write_text("before", encoding="utf-8")
            script = (
                "from pathlib import Path; "
                "Path('existing.txt').write_text('after'); "
                "Path('created.txt').write_text('new')"
            )

            result = run_command(
                RuntimeState(workspace),
                [sys.executable, "-c", script],
            )

        self.assertEqual(
            result["changed_files"],
            ["created.txt", "existing.txt"],
        )


class CheckpointTraceAndRunnerTest(unittest.TestCase):
    def test_interrupted_single_run_can_resume_from_checkpoint(self) -> None:
        first_provider = QueueProvider(
            single=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "write_file",
                            {"file_path": "partial.txt", "content": "saved"},
                            "write-partial",
                        )
                    ],
                ),
                KeyboardInterrupt(),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first = RunnerFactory(first_provider).create("single").run(
                RunConfig(task="resume task", workspace=workspace)
            )
            checkpoint_path = workspace / ".devagent" / "checkpoints" / "checkpoint.json"
            recovery_path = workspace / ".devagent" / "checkpoints" / "RECOVERY.md"

            self.assertEqual(first.metrics.error, "interrupted")
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(recovery_path.exists())
            checkpoint_payload = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            self.assertNotIn("runtime", checkpoint_payload["state"])
            self.assertNotIn("messages", checkpoint_payload["state"])

            second_provider = QueueProvider(
                single=[AIMessage(content="resumed successfully")]
            )
            resumed = RunnerFactory(second_provider).create("single").run(
                RunConfig(task="", workspace=workspace, resume=True)
            )

            self.assertEqual(resumed.final_answer, "resumed successfully")
            self.assertEqual(resumed.state["task"], "resume task")
            self.assertIn("partial.txt", resumed.changed_files)
            self.assertIn("resume task", recovery_path.read_text(encoding="utf-8"))

    def test_single_runner_counts_files_changed_by_command(self) -> None:
        provider = QueueProvider(
            single=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "run_command",
                            {
                                "command": [
                                    sys.executable,
                                    "-c",
                                    (
                                        "from pathlib import Path; "
                                        "Path('command-created.txt').write_text('ok')"
                                    ),
                                ]
                            },
                            "command-write",
                        )
                    ],
                ),
                AIMessage(content="created by command"),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = RunnerFactory(provider).create("single").run(
                RunConfig(task="create with a command", workspace=directory)
            )

        self.assertEqual(result.changed_files, ["command-created.txt"])
        self.assertEqual(result.metrics.changed_file_count, 1)

    def test_trace_files_are_complete_and_parseable(self) -> None:
        provider = QueueProvider(
            planner=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "SetPlanTool",
                            {
                                "plan_summary": "exercise harness",
                                "todos": ["attempt command"],
                                "acceptance_criteria": ["command is controlled"],
                                "verification_commands": [],
                            },
                            "plan",
                        ),
                        tool_call(
                            "CallCodeAgentTool",
                            {"instruction": "Attempt the controlled command."},
                            "delegate",
                        ),
                    ],
                ),
                AIMessage(content="delegated"),
            ],
            code=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "run_command",
                            {"command": ["curl", "https://example.com"]},
                            "network",
                        )
                    ],
                ),
                AIMessage(content="command was controlled"),
            ],
            verifier=[
                AIMessage(
                    content='{"verified": false, "reason": "denied as expected", "next_steps": []}'
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            result = RunnerFactory(provider).create("multi").run(
                RunConfig(
                    task="trace harness",
                    workspace=directory,
                    mode=ExecutionMode.MULTI,
                    max_attempts=1,
                    context_token_limit=40,
                    approval_mode="deny",
                )
            )
            trace_root = (
                Path(directory)
                / ".devagent"
                / "traces"
                / result.metrics.trace_id
            )
            events = [
                json.loads(line)
                for line in (trace_root / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            summary = json.loads(
                (trace_root / "summary.json").read_text(encoding="utf-8")
            )
            timeline = (trace_root / "timeline.md").read_text(encoding="utf-8")

        event_types = {event["type"] for event in events}
        self.assertTrue(
            {
                "run_start",
                "model_call",
                "tool_call",
                "tool_result",
                "handoff",
                "approval",
                "context_compression",
                "verifier_result",
                "checkpoint",
                "run_end",
            }
            <= event_types
        )
        self.assertEqual(summary["trace_id"], result.metrics.trace_id)
        self.assertEqual(summary["execution_mode"], "multi")
        self.assertGreater(result.metrics.context_compressions, 0)
        self.assertEqual(result.metrics.approval_requests, 1)
        self.assertEqual(result.metrics.approval_denials, 1)
        self.assertIn("DevAgent Trace Timeline", timeline)

    def test_single_and_multi_still_return_the_same_result_type(self) -> None:
        provider = QueueProvider(
            single=[AIMessage(content="single")],
            planner=[AIMessage(content="planned")],
            verifier=[
                AIMessage(
                    content='{"verified": true, "reason": "ok", "next_steps": []}'
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            factory = RunnerFactory(provider)
            single = factory.create("single").run(
                RunConfig(task="same", workspace=directory)
            )
            multi = factory.create("multi").run(
                RunConfig(
                    task="same",
                    workspace=directory,
                    mode=ExecutionMode.MULTI,
                )
            )

        self.assertIsInstance(single, RunResult)
        self.assertIsInstance(multi, RunResult)
        self.assertEqual(single.metrics.execution_mode, ExecutionMode.SINGLE)
        self.assertEqual(multi.metrics.execution_mode, ExecutionMode.MULTI)


if __name__ == "__main__":
    unittest.main()
