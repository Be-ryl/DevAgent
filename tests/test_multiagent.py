"""Offline tests for the stage-two single/multi architecture."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from devagent.agents.code_agent import CodeAgent
from devagent.agents.search_agent import SearchAgent
from devagent.agents.verifier import Verifier
from devagent.core.state import RuntimeState
from devagent.execution import ExecutionMode, RunConfig, RunMetrics, RunResult
from devagent.runners import RunnerFactory


def tool_call(name: str, args: dict, call_id: str) -> dict:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def plan_call(*delegations: dict) -> AIMessage:
    calls = [
        tool_call(
            "SetPlanTool",
            {
                "plan_summary": "Create and verify the requested file.",
                "todos": ["Create result.txt"],
                "acceptance_criteria": ["result.txt contains done"],
                "verification_commands": [
                    [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            "assert Path('result.txt').read_text() == 'done'"
                        ),
                    ]
                ],
            },
            "plan",
        )
    ]
    calls.extend(delegations)
    return AIMessage(content="", tool_calls=calls)


def code_responses(content: str = "done") -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "TodoUpdateTool",
                    {"todo_id": "todo-1", "status": "in_progress"},
                    "todo-start",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "write_file",
                    {"file_path": "result.txt", "content": content},
                    "write",
                )
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "TodoUpdateTool",
                    {"todo_id": "todo-1", "status": "completed"},
                    "todo-done",
                )
            ],
        ),
        AIMessage(content="Implemented result.txt and ran checks."),
    ]


class FakeModel:
    def __init__(self, provider: "FakeProvider", role: str) -> None:
        self.provider = provider
        self.role = role

    def bind_tools(self, tools):
        names = {
            item["function"]["name"]
            for item in tools
        }
        self.provider.bound_tools.setdefault(self.role, []).append(names)
        return self

    def invoke(self, messages):
        del messages
        responses = self.provider.responses[self.role]
        if not responses:
            raise AssertionError(f"no fake response remains for role {self.role}")
        return responses.pop(0)


class FakeProvider:
    def __init__(self, **responses: list[AIMessage]) -> None:
        self.responses = {role: list(items) for role, items in responses.items()}
        self.created_roles: list[str] = []
        self.bound_tools: dict[str, list[set[str]]] = {}

    def create_model(self, role: str, timeout_seconds: float):
        self.created_roles.append(role)
        self.last_timeout = timeout_seconds
        return FakeModel(self, role)


class MultiAgentWorkflowTest(unittest.TestCase):
    def test_planner_can_call_only_code_agent_for_simple_task(self) -> None:
        provider = FakeProvider(
            planner=[
                plan_call(
                    tool_call(
                        "CallCodeAgentTool",
                        {"instruction": "Create result.txt."},
                        "delegate-code",
                    )
                ),
                AIMessage(content="Implementation delegated."),
            ],
            code=code_responses(),
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
                                    (
                                        "from pathlib import Path; "
                                        "assert Path('result.txt').read_text() == 'done'"
                                    ),
                                ]
                            },
                            "verify-command",
                        )
                    ],
                ),
                AIMessage(
                    content=json.dumps(
                        {"verified": True, "reason": "all checks pass", "next_steps": []}
                    )
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                task="Create result.txt",
                workspace=directory,
                mode=ExecutionMode.MULTI,
            )
            result = RunnerFactory(provider).create(config.mode).run(config)

            self.assertTrue(result.verified)
            self.assertEqual(Path(directory, "result.txt").read_text(), "done")
            self.assertIn("Tests: passed", result.final_answer)
            self.assertNotIn("search", provider.created_roles)
            self.assertEqual(result.state["handoffs"][0]["to_agent"], "CodeAgent")
            self.assertEqual(
                provider.bound_tools["planner"][0],
                {"SetPlanTool", "CallSearchAgentTool", "CallCodeAgentTool"},
            )

    @patch(
        "devagent.tools.registry.web_search",
        return_value={
            "ok": True,
            "query": "current docs",
            "answer": "Use the documented API.",
            "results": [
                {
                    "title": "Official docs",
                    "url": "https://example.com/docs",
                    "content": "API details",
                    "score": 1.0,
                }
            ],
        },
    )
    def test_planner_can_search_before_calling_code_agent(self, mock_search) -> None:
        provider = FakeProvider(
            planner=[
                plan_call(
                    tool_call(
                        "CallSearchAgentTool",
                        {"instruction": "Research the current API."},
                        "delegate-search",
                    ),
                    tool_call(
                        "CallCodeAgentTool",
                        {"instruction": "Use the research and create result.txt."},
                        "delegate-code",
                    ),
                ),
                AIMessage(content="Research and implementation complete."),
            ],
            search=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "WebSearchTool",
                            {"query": "current docs"},
                            "web-search",
                        )
                    ],
                ),
                AIMessage(content="Research summary with source URL."),
            ],
            code=code_responses(),
            verifier=[
                AIMessage(
                    content='{"verified": true, "reason": "passed", "next_steps": []}'
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                task="Research then create a file",
                workspace=directory,
                mode=ExecutionMode.MULTI,
            )
            result = RunnerFactory(provider).create(config.mode).run(config)

        self.assertTrue(result.verified)
        self.assertEqual(
            [handoff["to_agent"] for handoff in result.state["handoffs"]],
            ["SearchAgent", "CodeAgent"],
        )
        self.assertEqual(result.state["sources"][0]["url"], "https://example.com/docs")
        self.assertEqual(provider.bound_tools["search"][0], {"WebSearchTool"})
        mock_search.assert_called_once()

    def test_verifier_failure_routes_back_to_planner_then_passes(self) -> None:
        provider = FakeProvider(
            planner=[
                plan_call(
                    tool_call(
                        "CallCodeAgentTool",
                        {"instruction": "Create initial result."},
                        "code-one",
                    )
                ),
                AIMessage(content="First pass done."),
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "CallCodeAgentTool",
                            {"instruction": "Fix the verifier failure."},
                            "code-two",
                        )
                    ],
                ),
                AIMessage(content="Fix delegated."),
            ],
            code=[*code_responses("wrong"), *code_responses("done")],
            verifier=[
                AIMessage(
                    content=(
                        '{"verified": false, "reason": "wrong content", '
                        '"next_steps": ["write done"]}'
                    )
                ),
                AIMessage(
                    content='{"verified": true, "reason": "fixed", "next_steps": []}'
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                task="Create correct result",
                workspace=directory,
                mode=ExecutionMode.MULTI,
                max_attempts=3,
            )
            result = RunnerFactory(provider).create(config.mode).run(config)

        self.assertTrue(result.verified)
        self.assertEqual(result.state["attempts"], 2)
        self.assertEqual(result.metrics.retry_count, 1)
        self.assertEqual(provider.created_roles.count("planner"), 2)

    def test_max_attempts_stops_failed_workflow(self) -> None:
        failure = AIMessage(
            content='{"verified": false, "reason": "still failing", "next_steps": ["fix it"]}'
        )
        provider = FakeProvider(
            planner=[
                AIMessage(content="No implementation available."),
                AIMessage(content="Retry could not fix it."),
            ],
            verifier=[failure, failure],
        )
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                task="Impossible fake task",
                workspace=directory,
                mode=ExecutionMode.MULTI,
                max_attempts=2,
            )
            result = RunnerFactory(provider).create(config.mode).run(config)

        self.assertFalse(result.verified)
        self.assertTrue(result.metrics.completed)
        self.assertEqual(result.state["attempts"], 2)
        self.assertIn("FAILED", result.final_answer)
        self.assertIn("still failing", result.final_answer)
        self.assertIn("fix it", result.final_answer)


class RoleIsolationTest(unittest.TestCase):
    def test_search_agent_cannot_write_files(self) -> None:
        provider = FakeProvider(
            search=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "write_file",
                            {"file_path": "forbidden.txt", "content": "bad"},
                            "forbidden-write",
                        )
                    ],
                ),
                AIMessage(content="Write was unavailable."),
            ]
        )
        metrics = RunMetrics(execution_mode=ExecutionMode.MULTI)
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(task="research", workspace=directory)
            SearchAgent(provider, config, metrics).run("research", "find facts")
            self.assertFalse(Path(directory, "forbidden.txt").exists())

        self.assertEqual(provider.bound_tools["search"][0], {"WebSearchTool"})

    def test_verifier_cannot_modify_files(self) -> None:
        provider = FakeProvider(
            verifier=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "write_file",
                            {"file_path": "forbidden.txt", "content": "bad"},
                            "forbidden-write",
                        )
                    ],
                ),
                AIMessage(
                    content='{"verified": false, "reason": "cannot write", "next_steps": []}'
                ),
            ]
        )
        metrics = RunMetrics(execution_mode=ExecutionMode.MULTI)
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(task="verify", workspace=directory)
            Verifier(provider, config, metrics).run(
                RuntimeState(directory),
                "verify",
                [],
                [],
                [],
            )
            self.assertFalse(Path(directory, "forbidden.txt").exists())

        self.assertEqual(
            provider.bound_tools["verifier"][0],
            {"read_file", "search_text"},
        )

    def test_code_agent_cannot_escape_workspace(self) -> None:
        provider = FakeProvider(
            code=[
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            "write_file",
                            {"file_path": "../outside.txt", "content": "bad"},
                            "escape",
                        )
                    ],
                ),
                AIMessage(content="Escape rejected."),
            ]
        )
        metrics = RunMetrics(execution_mode=ExecutionMode.MULTI)
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory, "workspace")
            config = RunConfig(task="escape", workspace=workspace)
            result = CodeAgent(provider, config, metrics).run(
                RuntimeState(workspace),
                "escape",
                "write outside",
                [],
            )
            self.assertFalse(Path(directory, "outside.txt").exists())
            tool_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
            ]
            self.assertIn("invalid_path", tool_messages[0].content)


class ComparableRunnerTest(unittest.TestCase):
    def test_same_task_runs_in_both_modes_with_unified_results(self) -> None:
        provider = FakeProvider(
            single=[
                AIMessage(
                    content="single complete",
                    usage_metadata={
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "total_tokens": 7,
                    },
                )
            ],
            planner=[AIMessage(content="Planning complete.")],
            verifier=[
                AIMessage(
                    content='{"verified": true, "reason": "complete", "next_steps": []}'
                )
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            single_config = RunConfig(
                task="Explain the task",
                workspace=directory,
                mode=ExecutionMode.SINGLE,
            )
            multi_config = RunConfig(
                task="Explain the task",
                workspace=directory,
                mode=ExecutionMode.MULTI,
            )
            factory = RunnerFactory(provider)
            single = factory.create("single").run(single_config)
            self.assertEqual(provider.created_roles, ["single"])
            multi = factory.create("multi").run(multi_config)

        self.assertIsInstance(single, RunResult)
        self.assertIsInstance(multi, RunResult)
        self.assertEqual(provider.created_roles, ["single", "planner", "verifier"])
        self.assertIn("WebSearchTool", provider.bound_tools["single"][0])
        self.assertEqual(single.metrics.execution_mode, ExecutionMode.SINGLE)
        self.assertEqual(multi.metrics.execution_mode, ExecutionMode.MULTI)
        self.assertEqual(single.metrics.model_calls, 1)
        self.assertEqual(single.metrics.input_tokens, 5)
        self.assertEqual(multi.metrics.model_calls, 2)
        self.assertNotEqual(id(single.metrics), id(multi.metrics))


if __name__ == "__main__":
    unittest.main()
