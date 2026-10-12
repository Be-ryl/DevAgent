"""Tests for the minimal tool-calling agent runtime."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import BaseModel

from devagent.agent import MaxToolRoundsError, run_agent
from devagent.core.state import RuntimeState
from devagent.rag.service import BM25RagService
from devagent.tools.registry import ToolRegistry, build_tools


class EchoArgs(BaseModel):
    """Arguments for the local test tool."""

    text: str


def tool_call_message(
    *,
    call_id: str = "call_001",
    name: str = "echo",
    arguments: str = '{"text":"hello"}',
) -> dict:
    """Build one assistant tool-call message for fake providers."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


class FakeChat:
    """Return predefined messages and retain snapshots of each request."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    def __call__(self, *, messages: list[dict], tools: list[dict]) -> dict:
        self.requests.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
            }
        )
        return deepcopy(self.responses[len(self.requests) - 1])


class AgentRuntimeTest(unittest.TestCase):
    """Tests for direct answers, tool calls, errors, and loop limits."""

    def make_registry(self, handler) -> ToolRegistry:
        return ToolRegistry(
            {
                "echo": {
                    "name": "echo",
                    "description": "Echo text for a local runtime test.",
                    "args_model": EchoArgs,
                    "function": handler,
                }
            }
        )

    def test_model_can_answer_without_calling_a_tool(self) -> None:
        calls: list[str] = []

        def handler(text: str) -> dict:
            calls.append(text)
            return {"ok": True, "text": text}

        registry = self.make_registry(handler)
        fake_chat = FakeChat(
            [{"role": "assistant", "content": "direct answer"}]
        )

        answer = run_agent("hello", registry, chat_fn=fake_chat)

        self.assertEqual(answer, "direct answer")
        self.assertEqual(calls, [])
        self.assertEqual(
            fake_chat.requests[0]["messages"],
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(
            fake_chat.requests[0]["tools"],
            registry.openai_schemas(),
        )

    def test_successful_tool_call_is_saved_and_followed_by_result(self) -> None:
        calls: list[str] = []

        def handler(text: str) -> dict:
            calls.append(text)
            return {"ok": True, "text": text}

        registry = self.make_registry(handler)
        assistant_tool_call = tool_call_message(call_id="call_success")
        fake_chat = FakeChat(
            [
                assistant_tool_call,
                {"role": "assistant", "content": "final answer"},
            ]
        )

        answer = run_agent("use the tool", registry, chat_fn=fake_chat)

        self.assertEqual(answer, "final answer")
        self.assertEqual(calls, ["hello"])
        second_messages = fake_chat.requests[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_messages],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(second_messages[1], assistant_tool_call)
        self.assertEqual(second_messages[2]["tool_call_id"], "call_success")
        self.assertEqual(
            json.loads(second_messages[2]["content"]),
            {"ok": True, "text": "hello"},
        )

    def test_unknown_tool_becomes_a_tool_message(self) -> None:
        calls: list[str] = []

        def handler(text: str) -> dict:
            calls.append(text)
            return {"ok": True, "text": text}

        registry = self.make_registry(handler)
        fake_chat = FakeChat(
            [
                tool_call_message(name="missing"),
                {"role": "assistant", "content": "unknown tool explained"},
            ]
        )

        answer = run_agent("use a missing tool", registry, chat_fn=fake_chat)

        self.assertEqual(answer, "unknown tool explained")
        self.assertEqual(calls, [])
        result = json.loads(fake_chat.requests[1]["messages"][2]["content"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "unknown_tool")

    def test_invalid_arguments_do_not_execute_handler(self) -> None:
        invalid_arguments = ('{"text":', "{}")

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                calls: list[str] = []

                def handler(text: str) -> dict:
                    calls.append(text)
                    return {"ok": True, "text": text}

                registry = self.make_registry(handler)
                fake_chat = FakeChat(
                    [
                        tool_call_message(arguments=arguments),
                        {"role": "assistant", "content": "invalid arguments explained"},
                    ]
                )

                answer = run_agent("use bad arguments", registry, chat_fn=fake_chat)

                self.assertEqual(answer, "invalid arguments explained")
                self.assertEqual(calls, [])
                result = json.loads(
                    fake_chat.requests[1]["messages"][2]["content"]
                )
                self.assertFalse(result["ok"])
                self.assertEqual(
                    result["error"]["type"],
                    "invalid_tool_arguments",
                )

    def test_handler_exception_becomes_safe_tool_message(self) -> None:
        def handler(text: str) -> dict:
            raise RuntimeError("secret detail that must not be exposed")

        registry = self.make_registry(handler)
        fake_chat = FakeChat(
            [
                tool_call_message(),
                {"role": "assistant", "content": "tool failure explained"},
            ]
        )

        answer = run_agent("trigger failure", registry, chat_fn=fake_chat)

        self.assertEqual(answer, "tool failure explained")
        tool_content = fake_chat.requests[1]["messages"][2]["content"]
        result = json.loads(tool_content)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "tool_execution_error")
        self.assertNotIn("secret detail", tool_content)
        self.assertNotIn("Traceback", tool_content)

    def test_max_tool_rounds_allows_one_final_model_response_then_stops(self) -> None:
        calls: list[str] = []

        def handler(text: str) -> dict:
            calls.append(text)
            return {"ok": True, "text": text}

        registry = self.make_registry(handler)
        fake_chat = FakeChat(
            [
                tool_call_message(call_id="call_1"),
                tool_call_message(call_id="call_2"),
                tool_call_message(call_id="call_3"),
            ]
        )

        with self.assertRaises(MaxToolRoundsError):
            run_agent(
                "keep using tools",
                registry,
                chat_fn=fake_chat,
                max_tool_rounds=2,
            )

        self.assertEqual(calls, ["hello", "hello"])
        self.assertEqual(len(fake_chat.requests), 3)
        self.assertEqual(
            [message["role"] for message in fake_chat.requests[2]["messages"]],
            ["user", "assistant", "tool", "assistant", "tool"],
        )


class KnowledgeSearchAgentIntegrationTest(unittest.TestCase):
    """Tests the local BM25 evidence path through one agent tool loop."""

    def test_bm25_evidence_enters_tool_message_before_final_answer(self) -> None:
        policy_directory = (
            Path(__file__).resolve().parents[1] / "data" / "policies"
        )
        rag_service = BM25RagService.from_directory(policy_directory)
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = RuntimeState(Path(temporary_directory) / "workspace")
            registry = build_tools(state, rag_service)

            assistant_tool_call = tool_call_message(
                call_id="call_knowledge",
                name="knowledge_search",
                arguments=json.dumps(
                    {
                        "query": "托运行李超过20公斤每公斤收费60元",
                        "top_k": 1,
                    },
                    ensure_ascii=False,
                ),
            )
            fake_chat = FakeChat(
                [
                    assistant_tool_call,
                    {
                        "role": "assistant",
                        "content": "超额行李按每公斤 60 元收费。",
                    },
                ]
            )
            trace_events: list[str] = []

            answer = run_agent(
                "请查询托运行李超重费用",
                registry,
                chat_fn=fake_chat,
                trace_fn=trace_events.append,
            )

        self.assertEqual(answer, "超额行李按每公斤 60 元收费。")
        second_messages = fake_chat.requests[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_messages],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(second_messages[2]["tool_call_id"], "call_knowledge")
        tool_result = json.loads(second_messages[2]["content"])
        self.assertTrue(tool_result["ok"])
        self.assertEqual(tool_result["results"][0]["source"], "baggage_policy.md")
        self.assertIn("60", tool_result["results"][0]["content"])
        self.assertEqual(
            trace_events[0],
            "[tool_call] knowledge_search "
            'query="托运行李超过20公斤每公斤收费60元" top_k=1',
        )
        self.assertRegex(
            trace_events[1],
            r"^\[retrieval_hit\] rank=1 source=baggage_policy\.md "
            r"chunk_id=baggage_policy-[0-9a-f]{16} score=\d+\.\d{6} "
            r'content_preview=".*60.*"$',
        )
        self.assertEqual(
            trace_events[2],
            "[tool_result] returned_to_model=true",
        )


if __name__ == "__main__":
    unittest.main()
