"""Minimal tool-calling agent runtime."""

from collections.abc import Callable
import json
from typing import Any

from devagent.providers.deepseek_provider import chat
from devagent.tools.registry import (
    InvalidToolArgumentsError,
    ToolRegistry,
    UnknownToolError,
)


ChatFunction = Callable[..., dict[str, Any]]


class MaxToolRoundsError(RuntimeError):
    """Raised when the model requests tools after the configured limit."""


def _invoke_tool(
    registry: ToolRegistry,
    name: str,
    raw_arguments: str | dict[str, Any],
) -> dict[str, Any]:
    """Invoke a tool and convert expected failures into safe results."""
    try:
        return registry.invoke(name, raw_arguments)
    except UnknownToolError as error:
        return {
            "ok": False,
            "error": {
                "type": "unknown_tool",
                "message": str(error),
            },
        }
    except InvalidToolArgumentsError as error:
        return {
            "ok": False,
            "error": {
                "type": "invalid_tool_arguments",
                "message": str(error),
            },
        }
    except Exception:
        return {
            "ok": False,
            "error": {
                "type": "tool_execution_error",
                "message": "tool execution failed",
            },
        }


def run_agent(
    user_message: str,
    registry: ToolRegistry,
    chat_fn: ChatFunction | None = None,
    max_tool_rounds: int = 4,
) -> str:
    """Run the minimal model-tool loop and return the final text response."""
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds must be non-negative")

    provider_chat = chat if chat_fn is None else chat_fn
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message}
    ]
    tool_rounds = 0

    while True:
        assistant_message = provider_chat(
            messages=messages,
            tools=registry.openai_schemas(),
        )
        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
            return assistant_message.get("content") or ""

        if tool_rounds >= max_tool_rounds:
            raise MaxToolRoundsError(
                f"model requested tools after {max_tool_rounds} tool rounds"
            )

        for tool_call in tool_calls:
            function = tool_call["function"]
            result = _invoke_tool(
                registry=registry,
                name=function["name"],
                raw_arguments=function["arguments"],
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        tool_rounds += 1
