"""Shared message, metric, and tool-call helpers for agent loops."""

import json
import re
from typing import Any, Callable

from langchain_core.messages import ToolMessage

from devagent.execution import RunMetrics
from devagent.tools.registry import (
    InvalidToolArgumentsError,
    ToolRegistry,
    UnknownToolError,
)


def bind_model_tools(model: Any, registry: ToolRegistry) -> Any:
    """Bind OpenAI-compatible schemas without exposing Python handlers."""
    return model.bind_tools(registry.openai_schemas())


def invoke_model(
    model: Any,
    messages: list[Any],
    metrics: RunMetrics,
    *,
    actor: str = "agent",
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
) -> Any:
    """Invoke a model and account for calls and reported token usage."""
    response = model.invoke(messages)
    metrics.model_calls += 1
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        metrics.input_tokens += int(usage.get("input_tokens", 0) or 0)
        metrics.output_tokens += int(usage.get("output_tokens", 0) or 0)
        if event_sink is not None:
            event_sink(
                "model_call",
                {
                    "actor": actor,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            )
        return response

    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage", {}) or {}
    metrics.input_tokens += int(token_usage.get("prompt_tokens", 0) or 0)
    metrics.output_tokens += int(token_usage.get("completion_tokens", 0) or 0)
    if event_sink is not None:
        event_sink(
            "model_call",
            {
                "actor": actor,
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
            },
        )
    return response


def message_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Return normalized tool calls from a LangChain-compatible message."""
    return list(getattr(message, "tool_calls", None) or [])


def invoke_tool_call(
    registry: ToolRegistry,
    call: dict[str, Any],
    metrics: RunMetrics,
    *,
    actor: str = "agent",
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[ToolMessage, dict[str, Any]]:
    """Execute one whitelisted tool call and return a safe tool message."""
    name = str(call.get("name") or "")
    arguments = call.get("args", {})
    metrics.tool_calls += 1
    if event_sink is not None:
        event_sink(
            "tool_call",
            {"actor": actor, "name": name, "arguments": arguments},
        )
    try:
        result = registry.invoke(name, arguments)
    except UnknownToolError as error:
        result = {"ok": False, "error": str(error)}
    except InvalidToolArgumentsError as error:
        result = {"ok": False, "error": str(error)}
    except Exception:
        result = {"ok": False, "error": "tool execution failed"}

    tool_message = ToolMessage(
        content=json.dumps(result, ensure_ascii=False),
        name=name or "unknown_tool",
        tool_call_id=str(call.get("id") or f"{name}-call"),
    )
    if event_sink is not None:
        event_sink(
            "tool_result",
            {"actor": actor, "name": name, "result": result},
        )
    return tool_message, result


def message_text(message: Any) -> str:
    """Convert common message content shapes to readable text."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse either a raw JSON object or one embedded in model prose."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
