"""Shared context budgeting, compression, and layered task memory."""

from dataclasses import dataclass
import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from devagent.execution import RunConfig, RunMetrics


STABLE_RULES = [
    "Operate only inside the configured workspace.",
    "Use relative paths for workspace file tools.",
    "Respect role-specific tool permissions and command approval decisions.",
    "Preserve task requirements and verification evidence during compression.",
]

PRESERVED_FIELDS = (
    "task",
    "plan_summary",
    "todos",
    "acceptance_criteria",
    "verification_commands",
    "research_notes",
    "sources",
    "handoffs",
    "changed_files",
    "verifier_feedback",
    "verifier_next_steps",
    "last_error",
    "attempts",
    "max_attempts",
)


class LayeredMemory:
    """Build stable rules, current working memory, and compressed history."""

    def build(self, state: dict[str, Any]) -> dict[str, Any]:
        working_memory = {
            key: _json_safe(state.get(key))
            for key in PRESERVED_FIELDS
            if key in state
        }
        return {
            "rules": {
                "workspace": str(
                    getattr(state.get("runtime"), "workspace", "")
                ),
                "items": list(STABLE_RULES),
            },
            "working_memory": working_memory,
            "history_summary": str(state.get("history_summary", "")),
        }


class ContextMonitor:
    """Estimate prompt size and decide whether compression is necessary."""

    def __init__(self, token_limit: int) -> None:
        if token_limit < 1:
            raise ValueError("token_limit must be at least 1")
        self.token_limit = token_limit

    def estimate_tokens(
        self,
        messages: list[Any],
        memory: dict[str, Any],
    ) -> int:
        text = json.dumps(
            {
                "messages": [_message_content(message) for message in messages],
                "memory": memory,
            },
            ensure_ascii=False,
            default=str,
        )
        return max(1, (len(text) + 3) // 4)

    def should_compress(
        self,
        messages: list[Any],
        memory: dict[str, Any],
    ) -> bool:
        return self.estimate_tokens(messages, memory) > self.token_limit


@dataclass
class CompressionResult:
    """Replacement transcript plus durable summary metadata."""

    messages: list[Any]
    history_summary: str
    before_messages: int
    after_messages: int


class ContextCompressor:
    """Replace raw history with a deterministic LayeredMemory summary."""

    def compress(
        self,
        messages: list[Any],
        state: dict[str, Any],
        memory: dict[str, Any],
    ) -> CompressionResult:
        previous_history = str(memory.get("history_summary", "")).strip()
        transcript = [
            text
            for text in (_message_content(message) for message in messages)
            if text
        ]
        recent_transcript = "\n".join(transcript[-12:])[-4000:]
        history_parts = [part for part in (previous_history, recent_transcript) if part]
        history_summary = "\n\n".join(history_parts)[-6000:]
        memory = {
            **memory,
            "history_summary": history_summary,
        }
        state["history_summary"] = history_summary
        state["context_summary"] = json.dumps(
            memory,
            ensure_ascii=False,
            default=str,
        )
        replacement = [
            SystemMessage(
                content=(
                    "DevAgent compressed prior context. Continue to obey these rules:\n"
                    + "\n".join(f"- {item}" for item in STABLE_RULES)
                )
            ),
            HumanMessage(
                content=(
                    "Authoritative layered memory after compression:\n"
                    + state["context_summary"]
                )
            ),
        ]
        return CompressionResult(
            messages=replacement,
            history_summary=history_summary,
            before_messages=len(messages),
            after_messages=len(replacement),
        )


def prepare_context(
    messages: list[Any],
    state: dict[str, Any],
    config: RunConfig,
    metrics: RunMetrics,
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[Any]:
    """Apply the common context budget and return the active transcript."""
    memory = LayeredMemory().build(state)
    monitor = ContextMonitor(int(config.context_token_limit or 1))
    estimated_tokens = monitor.estimate_tokens(messages, memory)
    if not monitor.should_compress(messages, memory):
        return messages
    result = ContextCompressor().compress(messages, state, memory)
    metrics.context_compressions += 1
    if event_sink is not None:
        event_sink(
            "context_compression",
            {
                "estimated_tokens": estimated_tokens,
                "token_limit": config.context_token_limit,
                "before_messages": result.before_messages,
                "after_messages": result.after_messages,
                "preserved_fields": list(PRESERVED_FIELDS),
            },
        )
    return result.messages


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(_json_safe(content), ensure_ascii=False, default=str)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
