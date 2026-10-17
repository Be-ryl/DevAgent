"""Independent, read-only acceptance verifier."""

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from devagent.agents.common import (
    bind_model_tools,
    extract_json,
    invoke_model,
    invoke_tool_call,
    message_text,
    message_tool_calls,
)
from devagent.agents.prompts import VERIFIER_PROMPT
from devagent.core.context import prepare_context
from devagent.core.state import RuntimeState
from devagent.execution import RunConfig, RunMetrics
from devagent.providers.model_provider import ModelProvider
from devagent.tools.registry import build_verifier_tools


class Verifier:
    """Inspect deliverables without any file mutation tools."""

    def __init__(
        self,
        provider: ModelProvider,
        config: RunConfig,
        metrics: RunMetrics,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.metrics = metrics
        self.event_sink = event_sink

    def run(
        self,
        runtime: RuntimeState,
        task: str,
        acceptance_criteria: list[str],
        verification_commands: list[list[str]],
        handoffs: list[dict[str, Any]],
        verification_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        verification_results = verification_results or []
        registry = build_verifier_tools(
            runtime,
            self.config.timeout_seconds,
        )
        model = self.provider.create_model("verifier", self.config.timeout_seconds)
        model = bind_model_tools(model, registry)
        messages: list[Any] = [
            SystemMessage(content=VERIFIER_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {task}\n\nAcceptance criteria: {acceptance_criteria}\n\n"
                    f"Verification commands: {verification_commands}\n\n"
                    f"Pre-executed verification results: {verification_results}\n\n"
                    f"Specialist handoffs: {handoffs}"
                )
            ),
        ]
        text = ""
        context_state: dict[str, Any] = {
            "runtime": runtime,
            "task": task,
            "acceptance_criteria": acceptance_criteria,
            "verification_commands": verification_commands,
            "verification_results": verification_results,
            "handoffs": handoffs,
            "changed_files": [],
            "last_error": "",
        }

        for _ in range(self.config.max_tool_rounds):
            messages = prepare_context(
                messages,
                context_state,
                self.config,
                self.metrics,
                self.event_sink,
            )
            response = invoke_model(
                model,
                messages,
                self.metrics,
                actor="Verifier",
                event_sink=self.event_sink,
            )
            messages.append(response)
            calls = message_tool_calls(response)
            if not calls:
                text = message_text(response)
                break
            for call in calls:
                tool_message, result = invoke_tool_call(
                    registry,
                    call,
                    self.metrics,
                    actor="Verifier",
                    event_sink=self.event_sink,
                )
                messages.append(tool_message)

        parsed = extract_json(text)
        if parsed is None:
            return {
                "verified": False,
                "feedback": "Verifier did not return valid JSON.",
                "next_steps": ["Return a valid verification decision."],
                "messages": messages[2:],
                "context_summary": context_state.get("context_summary", ""),
                "history_summary": context_state.get("history_summary", ""),
            }
        return {
            "verified": bool(parsed.get("verified")),
            "feedback": str(parsed.get("reason") or ""),
            "next_steps": [str(item) for item in parsed.get("next_steps", [])],
            "messages": messages[2:],
            "context_summary": context_state.get("context_summary", ""),
            "history_summary": context_state.get("history_summary", ""),
        }
