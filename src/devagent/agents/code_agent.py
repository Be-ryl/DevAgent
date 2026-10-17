"""Implementation specialist with workspace-constrained coding tools."""

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from devagent.agents.common import (
    bind_model_tools,
    invoke_model,
    invoke_tool_call,
    message_text,
    message_tool_calls,
)
from devagent.agents.prompts import CODE_AGENT_PROMPT
from devagent.core.approval import ApprovalPolicy
from devagent.core.context import prepare_context
from devagent.core.state import RuntimeState
from devagent.execution import RunConfig, RunMetrics
from devagent.providers.model_provider import ModelProvider
from devagent.tools.registry import build_code_tools


class CodeAgent:
    """Apply Planner instructions using only coding and todo tools."""

    def __init__(
        self,
        provider: ModelProvider,
        config: RunConfig,
        metrics: RunMetrics,
        approval_policy: ApprovalPolicy | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.metrics = metrics
        self.approval_policy = approval_policy
        self.event_sink = event_sink

    def run(
        self,
        runtime: RuntimeState,
        task: str,
        instruction: str,
        todos: list[dict[str, str]],
        research_notes: str = "",
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        working_todos = [dict(todo) for todo in todos]
        registry = build_code_tools(
            runtime,
            working_todos,
            self.config.timeout_seconds,
            self.approval_policy,
        )
        model = self.provider.create_model("code", self.config.timeout_seconds)
        model = bind_model_tools(model, registry)
        messages: list[Any] = [
            SystemMessage(content=CODE_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {task}\n\nPlanner instruction: {instruction}\n\n"
                    f"Todos: {working_todos}\n\nResearch: {research_notes}\n\n"
                    f"Sources: {sources or []}"
                )
            ),
        ]
        summary = ""
        changed_files: list[str] = []
        context_state: dict[str, Any] = {
            "runtime": runtime,
            "task": task,
            "todos": working_todos,
            "research_notes": research_notes,
            "sources": sources or [],
            "handoffs": [],
            "changed_files": changed_files,
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
                actor="CodeAgent",
                event_sink=self.event_sink,
            )
            messages.append(response)
            calls = message_tool_calls(response)
            if not calls:
                summary = message_text(response)
                break
            for call in calls:
                tool_message, result = invoke_tool_call(
                    registry,
                    call,
                    self.metrics,
                    actor="CodeAgent",
                    event_sink=self.event_sink,
                )
                messages.append(tool_message)
                if call.get("name") in {"write_file", "edit_file"}:
                    result_path = result.get("path")
                    if result.get("ok") and result_path:
                        changed_files.append(str(result_path))
                if call.get("name") == "run_command":
                    changed_files.extend(result.get("changed_files", []))
                context_state["changed_files"] = list(
                    dict.fromkeys(changed_files)
                )

        return {
            "summary": summary,
            "todos": working_todos,
            "changed_files": list(dict.fromkeys(changed_files)),
            "messages": messages[2:],
            "context_summary": context_state.get("context_summary", ""),
            "history_summary": context_state.get("history_summary", ""),
        }
