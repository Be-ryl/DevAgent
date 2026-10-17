"""Supervisor/Planner that delegates work through specialist tools."""

import shlex
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from devagent.agents.code_agent import CodeAgent
from devagent.agents.common import (
    bind_model_tools,
    invoke_model,
    invoke_tool_call,
    message_text,
    message_tool_calls,
)
from devagent.agents.prompts import PLANNER_PROMPT
from devagent.agents.search_agent import SearchAgent
from devagent.core.approval import ApprovalPolicy
from devagent.core.context import prepare_context
from devagent.execution import RunConfig, RunMetrics
from devagent.multiagent.state import MultiAgentState
from devagent.providers.model_provider import ModelProvider
from devagent.tools.registry import ToolArgs, ToolRegistry


class SetPlanArgs(ToolArgs):
    """Structured plan produced by the Supervisor."""

    plan_summary: str
    todos: list[str]
    acceptance_criteria: list[str]
    verification_commands: list[list[str] | str]


class DelegateArgs(BaseModel):
    """Instruction sent to one specialist agent."""

    instruction: str


class Supervisor:
    """Plan work and call specialists without direct workspace access."""

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

    def run(self, state: MultiAgentState) -> dict[str, Any]:
        working: dict[str, Any] = {
            **state,
            "todos": [dict(item) for item in state.get("todos", [])],
            "sources": [dict(item) for item in state.get("sources", [])],
            "handoffs": [dict(item) for item in state.get("handoffs", [])],
            "changed_files": list(state.get("changed_files", [])),
        }
        registry = self._build_registry(working)
        model = self.provider.create_model("planner", self.config.timeout_seconds)
        model = bind_model_tools(model, registry)
        messages: list[Any] = [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=self._planner_input(working)),
        ]
        summary = ""

        for _ in range(self.config.max_tool_rounds):
            messages = prepare_context(
                messages,
                working,
                self.config,
                self.metrics,
                self.event_sink,
            )
            response = invoke_model(
                model,
                messages,
                self.metrics,
                actor="Supervisor",
                event_sink=self.event_sink,
            )
            messages.append(response)
            calls = message_tool_calls(response)
            if not calls:
                summary = message_text(response)
                break
            for call in calls:
                tool_message, _ = invoke_tool_call(
                    registry,
                    call,
                    self.metrics,
                    actor="Supervisor",
                    event_sink=self.event_sink,
                )
                messages.append(tool_message)

        if not working.get("todos"):
            working.update(
                {
                    "plan_summary": summary or "Complete the requested task.",
                    "todos": [
                        {
                            "id": "todo-1",
                            "content": state["task"],
                            "status": "pending",
                            "note": "",
                        }
                    ],
                    "acceptance_criteria": ["The requested task is complete."],
                    "verification_commands": [],
                }
            )

        return {
            "plan_summary": working.get("plan_summary", summary),
            "todos": working["todos"],
            "acceptance_criteria": working.get("acceptance_criteria", []),
            "verification_commands": working.get("verification_commands", []),
            "research_notes": working.get("research_notes", ""),
            "sources": working["sources"],
            "handoffs": working["handoffs"],
            "changed_files": list(dict.fromkeys(working["changed_files"])),
            "messages": messages[2:],
            "context_summary": working.get("context_summary", ""),
            "history_summary": working.get("history_summary", ""),
        }

    def _build_registry(self, state: dict[str, Any]) -> ToolRegistry:
        def set_plan(
            plan_summary: str,
            todos: list[str],
            acceptance_criteria: list[str],
            verification_commands: list[list[str] | str],
        ) -> dict[str, Any]:
            state["plan_summary"] = plan_summary
            state["todos"] = [
                {
                    "id": f"todo-{index}",
                    "content": content,
                    "status": "pending",
                    "note": "",
                }
                for index, content in enumerate(todos, start=1)
            ]
            state["acceptance_criteria"] = list(acceptance_criteria)
            state["verification_commands"] = [
                shlex.split(command) if isinstance(command, str) else command
                for command in verification_commands
            ]
            return {
                "ok": True,
                "todos": state["todos"],
                "acceptance_criteria": state["acceptance_criteria"],
                "verification_commands": state["verification_commands"],
            }

        def call_search_agent(instruction: str) -> dict[str, Any]:
            if not state.get("todos"):
                return {
                    "ok": False,
                    "error": "SetPlanTool must be called before delegation.",
                }
            result = SearchAgent(
                self.provider,
                self.config,
                self.metrics,
                self.event_sink,
            ).run(
                state["task"],
                instruction,
                state.get("research_notes", ""),
            )
            state["research_notes"] = result["summary"]
            state["sources"] = self._merge_sources(
                state.get("sources", []),
                result["sources"],
            )
            state["handoffs"].append(
                {
                    "from_agent": "Supervisor",
                    "to_agent": "SearchAgent",
                    "instruction": instruction,
                    "result": result["summary"],
                    "queries": result["queries"],
                }
            )
            state["context_summary"] = result.get("context_summary", "")
            state["history_summary"] = result.get("history_summary", "")
            if self.event_sink is not None:
                self.event_sink("handoff", state["handoffs"][-1])
            return {
                "ok": True,
                "summary": result["summary"],
                "queries": result["queries"],
                "sources": result["sources"],
            }

        def call_code_agent(instruction: str) -> dict[str, Any]:
            if not state.get("todos"):
                return {
                    "ok": False,
                    "error": "SetPlanTool must be called before delegation.",
                }
            result = CodeAgent(
                self.provider,
                self.config,
                self.metrics,
                self.approval_policy,
                self.event_sink,
            ).run(
                runtime=state["runtime"],
                task=state["task"],
                instruction=instruction,
                todos=state.get("todos", []),
                research_notes=state.get("research_notes", ""),
                sources=state.get("sources", []),
            )
            state["todos"] = result["todos"]
            state["changed_files"].extend(result["changed_files"])
            state["handoffs"].append(
                {
                    "from_agent": "Supervisor",
                    "to_agent": "CodeAgent",
                    "instruction": instruction,
                    "result": result["summary"],
                }
            )
            state["context_summary"] = result.get("context_summary", "")
            state["history_summary"] = result.get("history_summary", "")
            if self.event_sink is not None:
                self.event_sink("handoff", state["handoffs"][-1])
            return {
                "ok": True,
                "summary": result["summary"],
                "todos": result["todos"],
                "changed_files": result["changed_files"],
            }

        return ToolRegistry(
            {
                "SetPlanTool": {
                    "name": "SetPlanTool",
                    "description": (
                        "Publish todos, acceptance criteria, and verification commands."
                    ),
                    "args_model": SetPlanArgs,
                    "function": set_plan,
                },
                "CallSearchAgentTool": {
                    "name": "CallSearchAgentTool",
                    "description": "Delegate external research to SearchAgent.",
                    "args_model": DelegateArgs,
                    "function": call_search_agent,
                },
                "CallCodeAgentTool": {
                    "name": "CallCodeAgentTool",
                    "description": "Delegate workspace implementation to CodeAgent.",
                    "args_model": DelegateArgs,
                    "function": call_code_agent,
                },
            }
        )

    @staticmethod
    def _planner_input(state: dict[str, Any]) -> str:
        return (
            f"Task: {state['task']}\n\n"
            f"Existing plan: {state.get('plan_summary', '')}\n"
            f"Todos: {state.get('todos', [])}\n"
            f"Research: {state.get('research_notes', '')}\n"
            f"Verifier feedback: {state.get('verifier_feedback', '')}\n"
            f"Verifier next steps: {state.get('verifier_next_steps', [])}\n"
            f"Verification attempt count: {state.get('attempts', 0)}"
        )

    @staticmethod
    def _merge_sources(
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in [*existing, *incoming]:
            url = str(source.get("url", ""))
            if url and url not in seen:
                seen.add(url)
                merged.append(source)
        return merged
