"""State contract for the stage-two MultiAgent workflow."""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from devagent.core.state import RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class MultiAgentState(TypedDict, total=False):
    """Data shared by Planner, specialists, Verifier, and Final."""

    task: str
    messages: list[BaseMessage]
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[list[str]]
    research_notes: str
    sources: list[dict[str, Any]]
    handoffs: list[dict[str, Any]]
    verifier_feedback: str
    verifier_next_steps: list[str]
    verification_results: list[dict[str, Any]]
    attempts: int
    max_attempts: int
    verified: bool
    final_answer: str
    runtime: RuntimeState
    plan_summary: str
    changed_files: list[str]
    context_summary: str
    history_summary: str
    last_error: str
