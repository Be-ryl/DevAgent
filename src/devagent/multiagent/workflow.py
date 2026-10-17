"""LangGraph assembly for the stage-two MultiAgent workflow."""

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from devagent.agents.supervisor import Supervisor
from devagent.agents.verifier import Verifier
from devagent.core.approval import ApprovalPolicy
from devagent.core.checkpoint import CheckpointManager
from devagent.core.verification import VerificationExecutor
from devagent.execution import RunConfig, RunMetrics
from devagent.multiagent.state import MultiAgentState
from devagent.providers.model_provider import ModelProvider


def build_multiagent_workflow(
    provider: ModelProvider,
    config: RunConfig,
    metrics: RunMetrics,
    approval_policy: ApprovalPolicy,
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    checkpoint: CheckpointManager | None = None,
):
    """Compile ``START -> planner -> verifier -> final -> END`` with retries."""
    supervisor = Supervisor(
        provider,
        config,
        metrics,
        approval_policy,
        event_sink,
    )

    def planner_node(state: MultiAgentState) -> dict[str, Any]:
        update = supervisor.run(state)
        if checkpoint is not None:
            checkpoint.save(
                {**state, **update},
                status="running",
                latest_node="planner",
            )
        return update

    def verifier_node(state: MultiAgentState) -> dict[str, Any]:
        executor = VerificationExecutor(
            state["runtime"],
            state.get("verification_commands", []),
            timeout_seconds=config.timeout_seconds,
            approval_policy=approval_policy,
            metrics=metrics,
            event_sink=event_sink,
        )
        command_results = executor.execute_planned()
        changed_files = list(state.get("changed_files", []))
        for command_result in command_results:
            changed_files.extend(command_result.get("changed_files", []))
        changed_files = list(dict.fromkeys(changed_files))

        result = Verifier(provider, config, metrics, event_sink).run(
            runtime=state["runtime"],
            task=state["task"],
            acceptance_criteria=state.get("acceptance_criteria", []),
            verification_commands=state.get("verification_commands", []),
            handoffs=state.get("handoffs", []),
            verification_results=command_results,
        )
        attempts = state.get("attempts", 0) + 1
        todos = [dict(todo) for todo in state.get("todos", [])]
        if result["verified"]:
            for todo in todos:
                if todo.get("status") != "blocked":
                    todo["status"] = "completed"
                    todo["note"] = todo.get("note") or "verified"
        update = {
            "verified": result["verified"],
            "verifier_feedback": result["feedback"],
            "verifier_next_steps": result["next_steps"],
            "verification_results": command_results,
            "attempts": attempts,
            "todos": todos,
            "messages": result["messages"],
            "changed_files": changed_files,
            "context_summary": result.get("context_summary", ""),
            "history_summary": result.get("history_summary", ""),
            "last_error": "" if result["verified"] else result["feedback"],
        }
        if event_sink is not None:
            event_sink(
                "verifier_result",
                {
                    "verified": result["verified"],
                    "feedback": result["feedback"],
                    "attempt": attempts,
                },
            )
        if checkpoint is not None:
            checkpoint.save(
                {**state, **update},
                status="running",
                latest_node="verifier",
            )
        return update

    def verifier_route(state: MultiAgentState) -> str:
        if state.get("verified"):
            return "final"
        if state.get("attempts", 0) >= state.get("max_attempts", 1):
            return "final"
        return "planner"

    def final_node(state: MultiAgentState) -> dict[str, str]:
        status = "PASSED" if state.get("verified") else "FAILED"
        files = state.get("changed_files", [])
        implementation_results = [
            str(handoff.get("result", ""))
            for handoff in state.get("handoffs", [])
            if handoff.get("to_agent") == "CodeAgent" and handoff.get("result")
        ]
        test_results = []
        for result in state.get("verification_results", []):
            returncode = result.get("returncode")
            outcome = "passed" if result.get("ok") and returncode == 0 else "failed"
            test_results.append(
                f"{outcome} (returncode={returncode})"
            )
        sources = [
            str(source.get("url"))
            for source in state.get("sources", [])
            if source.get("url")
        ]
        risks = (
            "None identified by the verifier."
            if state.get("verified")
            else state.get("verifier_feedback", "Acceptance was not verified.")
        )
        answer = "\n".join(
            [
                f"MultiAgent result: {status}",
                f"Plan: {state.get('plan_summary', '')}",
                "Implementation: "
                + ("; ".join(implementation_results) if implementation_results else "none reported"),
                "Tests: "
                + ("; ".join(test_results) if test_results else "none reported"),
                f"Files: {', '.join(files) if files else 'none reported'}",
                f"Verification: {state.get('verifier_feedback', '')}",
                "Next steps: "
                + (
                    "; ".join(state.get("verifier_next_steps", []))
                    if state.get("verifier_next_steps")
                    else "none"
                ),
                f"Sources: {', '.join(sources) if sources else 'none'}",
                f"Unresolved risks: {risks}",
            ]
        )
        update = {"final_answer": answer}
        if checkpoint is not None:
            checkpoint.save(
                {**state, **update},
                status="completed",
                latest_node="final",
            )
        return update

    graph = StateGraph(MultiAgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {"planner": "planner", "final": "final"},
    )
    graph.add_edge("final", END)
    return graph.compile()
