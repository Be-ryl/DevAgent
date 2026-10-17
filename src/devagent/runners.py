"""Comparable single-agent and multi-agent execution runners."""

from time import perf_counter
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from devagent.agents.common import (
    bind_model_tools,
    invoke_model,
    invoke_tool_call,
    message_text,
    message_tool_calls,
)
from devagent.agents.prompts import SINGLE_AGENT_PROMPT
from devagent.core.approval import ApprovalPolicy
from devagent.core.checkpoint import CheckpointManager
from devagent.core.context import prepare_context
from devagent.core.state import RuntimeState
from devagent.core.trace import TraceRecorder
from devagent.execution import (
    ExecutionMode,
    RunConfig,
    RunMetrics,
    RunResult,
)
from devagent.multiagent.workflow import build_multiagent_workflow
from devagent.providers.model_provider import DeepSeekModelProvider, ModelProvider
from devagent.tools.registry import build_single_tools


class Runner(Protocol):
    """Common execution interface."""

    def run(self, config: RunConfig) -> RunResult:
        """Run one task and return a mode-independent result."""


class SingleAgentRunner:
    """One model loop with the complete tool set and shared harness."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def run(self, config: RunConfig) -> RunResult:
        metrics = RunMetrics(execution_mode=ExecutionMode.SINGLE)
        started_at = perf_counter()
        runtime = RuntimeState(config.workspace)
        trace = TraceRecorder(runtime, metrics, config.task)
        approval = ApprovalPolicy(
            config.approval_mode,
            config.approval_callback,
            metrics,
            trace.record,
        )
        checkpoint = CheckpointManager(runtime, metrics, trace.record)
        state = _initial_single_state(config, runtime)
        changed_files: list[str] = []
        final_answer = ""
        status = "failed"

        try:
            if config.resume:
                resumed = checkpoint.load()
                state.update(resumed)
                state["runtime"] = runtime
                state["execution_mode"] = ExecutionMode.SINGLE.value
                if config.task:
                    state["task"] = config.task
                changed_files.extend(state.get("changed_files", []))
                trace.record("checkpoint_resume", {"state": resumed})

            trace.task = str(state.get("task", config.task))
            trace.start(
                execution_mode=ExecutionMode.SINGLE.value,
                resumed=config.resume,
            )
            checkpoint.save(
                state,
                status="running",
                latest_node="single",
            )
            registry = build_single_tools(
                runtime,
                config.timeout_seconds,
                approval,
            )
            model = self.provider.create_model(
                "single",
                config.timeout_seconds,
            )
            model = bind_model_tools(model, registry)
            messages: list[Any] = [
                SystemMessage(content=SINGLE_AGENT_PROMPT),
                HumanMessage(content=_single_task_input(state)),
            ]

            for _ in range(config.max_tool_rounds):
                messages = prepare_context(
                    messages,
                    state,
                    config,
                    metrics,
                    trace.record,
                )
                response = invoke_model(
                    model,
                    messages,
                    metrics,
                    actor="SingleAgent",
                    event_sink=trace.record,
                )
                messages.append(response)
                calls = message_tool_calls(response)
                if not calls:
                    final_answer = message_text(response)
                    state["final_answer"] = final_answer
                    metrics.completed = True
                    status = "completed"
                    break
                for call in calls:
                    tool_message, result = invoke_tool_call(
                        registry,
                        call,
                        metrics,
                        actor="SingleAgent",
                        event_sink=trace.record,
                    )
                    messages.append(tool_message)
                    if call.get("name") in {"write_file", "edit_file"}:
                        result_path = result.get("path")
                        if result.get("ok") and result_path:
                            changed_files.append(str(result_path))
                    if call.get("name") == "run_command":
                        changed_files.extend(result.get("changed_files", []))
                    state["changed_files"] = list(dict.fromkeys(changed_files))
                    checkpoint.save(
                        state,
                        status="running",
                        latest_node="single_tool",
                    )
            else:
                metrics.error = "single-agent tool loop limit reached"
                state["last_error"] = metrics.error
                final_answer = metrics.error
        except KeyboardInterrupt:
            metrics.error = "interrupted"
            state["last_error"] = metrics.error
            final_answer = "DevAgent interrupted. Resume with --resume."
            status = "interrupted"
            trace.record("error", {"error": metrics.error})
            checkpoint.save(
                state,
                status="interrupted",
                latest_node="single",
            )
        except Exception as error:
            metrics.error = f"{type(error).__name__}: {error}"
            state["last_error"] = metrics.error
            final_answer = f"DevAgent failed: {metrics.error}"
            trace.record("error", {"error": metrics.error})
            checkpoint.save(
                state,
                status="failed",
                latest_node="single",
            )
        finally:
            changed_files = list(dict.fromkeys(changed_files))
            state["changed_files"] = changed_files
            metrics.changed_file_count = len(changed_files)
            metrics.elapsed_seconds = perf_counter() - started_at
            if status == "completed":
                checkpoint.save(
                    state,
                    status="completed",
                    latest_node="single",
                )
            trace.finish(status=status, state=state)

        return RunResult(
            final_answer=final_answer,
            metrics=metrics,
            verified=None,
            changed_files=changed_files,
            state=state,
        )


class MultiAgentRunner:
    """Run the role-isolated LangGraph workflow with the shared harness."""

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def run(self, config: RunConfig) -> RunResult:
        metrics = RunMetrics(execution_mode=ExecutionMode.MULTI)
        started_at = perf_counter()
        runtime = RuntimeState(config.workspace)
        trace = TraceRecorder(runtime, metrics, config.task)
        approval = ApprovalPolicy(
            config.approval_mode,
            config.approval_callback,
            metrics,
            trace.record,
        )
        checkpoint = CheckpointManager(runtime, metrics, trace.record)
        initial_state = _initial_multi_state(config, runtime)
        final_state: dict[str, Any] = {}
        final_answer = ""
        status = "failed"

        try:
            if config.resume:
                resumed = checkpoint.load()
                initial_state.update(resumed)
                initial_state["runtime"] = runtime
                initial_state["messages"] = []
                initial_state["execution_mode"] = ExecutionMode.MULTI.value
                initial_state["max_attempts"] = config.max_attempts
                if config.task:
                    initial_state["task"] = config.task
                trace.record("checkpoint_resume", {"state": resumed})

            trace.task = str(initial_state.get("task", config.task))
            trace.start(
                execution_mode=ExecutionMode.MULTI.value,
                resumed=config.resume,
            )
            checkpoint.save(
                initial_state,
                status="running",
                latest_node="start",
            )
            workflow = build_multiagent_workflow(
                self.provider,
                config,
                metrics,
                approval,
                trace.record,
                checkpoint,
            )
            final_state = workflow.invoke(
                initial_state,
                config={"recursion_limit": config.max_attempts * 3 + 5},
            )
            final_answer = str(final_state.get("final_answer", ""))
            metrics.retry_count = max(
                0,
                int(final_state.get("attempts", 0)) - 1,
            )
            metrics.completed = bool(final_answer)
            status = "completed" if metrics.completed else "failed"
        except KeyboardInterrupt:
            metrics.error = "interrupted"
            try:
                recovered_state = checkpoint.load()
            except (FileNotFoundError, OSError, ValueError):
                recovered_state = initial_state
            final_state = final_state or recovered_state
            final_state["runtime"] = runtime
            final_state["messages"] = []
            final_state["last_error"] = metrics.error
            final_answer = "DevAgent interrupted. Resume with --resume."
            status = "interrupted"
            trace.record("error", {"error": metrics.error})
            checkpoint.save(
                final_state,
                status="interrupted",
                latest_node="workflow",
            )
        except Exception as error:
            metrics.error = f"{type(error).__name__}: {error}"
            final_state = final_state or initial_state
            final_state["last_error"] = metrics.error
            final_answer = f"DevAgent failed: {metrics.error}"
            trace.record("error", {"error": metrics.error})
            checkpoint.save(
                final_state,
                status="failed",
                latest_node="workflow",
            )
        finally:
            changed_files = list(final_state.get("changed_files", []))
            metrics.changed_file_count = len(changed_files)
            metrics.elapsed_seconds = perf_counter() - started_at
            trace.finish(status=status, state=final_state or initial_state)

        return RunResult(
            final_answer=final_answer,
            metrics=metrics,
            verified=final_state.get("verified") if final_state else False,
            changed_files=list(final_state.get("changed_files", [])),
            state=final_state,
        )


class RunnerFactory:
    """Create the requested architecture without initializing the other one."""

    def __init__(self, provider: ModelProvider | None = None) -> None:
        self.provider = provider or DeepSeekModelProvider()

    def create(self, mode: ExecutionMode | str) -> Runner:
        execution_mode = ExecutionMode(mode)
        if execution_mode is ExecutionMode.SINGLE:
            return SingleAgentRunner(self.provider)
        return MultiAgentRunner(self.provider)


def _initial_single_state(
    config: RunConfig,
    runtime: RuntimeState,
) -> dict[str, Any]:
    return {
        "task": config.task,
        "runtime": runtime,
        "execution_mode": ExecutionMode.SINGLE.value,
        "todos": [],
        "plan_summary": "",
        "acceptance_criteria": [],
        "verification_commands": [],
        "sources": [],
        "handoffs": [],
        "changed_files": [],
        "last_error": "",
        "context_summary": "",
        "history_summary": "",
    }


def _initial_multi_state(
    config: RunConfig,
    runtime: RuntimeState,
) -> dict[str, Any]:
    return {
        "task": config.task,
        "runtime": runtime,
        "execution_mode": ExecutionMode.MULTI.value,
        "messages": [],
        "todos": [],
        "acceptance_criteria": [],
        "verification_commands": [],
        "research_notes": "",
        "sources": [],
        "handoffs": [],
        "verifier_feedback": "",
        "verifier_next_steps": [],
        "verification_results": [],
        "attempts": 0,
        "max_attempts": config.max_attempts,
        "verified": False,
        "final_answer": "",
        "plan_summary": "",
        "changed_files": [],
        "context_summary": "",
        "history_summary": "",
        "last_error": "",
    }


def _single_task_input(state: dict[str, Any]) -> str:
    context_summary = str(state.get("context_summary", ""))
    if not context_summary:
        return str(state.get("task", ""))
    return (
        f"Task: {state.get('task', '')}\n\n"
        f"Recovered context:\n{context_summary}"
    )
