"""Research specialist restricted to Tavily web search."""

from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from devagent.agents.common import (
    bind_model_tools,
    invoke_model,
    invoke_tool_call,
    message_text,
    message_tool_calls,
)
from devagent.agents.prompts import SEARCH_AGENT_PROMPT
from devagent.core.context import prepare_context
from devagent.execution import RunConfig, RunMetrics
from devagent.providers.model_provider import ModelProvider
from devagent.tools.registry import build_web_search_tools


class SearchAgent:
    """Run a model loop whose sole capability is ``WebSearchTool``."""

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

    def run(self, task: str, instruction: str, notes: str = "") -> dict[str, Any]:
        registry = build_web_search_tools()
        model = self.provider.create_model("search", self.config.timeout_seconds)
        model = bind_model_tools(model, registry)
        messages: list[Any] = [
            SystemMessage(content=SEARCH_AGENT_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {task}\n\nPlanner instruction: {instruction}\n\n"
                    f"Existing research notes: {notes}"
                )
            ),
        ]
        queries: list[str] = []
        sources: list[dict[str, Any]] = []
        summary = ""
        context_state: dict[str, Any] = {
            "task": task,
            "research_notes": notes,
            "sources": sources,
            "handoffs": [],
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
                actor="SearchAgent",
                event_sink=self.event_sink,
            )
            messages.append(response)
            calls = message_tool_calls(response)
            if not calls:
                summary = message_text(response)
                break
            for call in calls:
                if call.get("name") == "WebSearchTool":
                    query = str((call.get("args") or {}).get("query", ""))
                    if query:
                        queries.append(query)
                tool_message, result = invoke_tool_call(
                    registry,
                    call,
                    self.metrics,
                    actor="SearchAgent",
                    event_sink=self.event_sink,
                )
                messages.append(tool_message)
                if result.get("answer") and not summary:
                    summary = str(result["answer"])
                for source in result.get("results", []) or []:
                    if isinstance(source, dict):
                        sources.append(source)
                context_state["research_notes"] = summary
                context_state["sources"] = sources

        unique_sources: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for source in sources:
            url = str(source.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
        return {
            "summary": summary,
            "queries": queries,
            "sources": unique_sources,
            "messages": messages[2:],
            "context_summary": context_state.get("context_summary", ""),
            "history_summary": context_state.get("history_summary", ""),
        }
