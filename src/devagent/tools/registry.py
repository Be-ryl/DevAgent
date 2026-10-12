"""Tool registry bound to a runtime state."""

from collections.abc import Callable
from functools import partial
import json
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from devagent.core.state import RuntimeState
from devagent.tools.bash_tool import run_command
from devagent.tools.file_tools import edit_file, read_file, search_text, write_file
from devagent.tools.knowledge_tool import knowledge_search

if TYPE_CHECKING:
    from devagent.rag.dense_index import DenseRagService
    from devagent.rag.service import BM25RagService


class ToolArgs(BaseModel):
    """Base contract shared by arguments accepted from model tool calls."""

    model_config = ConfigDict(extra="forbid")


class ReadFileArgs(ToolArgs):
    """Arguments for ``read_file``."""

    file_path: str
    offset: int = 0
    limit: int = 2000


class WriteFileArgs(ToolArgs):
    """Arguments for ``write_file``."""

    file_path: str
    content: str


class EditFileArgs(ToolArgs):
    """Arguments for ``edit_file``."""

    file_path: str
    old_text: str
    new_text: str


class SearchTextArgs(ToolArgs):
    """Arguments for ``search_text``."""

    search_path: str
    query: str
    max_results: int = 100


class RunCommandArgs(ToolArgs):
    """Arguments for ``run_command``."""

    command: list[str]
    timeout: float = 30.0


class KnowledgeSearchArgs(ToolArgs):
    """Arguments for ``knowledge_search``."""

    query: str
    top_k: int = Field(default=5, ge=1, le=10)


class ToolDefinition(TypedDict):
    """Contract and callable needed to expose and execute a tool."""

    name: str
    description: str
    args_model: type[BaseModel]
    function: Callable[..., dict[str, Any]]


class UnknownToolError(LookupError):
    """Raised when a requested tool is not registered."""


class InvalidToolArgumentsError(ValueError):
    """Raised when tool-call arguments cannot be parsed or validated."""


class ToolRegistry(dict[str, ToolDefinition]):
    """Dictionary-compatible registry with schema export and invocation."""

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-serializable OpenAI-compatible tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["args_model"].model_json_schema(),
                },
            }
            for tool in self.values()
        ]

    def invoke(
        self,
        name: str,
        raw_arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        """Parse, validate, and execute one registered tool call."""
        try:
            tool = self[name]
        except KeyError as error:
            raise UnknownToolError(f"unknown tool: {name}") from error

        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise InvalidToolArgumentsError("tool arguments are not valid JSON") from error
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise InvalidToolArgumentsError(
                "tool arguments must be a JSON object or dictionary"
            )

        if not isinstance(arguments, dict):
            raise InvalidToolArgumentsError("tool arguments must decode to an object")

        try:
            validated = tool["args_model"].model_validate(arguments)
        except ValidationError as error:
            raise InvalidToolArgumentsError("tool arguments failed validation") from error

        return tool["function"](**validated.model_dump())


def build_tools(
    state: RuntimeState,
    rag_service: "BM25RagService | DenseRagService | None" = None,
) -> ToolRegistry:
    """Build tools and optionally bind one prebuilt local RAG service."""
    tools = ToolRegistry({
        "read_file": {
            "name": "read_file",
            "description": "Read lines from a UTF-8 file in the workspace.",
            "args_model": ReadFileArgs,
            "function": partial(read_file, state),
        },
        "write_file": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 file in the workspace.",
            "args_model": WriteFileArgs,
            "function": partial(write_file, state),
        },
        "edit_file": {
            "name": "edit_file",
            "description": "Replace text that occurs exactly once in a file.",
            "args_model": EditFileArgs,
            "function": partial(edit_file, state),
        },
        "search_text": {
            "name": "search_text",
            "description": "Search workspace text files for a literal string.",
            "args_model": SearchTextArgs,
            "function": partial(search_text, state),
        },
        "run_command": {
            "name": "run_command",
            "description": "Run a command with the workspace as its directory.",
            "args_model": RunCommandArgs,
            "function": partial(run_command, state),
        },
    })
    if rag_service is not None:
        tools["knowledge_search"] = {
            "name": "knowledge_search",
            "description": (
                "Search the local travel policy knowledge base and return "
                "evidence chunks needed to answer the user's question."
            ),
            "args_model": KnowledgeSearchArgs,
            "function": partial(knowledge_search, rag_service),
        }
    return tools

