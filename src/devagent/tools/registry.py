"""Tool registry bound to a runtime state."""

from collections.abc import Callable
from functools import partial
from typing import Any, TypedDict

from devagent.core.state import RuntimeState
from devagent.tools.bash_tool import run_command
from devagent.tools.file_tools import edit_file, read_file, search_text, write_file


class ToolDefinition(TypedDict):
    """Minimal metadata needed to expose a callable tool."""

    name: str
    description: str
    function: Callable[..., dict[str, Any]]


def build_tools(state: RuntimeState) -> dict[str, ToolDefinition]:
    """Build the tool registry with ``state`` bound to every callable."""
    return {
        "read_file": {
            "name": "read_file",
            "description": "Read lines from a UTF-8 file in the workspace.",
            "function": partial(read_file, state),
        },
        "write_file": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 file in the workspace.",
            "function": partial(write_file, state),
        },
        "edit_file": {
            "name": "edit_file",
            "description": "Replace text that occurs exactly once in a file.",
            "function": partial(edit_file, state),
        },
        "search_text": {
            "name": "search_text",
            "description": "Search workspace text files for a literal string.",
            "function": partial(search_text, state),
        },
        "run_command": {
            "name": "run_command",
            "description": "Run a command with the workspace as its directory.",
            "function": partial(run_command, state),
        },
    }

