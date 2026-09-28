"""File tools constrained to a runtime workspace."""

from itertools import islice
from pathlib import Path
from typing import Any

from devagent.core.state import RuntimeState


def _error_result(
    path: str | None,
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    """Build a structured file-tool error result."""
    error = {
        "code": code,
        "message": message,
    }
    error.update(details)
    return {
        "ok": False,
        "path": path,
        "error": error,
    }


def read_file(
    state: RuntimeState,
    file_path: Path | str,
    offset: int = 0,
    limit: int = 2000,
) -> dict[str, Any]:
    """Read at most ``limit`` text lines starting at ``offset``."""
    try:
        resolved_path = state.resolve_path(file_path)
    except (TypeError, ValueError):
        return _error_result(
            None,
            "invalid_path",
            "path is not valid within the workspace",
        )

    relative_path = str(resolved_path.relative_to(state.workspace))

    if offset < 0 or limit < 0:
        return _error_result(
            relative_path,
            "invalid_range",
            "offset and limit must be non-negative",
        )

    if not resolved_path.exists():
        return _error_result(
            relative_path,
            "file_not_found",
            "file does not exist",
        )

    if not resolved_path.is_file():
        return _error_result(
            relative_path,
            "not_a_file",
            "path is not a file",
        )

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            lines = list(islice(file, offset, offset + limit))
    except (OSError, UnicodeError):
        return _error_result(
            relative_path,
            "read_error",
            "unable to read file as UTF-8 text",
        )

    return {
        "ok": True,
        "path": relative_path,
        "content": "".join(lines),
        "offset": offset,
        "lines_returned": len(lines),
    }


def write_file(
    state: RuntimeState,
    file_path: Path | str,
    content: str,
) -> dict[str, Any]:
    """Create or replace a UTF-8 text file within the workspace."""
    try:
        resolved_path = state.resolve_path(file_path)
    except (TypeError, ValueError):
        return _error_result(
            None,
            "invalid_path",
            "path is not valid within the workspace",
        )

    relative_path = str(resolved_path.relative_to(state.workspace))

    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(content, encoding="utf-8")
    except (OSError, UnicodeError):
        return _error_result(
            relative_path,
            "write_error",
            "unable to write UTF-8 text file",
        )

    return {
        "ok": True,
        "path": relative_path,
        "characters_written": len(content),
    }


def edit_file(
    state: RuntimeState,
    file_path: Path | str,
    old_text: str,
    new_text: str,
) -> dict[str, Any]:
    """Replace text that occurs exactly once in an existing UTF-8 file."""
    try:
        resolved_path = state.resolve_path(file_path)
    except (TypeError, ValueError):
        return _error_result(
            None,
            "invalid_path",
            "path is not valid within the workspace",
        )

    relative_path = str(resolved_path.relative_to(state.workspace))

    if not resolved_path.exists():
        return _error_result(
            relative_path,
            "file_not_found",
            "file does not exist",
        )

    if not resolved_path.is_file():
        return _error_result(
            relative_path,
            "not_a_file",
            "path is not a file",
        )

    try:
        content = resolved_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return _error_result(
            relative_path,
            "read_error",
            "unable to read file as UTF-8 text",
        )

    matches = content.count(old_text)
    if matches != 1:
        return _error_result(
            relative_path,
            "match_count_mismatch",
            "old_text must match exactly once",
            matches=matches,
        )

    try:
        resolved_path.write_text(
            content.replace(old_text, new_text, 1),
            encoding="utf-8",
        )
    except (OSError, UnicodeError):
        return _error_result(
            relative_path,
            "write_error",
            "unable to write UTF-8 text file",
        )

    return {
        "ok": True,
        "path": relative_path,
        "replacements": 1,
    }
