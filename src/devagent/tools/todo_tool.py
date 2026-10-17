"""In-memory todo status updates for CodeAgent."""

from typing import Any


VALID_TODO_STATUSES = {"pending", "in_progress", "completed", "blocked"}


def update_todo(
    todos: list[dict[str, str]],
    todo_id: str,
    status: str,
    note: str = "",
) -> dict[str, Any]:
    """Update one existing todo while preserving all other items."""
    if status not in VALID_TODO_STATUSES:
        return {
            "ok": False,
            "error": "invalid todo status",
            "todos": todos,
        }

    for todo in todos:
        if todo.get("id") == todo_id:
            todo["status"] = status
            todo["note"] = note
            return {
                "ok": True,
                "todo_id": todo_id,
                "status": status,
                "note": note,
                "todos": todos,
            }

    return {
        "ok": False,
        "error": f"unknown todo_id: {todo_id}",
        "todos": todos,
    }
