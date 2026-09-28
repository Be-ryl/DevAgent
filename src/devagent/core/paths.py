"""Workspace path validation helpers."""

from pathlib import Path


def resolve_workspace_path(
    workspace: Path | str,
    user_path: Path | str,
) -> Path:
    """Resolve a relative user path while keeping it inside ``workspace``.

    Absolute user paths and paths that escape through parent components or
    symbolic links are rejected.
    """
    relative_path = Path(user_path)
    if relative_path.is_absolute():
        raise ValueError("user path must be relative to the workspace")

    resolved_workspace = Path(workspace).expanduser().resolve()
    resolved_path = (resolved_workspace / relative_path).resolve()

    try:
        resolved_path.relative_to(resolved_workspace)
    except ValueError as error:
        raise ValueError("user path resolves outside the workspace") from error

    return resolved_path

