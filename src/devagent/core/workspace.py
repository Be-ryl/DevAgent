"""Workspace snapshots used to detect command-driven file changes."""

from hashlib import sha256
from pathlib import Path


def snapshot_workspace(workspace: Path) -> dict[str, str]:
    """Hash workspace files while excluding internal runtime artifacts."""
    snapshot: dict[str, str] = {}
    if not workspace.exists():
        return snapshot
    for candidate in sorted(workspace.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(workspace)
        if relative.parts and relative.parts[0] in {".devagent", ".git"}:
            continue
        if "__pycache__" in relative.parts:
            continue
        try:
            digest = sha256(candidate.read_bytes()).hexdigest()
        except OSError:
            continue
        snapshot[relative.as_posix()] = digest
    return snapshot


def changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Return created, modified, and deleted paths between snapshots."""
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
