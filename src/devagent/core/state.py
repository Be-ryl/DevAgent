"""Runtime state for DevAgent."""

from dataclasses import dataclass
from pathlib import Path

from devagent.core.paths import resolve_workspace_path


@dataclass
class RuntimeState:
    """State shared by a DevAgent runtime."""

    workspace: Path

    def __post_init__(self) -> None:
        """Normalize the workspace and create it when necessary."""
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, user_path: Path | str) -> Path:
        """Resolve a user path safely within this runtime's workspace."""
        return resolve_workspace_path(self.workspace, user_path)

