"""Shared execution contracts for single-agent and multi-agent runners."""

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Any, Callable


class ExecutionMode(str, Enum):
    """Supported execution architectures."""

    SINGLE = "single"
    MULTI = "multi"


@dataclass
class RunConfig:
    """Configuration shared by both runner implementations."""

    task: str
    workspace: Path | str = field(default_factory=Path.cwd)
    mode: ExecutionMode = ExecutionMode.SINGLE
    max_tool_rounds: int = 8
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    context_token_limit: int | None = None
    approval_mode: str = "inline"
    approval_callback: Callable[[Any], Any] | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)
        if isinstance(self.mode, str):
            self.mode = ExecutionMode(self.mode)
        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.context_token_limit is None:
            raw_limit = os.environ.get("DEVAGENT_CONTEXT_TOKEN_LIMIT", "120000")
            try:
                self.context_token_limit = int(raw_limit)
            except ValueError:
                self.context_token_limit = 120000
        if self.context_token_limit < 1:
            raise ValueError("context_token_limit must be at least 1")
        if self.approval_mode not in {"inline", "deny", "auto"}:
            raise ValueError("approval_mode must be inline, deny, or auto")


@dataclass
class RunMetrics:
    """Comparable counters collected independently for one run."""

    elapsed_seconds: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0
    context_compressions: int = 0
    approval_requests: int = 0
    approval_denials: int = 0
    checkpoints: int = 0
    trace_id: str = ""
    changed_file_count: int = 0
    execution_mode: ExecutionMode = ExecutionMode.SINGLE
    completed: bool = False
    error: str | None = None


@dataclass
class RunResult:
    """Mode-independent result returned to the CLI and future evaluators."""

    final_answer: str
    metrics: RunMetrics
    verified: bool | None = None
    changed_files: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Reserved interface for a future independent architecture evaluator."""

    def evaluate(self, result: RunResult) -> dict[str, Any]:
        raise NotImplementedError
