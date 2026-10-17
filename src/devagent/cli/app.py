"""Command-line interface for one DevAgent execution."""

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from devagent.execution import ExecutionMode, RunConfig
from devagent.core.approval import ApprovalDecision, ApprovalRequest
from devagent.core.checkpoint import load_checkpoint_state
from devagent.runners import RunnerFactory


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="devagent",
        description="Run one DevAgent request.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace available to local tools (default: current directory)",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ExecutionMode],
        default=None,
        help="execution architecture (default: single)",
    )
    parser.add_argument(
        "--approval-mode",
        choices=["inline", "deny", "auto"],
        default="inline",
        help="policy for high-risk commands (default: inline)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from .devagent/checkpoints in this workspace",
    )
    parser.add_argument(
        "message",
        nargs="?",
        help="user message to send to the agent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI request and return a process exit status."""
    args = build_parser().parse_args(argv)
    if not args.message and args.resume is None:
        build_parser().error("message is required unless --resume is used")
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "Error: DEEPSEEK_API_KEY is not set. Export it before running DevAgent.",
            file=sys.stderr,
        )
        return 2

    workspace = args.resume if args.resume is not None else args.workspace
    mode_value = args.mode
    if args.resume is not None and mode_value is None:
        try:
            mode_value = str(
                load_checkpoint_state(args.resume).get("execution_mode", "")
            )
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            mode_value = ""
    mode = ExecutionMode(mode_value or ExecutionMode.SINGLE.value)
    approval_callback = (
        _interactive_approval
        if args.approval_mode == "inline"
        else None
    )
    config = RunConfig(
        task=args.message or "",
        workspace=workspace,
        mode=mode,
        approval_mode=args.approval_mode,
        approval_callback=approval_callback,
        resume=args.resume is not None,
    )
    runner = RunnerFactory().create(config.mode)
    result = runner.run(config)
    print(result.final_answer)
    return 1 if result.metrics.error else 0


def _interactive_approval(request: ApprovalRequest) -> ApprovalDecision:
    """CLI-only prompt; core approval logic remains callback-driven."""
    print(
        (
            f"High-risk command requested by {request.actor}: "
            f"{request.display_command}\nRisk: {request.risk_reason}"
        ),
        file=sys.stderr,
    )
    answer = input("Approve? [y/N] ").strip().lower()
    return ApprovalDecision(answer in {"y", "yes"}, "user decision")
