"""Command-line interface for one DevAgent request."""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from devagent.agent import run_agent
from devagent.core.state import RuntimeState
from devagent.tools.registry import build_tools


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
    parser.add_argument("message", help="user message to send to the agent")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI request and return a process exit status."""
    args = build_parser().parse_args(argv)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "Error: DEEPSEEK_API_KEY is not set. Export it before running DevAgent.",
            file=sys.stderr,
        )
        return 2

    state = RuntimeState(args.workspace)
    registry = build_tools(state)
    answer = run_agent(args.message, registry)
    print(answer)
    return 0
