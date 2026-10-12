"""Command-line interface for one DevAgent request."""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from devagent.agent import run_agent
from devagent.core.state import RuntimeState
from devagent.rag.service import BM25RagService
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
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        help="directory containing local policy Markdown files",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print knowledge tool calls and retrieval hits to stderr",
    )
    parser.add_argument("message", help="user message to send to the agent")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI request and return a process exit status."""
    args = build_parser().parse_args(argv)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    knowledge_directory = (
        args.knowledge_dir.expanduser()
        if args.knowledge_dir is not None
        else None
    )
    if knowledge_directory is not None and not knowledge_directory.is_dir():
        print(
            f"Error: knowledge directory is not a directory: {knowledge_directory}",
            file=sys.stderr,
        )
        return 2

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "Error: DEEPSEEK_API_KEY is not set. Export it before running DevAgent.",
            file=sys.stderr,
        )
        return 2

    state = RuntimeState(args.workspace)
    if knowledge_directory is None:
        registry = build_tools(state)
    else:
        rag_service = BM25RagService.from_directory(knowledge_directory)
        registry = build_tools(state, rag_service)
    if args.trace:
        answer = run_agent(
            args.message,
            registry,
            trace_fn=lambda event: print(event, file=sys.stderr),
        )
    else:
        answer = run_agent(args.message, registry)
    print(answer)
    return 0
