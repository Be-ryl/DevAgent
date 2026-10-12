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


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


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
    parser.add_argument(
        "--retrieval-mode",
        choices=("bm25", "dense"),
        default="bm25",
        help="knowledge retrieval implementation (default: bm25)",
    )
    parser.add_argument(
        "--dense-index-dir",
        type=Path,
        help="local Dense index directory (required for dense mode)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="embedding model expected by the Dense index",
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
    if args.retrieval_mode == "dense":
        if args.dense_index_dir is None:
            print(
                "Error: --dense-index-dir is required for dense retrieval mode.",
                file=sys.stderr,
            )
            return 2
        try:
            from devagent.rag.dense_index import DenseIndexError, DenseRagService

            rag_service = DenseRagService.from_index(
                args.dense_index_dir,
                model_name=args.embedding_model,
                knowledge_directory=knowledge_directory,
            )
        except ImportError:
            print(
                "Error: Dense Retrieval dependencies are missing. "
                "Install with `pip install -e '.[dense]'`.",
                file=sys.stderr,
            )
            return 2
        except DenseIndexError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        registry = build_tools(state, rag_service)
    elif knowledge_directory is None:
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
