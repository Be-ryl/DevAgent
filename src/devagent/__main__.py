"""Module entry point for the DevAgent CLI."""

from collections.abc import Sequence

from devagent.cli.app import main as cli_main


def main(argv: Sequence[str] | None = None) -> int:
    """Forward module execution to the command-line application."""
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

