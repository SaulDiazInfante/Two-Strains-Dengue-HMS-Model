"""Legacy script entry point that forwards to the package CLI."""

import sys

try:
    from .cli import run_cli
except ImportError:
    from cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(["search", *sys.argv[1:]]))
