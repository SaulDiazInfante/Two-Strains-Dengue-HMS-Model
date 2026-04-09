"""Legacy script entry point that forwards to the package CLI."""

import sys

try:
    from .cli import main
except ImportError:
    from cli import main


if __name__ == "__main__":
    raise SystemExit(main(["search", *sys.argv[1:]]))
