"""RouteSmith CLI entry point."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        prog="routesmith",
        description="RouteSmith - Adaptive LLM Execution Engine",
        epilog="Use 'routesmith <command> --help' for help on specific commands.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Interactive setup: browse OpenRouter catalog and generate routesmith.yaml",
        description=(
            "Fetches the OpenRouter model catalog with live pricing, "
            "lets you select 3–10 models, and writes routesmith.yaml."
        ),
    )
    init_parser.add_argument(
        "--output", "-o",
        type=str,
        default="routesmith.yaml",
        help="Output config file path (default: routesmith.yaml)",
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing config file",
    )

    # serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the RouteSmith proxy server",
        description="Start an OpenAI-compatible proxy server that routes requests through RouteSmith.",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=9119,
        help="Port to listen on (default: 9119)",
    )
    serve_parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--config", "-c",
        type=str,
        default="routesmith.yaml",
        help="Config file path (default: routesmith.yaml)",
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # openclaw-config command
    openclaw_parser = subparsers.add_parser(
        "openclaw-config",
        help="Generate OpenClaw provider config for RouteSmith proxy",
        description="Output a provider config snippet for adding RouteSmith to OpenClaw.",
    )
    openclaw_parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:9119",
        help="RouteSmith proxy URL (default: http://localhost:9119)",
    )
    openclaw_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Write config to file instead of stdout",
    )

    # stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show cost savings statistics",
        description="Fetch and display statistics from a running RouteSmith server.",
    )
    stats_parser.add_argument(
        "--server", "-s",
        type=str,
        default="http://127.0.0.1:9119",
        help="RouteSmith server URL (default: http://127.0.0.1:9119)",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted table",
    )

    # version command
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args(argv)

    if args.command == "init":
        from routesmith.cli.init import run_init
        return run_init(args)
    elif args.command == "serve":
        from routesmith.cli.serve import run_serve
        return run_serve(args)
    elif args.command == "stats":
        from routesmith.cli.stats import run_stats
        return run_stats(args)
    elif args.command == "openclaw-config":
        from routesmith.cli.openclaw import run_openclaw_config
        return run_openclaw_config(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
