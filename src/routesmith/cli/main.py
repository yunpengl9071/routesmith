"""RouteSmith CLI entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from routesmith import __version__


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
    init_parser.add_argument(
        "--provider",
        action="append",
        dest="provider",
        choices=["anthropic", "openai", "openrouter", "groq"],
        help="Force provider selection (repeatable, skips interactive picker)",
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
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Require Bearer auth on proxy requests (default: no auth)",
    )
    serve_parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as a background daemon (double-fork, pidfile, logfile)",
    )

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a command with RouteSmith proxy (starts daemon if needed)",
        description=(
            "Run a CLI command through the RouteSmith proxy. "
            "Automatically injects the correct env vars into the child process "
            "and starts the proxy as a daemon if it's not already running."
        ),
    )
    run_parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="Proxy host (default: 127.0.0.1)",
    )
    run_parser.add_argument(
        "--port", "-p",
        type=int,
        default=9119,
        help="Proxy port (default: 9119)",
    )
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Config file path (default: ./routesmith.yaml → ~/.routesmith/routesmith.yaml)",
    )
    run_parser.add_argument(
        "--family",
        type=str,
        choices=["anthropic", "openai"],
        default=None,
        help="Force env var family (default: auto-detect from command basename)",
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run and its arguments",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show RouteSmith proxy status",
        description="Check if the RouteSmith proxy daemon is running and show summary info.",
    )
    status_parser.add_argument(
        "--host", "-H",
        type=str,
        default="127.0.0.1",
        help="Proxy host (default: 127.0.0.1)",
    )
    status_parser.add_argument(
        "--port", "-p",
        type=int,
        default=9119,
        help="Proxy port (default: 9119)",
    )

    # down command
    subparsers.add_parser(
        "down",
        help="Stop the RouteSmith proxy daemon",
        description="Send SIGTERM to the RouteSmith proxy daemon and clean up the pidfile.",
    )

    # quickstart command
    quickstart_parser = subparsers.add_parser(
        "quickstart",
        help="Detect provider, generate config, start server in one command",
        description="Detect API keys, generate routesmith.yaml, show connection snippets.",
    )
    quickstart_parser.add_argument(
        "--port", "-p",
        type=int,
        default=9119,
        help="Port for proxy server (default: 9119)",
    )
    quickstart_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Overwrite existing config without prompting",
    )
    quickstart_parser.add_argument(
        "--provider",
        action="append",
        dest="provider",
        choices=["anthropic", "openai", "openrouter", "groq"],
        help="Force provider selection (repeatable, overrides env detection)",
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
        description="Fetch and display statistics from a running RouteSmith server or local storage.",
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
    stats_parser.add_argument(
        "--local",
        action="store_true",
        help="Read stats from local SQLite storage instead of server",
    )
    stats_parser.add_argument(
        "--db",
        type=str,
        help="SQLite database path (with --local, default: routesmith_feedback.db)",
    )
    stats_parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Live refresh stats every 2 seconds (--local only)",
    )
    stats_parser.add_argument(
        "--project", "--proj",
        type=str,
        default="",
        help="Filter stats to a specific project name",
    )

    # audit command
    audit_parser = subparsers.add_parser(
        "audit",
        help="View routing decision audit log",
        description="Retrieve and display the structured audit log of routing decisions.",
    )
    audit_parser.add_argument(
        "--db",
        type=str,
        default="routesmith_feedback.db",
        help="SQLite database path (default: routesmith_feedback.db)",
    )
    audit_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=50,
        help="Number of audit entries to show (default: 50)",
    )
    audit_parser.add_argument(
        "--project", "--proj",
        type=str,
        default="",
        help="Filter by project name",
    )
    audit_parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Filter by selected model",
    )
    audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # roles command
    roles_parser = subparsers.add_parser(
        "roles",
        help="Manage per-role routing policies",
        description="Configure routing policies (reward functions, model pools) per agent role.",
    )
    roles_parser.add_argument(
        "action",
        nargs="?",
        choices=["list", "set", "unset"],
        default="list",
        help="Action: list (default), set, or unset a policy",
    )
    roles_parser.add_argument(
        "--role", "-r",
        type=str,
        default="",
        help="Agent role name",
    )
    roles_parser.add_argument(
        "--config",
        type=str,
        default="routesmith.yaml",
        help="Routesmith config file path (default: routesmith.yaml)",
    )
    roles_parser.add_argument(
        "--model-pool",
        type=str,
        nargs="*",
        help="List of model IDs for this role (with set action)",
    )
    roles_parser.add_argument(
        "--reward",
        type=str,
        nargs="*",
        help="Reward function names for this role (with set action)",
    )
    roles_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch interactive TUI dashboard",
        description="Launch a live-refreshing terminal dashboard showing RouteSmith stats.",
    )
    dashboard_parser.add_argument(
        "--db",
        type=str,
        default="routesmith_feedback.db",
        help="SQLite database path (default: routesmith_feedback.db)",
    )

    # evaluate command
    from routesmith.cli.evaluate import register_subparser as register_evaluate
    register_evaluate(subparsers)

    # models command
    models_parser = subparsers.add_parser(
        "models",
        help="List and refresh the model pool",
        description="List registered models from config or refresh from provider catalogs.",
    )
    models_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Config file path (default: ./routesmith.yaml → ~/.routesmith/routesmith.yaml)",
    )
    models_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-detect providers and rebuild model pool from catalogs",
    )
    models_parser.add_argument(
        "--provider",
        action="append",
        dest="provider",
        choices=["anthropic", "openai", "openrouter", "groq"],
        help="Force provider(s) for --refresh (repeatable, overrides env detection)",
    )
    models_parser.add_argument(
        "--json",
        action="store_true",
        help="Output model list as JSON",
    )

    # `models list` / `models refresh` subcommands (documented, primary form).
    # The flags above remain valid directly on `models` for back-compat.
    models_subparsers = models_parser.add_subparsers(dest="models_command")

    models_list_parser = models_subparsers.add_parser(
        "list",
        help="List registered models from config",
    )
    models_list_parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Config file path (default: ./routesmith.yaml → ~/.routesmith/routesmith.yaml)",
    )
    models_list_parser.add_argument(
        "--json", action="store_true", help="Output model list as JSON",
    )
    models_list_parser.set_defaults(refresh=False)

    models_refresh_parser = models_subparsers.add_parser(
        "refresh",
        help="Re-detect providers and rebuild model pool from catalogs",
    )
    models_refresh_parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Config file path (default: ./routesmith.yaml → ~/.routesmith/routesmith.yaml)",
    )
    models_refresh_parser.add_argument(
        "--provider", action="append", dest="provider",
        choices=["anthropic", "openai", "openrouter", "groq"],
        help="Force provider(s) (repeatable, overrides env detection)",
    )
    models_refresh_parser.add_argument(
        "--json", action="store_true", help="Output model list as JSON",
    )
    models_refresh_parser.set_defaults(refresh=True)

    # connect command
    connect_parser = subparsers.add_parser(
        "connect",
        help="Generate tool-specific config and verify proxy setup",
        description="Emit copy-pasteable configuration for connecting an AI coding tool to RouteSmith.",
    )
    connect_parser.add_argument(
        "tool",
        type=str,
        help="Tool name: claude-code, codex, opencode, openclaw, pi, hermes, openai-sdk, anthropic-sdk",
    )
    connect_parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:9119",
        help="RouteSmith proxy URL (default: http://localhost:9119)",
    )
    connect_parser.add_argument(
        "--apply",
        action="store_true",
        help="Write config file for supported tools",
    )
    connect_parser.add_argument(
        "--verify",
        action="store_true",
        help="Live end-to-end verification against the proxy",
    )
    connect_parser.add_argument(
        "--yes",
        action="store_true",
        help="Overwrite existing config files without prompting",
    )
    connect_parser.add_argument(
        "--proxy-api-key",
        type=str,
        default="",
        help="API key for proxy authentication",
    )

    # version command
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    if args.command == "connect":
        from routesmith.cli.connect import run_connect
        return run_connect(args)
    elif args.command == "init":
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
    elif args.command == "models":
        from routesmith.cli.models import run_models
        return run_models(args)
    elif args.command == "quickstart":
        from routesmith.cli.quickstart import run_quickstart
        return run_quickstart(args)
    elif args.command == "audit":
        from routesmith.cli.audit import run_audit
        return run_audit(args)
    elif args.command == "roles":
        from routesmith.cli.roles import run_roles
        return run_roles(args)
    elif args.command == "dashboard":
        from routesmith.cli_dashboard import run_dashboard
        return run_dashboard(db_path=args.db)
    elif args.command == "evaluate":
        from routesmith.cli.evaluate import run_evaluate
        return run_evaluate(args)
    elif args.command == "run":
        from routesmith.cli.run import run_run
        return run_run(args)
    elif args.command == "status":
        from routesmith.cli.run import run_status
        return run_status(args)
    elif args.command == "down":
        from routesmith.cli.run import run_down
        return run_down(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
