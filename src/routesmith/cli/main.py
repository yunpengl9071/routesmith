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

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark RL-based routing against RouteLLM-SW and baselines",
        description=(
            "Run the CB-RouteSmith vs RouteLLM-SW benchmark.\n\n"
            "Uses synthetic MMLU-calibrated data by default. For authoritative\n"
            "APGR results, set ROUTELLM_DATA_DIR to a cloned RouteLLM repository\n"
            "with pre-computed evaluation responses. See\n"
            "benchmarks/RERUN_WITH_REAL_DATA.md for detailed instructions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bench_parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu", "gsm8k"],
        choices=["mmlu", "gsm8k", "mtbench"],
        help="Benchmarks to run (default: mmlu gsm8k)",
    )
    bench_parser.add_argument(
        "--fast",
        action="store_true",
        help="Use 3 seeds for faster results (default: 10 seeds)",
    )
    bench_parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of random seeds",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    bench_parser.add_argument(
        "--routellm-data",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Path to cloned RouteLLM repo with pre-computed eval data "
            "(overrides ROUTELLM_DATA_DIR env var)"
        ),
    )

    # version command
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        from routesmith.cli.serve import run_serve
        return run_serve(args)
    elif args.command == "stats":
        from routesmith.cli.stats import run_stats
        return run_stats(args)
    elif args.command == "benchmark":
        return _run_benchmark(args)
    else:
        parser.print_help()
        return 0


def _run_benchmark(args) -> int:
    """Run the CB-RouteSmith vs RouteLLM-SW benchmark."""
    import os
    import sys

    # Locate the benchmarks module relative to the routesmith package
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    repo_root = os.path.dirname(pkg_root)

    # Try src layout first, then repo root
    for search_root in [repo_root, pkg_root]:
        bench_dir = os.path.join(search_root, "benchmarks")
        if os.path.isdir(bench_dir):
            if search_root not in sys.path:
                sys.path.insert(0, search_root)
            break
    else:
        print("ERROR: Could not locate benchmarks/ directory.", file=sys.stderr)
        print("Run 'python benchmarks/run_benchmark.py' from the repo root instead.", file=sys.stderr)
        return 1

    # Set ROUTELLM_DATA_DIR if --routellm-data was provided
    if args.routellm_data:
        os.environ["ROUTELLM_DATA_DIR"] = args.routellm_data

    # Build argv for the benchmark script
    bench_argv = []
    bench_argv += ["--benchmarks"] + args.benchmarks
    if args.fast:
        bench_argv.append("--fast")
    if args.seeds is not None:
        bench_argv += ["--seeds", str(args.seeds)]
    if args.output:
        bench_argv += ["--output", args.output]

    # Invoke the benchmark main function
    import sys as _sys
    old_argv = _sys.argv
    try:
        _sys.argv = ["routesmith benchmark"] + bench_argv
        from benchmarks.run_benchmark import main as bench_main
        bench_main()
        return 0
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except Exception as e:
        print(f"Benchmark error: {e}", file=sys.stderr)
        return 1
    finally:
        _sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
