"""CLI stats command implementation."""

from __future__ import annotations

import json
import sys
from argparse import Namespace

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def run_stats(args: Namespace) -> int:
    """
    Run the stats command.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code.
    """
    if getattr(args, 'local', False):
        return _run_local_stats(args)

    if not HAS_HTTPX:
        print("Error: httpx is required for stats command", file=sys.stderr)
        print("Install with: pip install httpx", file=sys.stderr)
        return 1

    try:
        response = httpx.get(f"{args.server}/v1/stats", timeout=10.0)
        response.raise_for_status()
        stats = response.json()
    except httpx.ConnectError:
        print(f"Error: Could not connect to {args.server}", file=sys.stderr)
        print("Is the RouteSmith server running?", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"Error fetching stats: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_stats_table(stats)

    return 0


def _run_local_stats(args: Namespace) -> int:
    """Run stats from local SQLite storage."""
    import time

    db_path = args.db or "routesmith_feedback.db"
    project = getattr(args, "project", "") or None

    def _fetch_local_stats() -> dict:
        from routesmith.feedback.storage import FeedbackStorage
        storage = FeedbackStorage(db_path)
        records = storage.get_all_records(limit=10000, project_id=project)
        total_cost = 0.0
        request_count = len(records)
        by_model: dict[str, int] = {}
        for r in records:
            total_cost += float(r.get("estimated_cost_usd", 0) or 0)
            model = r.get("model_id", "unknown")
            by_model[model] = by_model.get(model, 0) + 1

        result: dict = {
            "request_count": request_count,
            "total_cost_usd": round(total_cost, 6),
            "registered_models": len(by_model),
            "feedback_samples": request_count,
        }
        if project:
            result["project"] = project
            result["title"] = f"RouteSmith Cost Report (project: {project})"
        else:
            result["project"] = "local"
            # Include per-project breakdown when not filtering
            proj_stats = storage.get_project_stats()
            if proj_stats:
                result["projects"] = proj_stats
        return result

    while True:
        stats = _fetch_local_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print_stats_table(stats)
        if not args.watch:
            break
        time.sleep(2)
        print("\033[2J\033[H")  # clear screen

    return 0


def print_stats_table(stats: dict) -> None:
    """
    Print stats as a formatted table.

    Args:
        stats: Stats dictionary from RouteSmith server.
    """
    print()
    title = stats.get("title", "RouteSmith Cost Report")
    title_line = f"\u2502  {title:47s}\u2502"
    print("\u256d\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256e")
    print(title_line)
    print("\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")

    request_count = stats.get("request_count", 0)
    total_cost = stats.get("total_cost_usd", 0)
    without_routing = stats.get("estimated_without_routing", 0)
    savings = stats.get("cost_savings_usd", 0)
    savings_pct = stats.get("savings_percent", 0)
    models = stats.get("registered_models", 0)
    samples = stats.get("feedback_samples", 0)

    print(f"\u2502  Requests:           {request_count:>15,}  \u2502")
    print(f"\u2502  Actual Cost:        ${total_cost:>14,.4f}  \u2502")
    if without_routing:
        print(f"\u2502  Without Routing:    ${without_routing:>14,.4f}  \u2502")
    if savings:
        print(f"\u2502  You Saved:          ${savings:>10,.4f} ({savings_pct:>4.1f}%)  \u2502")
    print("\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    print(f"\u2502  Registered Models:  {models:>15}  \u2502")
    print(f"\u2502  Feedback Samples:   {samples:>15}  \u2502")
    print("\u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256f")
    print()

    # Per-project breakdown if available
    if "projects" in stats and stats["projects"]:
        print("Per-project breakdown:")
        print(f"  {'Project':<20} {'Requests':<12} {'Avg Latency':<14} {'Avg Quality':<14}")
        print(f"  {'-'*20} {'-'*12} {'-'*14} {'-'*14}")
        for proj, pstats in stats["projects"].items():
            lat = pstats.get("avg_latency_ms", 0)
            qual = pstats.get("avg_quality", 0)
            qual_str = f"{qual:.4f}" if qual else "N/A"
            print(f"  {proj:<20} {pstats['request_count']:<12} {lat:>8.1f}ms{'':>5} {qual_str:<14}")
        print()

    # Show last routing if available
    if "last_routing" in stats:
        last = stats["last_routing"]
        print("Last routing decision:")
        print(f"  Model: {last.get('model_selected', 'N/A')}")
        print(f"  Reason: {last.get('routing_reason', 'N/A')}")
        print(f"  Cost: ${last.get('estimated_cost_usd', 0):.6f}")
        print(f"  Saved: ${last.get('cost_savings_usd', 0):.6f}")
        print()
