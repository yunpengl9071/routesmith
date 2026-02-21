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


def print_stats_table(stats: dict) -> None:
    """
    Print stats as a formatted table.

    Args:
        stats: Stats dictionary from RouteSmith server.
    """
    print()
    print("\u256d\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256e")
    print("\u2502         RouteSmith Cost Report              \u2502")
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
    print(f"\u2502  Without Routing:    ${without_routing:>14,.4f}  \u2502")
    print(f"\u2502  You Saved:          ${savings:>10,.4f} ({savings_pct:>4.1f}%)  \u2502")
    print("\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524")
    print(f"\u2502  Registered Models:  {models:>15}  \u2502")
    print(f"\u2502  Feedback Samples:   {samples:>15}  \u2502")
    print("\u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256f")
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
