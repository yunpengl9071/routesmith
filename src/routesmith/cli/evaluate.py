"""CLI command: routesmith evaluate - replay historical data through a candidate routing policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.feedback.storage import FeedbackStorage


def run_evaluate(args: argparse.Namespace) -> int:
    """Replay historical feedback through a candidate routing config and report differences."""
    db_path: str = args.db
    candidate_config: str = args.config
    strategy_name: str = args.strategy or ""
    limit: int = args.limit

    storage = FeedbackStorage(db_path=db_path)
    records = storage.get_all_records(limit=limit)
    if not records:
        print(f"No feedback records found in {db_path}")
        return 1

    print(f"Loaded {len(records)} historical records from {db_path}")

    # Build the candidate RouteSmith — load config from YAML if available
    try:
        from routesmith.cli.yaml_loader import load_config_file
        loaded = load_config_file(Path(candidate_config))
        routesmith_config: RouteSmithConfig = loaded[0] if isinstance(loaded, tuple) else loaded
    except Exception:
        # Fallback: default config, override strategy
        routesmith_config = RouteSmithConfig()

    # Maybe override strategy
    if strategy_name:
        try:
            routesmith_config.default_strategy = RoutingStrategy(strategy_name)
        except ValueError:
            print(f"Unknown strategy: {strategy_name}. Valid: {', '.join(s.value for s in RoutingStrategy)}")
            return 1

    candidate = RouteSmith(config=routesmith_config)

    # Register all models found in historical records
    registered_models: set[str] = set()
    for rec in records:
        mid = rec.get("model_id", "")
        if mid and mid not in registered_models:
            candidate.register_model(mid, cost_per_1k_input=0.001, cost_per_1k_output=0.002)
            registered_models.add(mid)

    if not registered_models:
        print("No models found in historical records")
        return 1

    print(f"Registered {len(registered_models)} models from historical data")
    print(f"Strategy: {routesmith_config.default_strategy.value}\n")

    # Replay
    matches = 0
    mismatches = 0
    total_original_cost = 0.0
    total_original_quality = 0.0
    quality_samples = 0
    results: list[dict[str, Any]] = []

    for i, rec in enumerate(records):
        messages = rec.get("messages", [])
        if not messages:
            continue

        original_model = rec.get("model_id", "?")
        original_quality = rec.get("quality_score")
        metadata = rec.get("metadata", {})
        original_cost = metadata.get("estimated_cost_usd", 0.0) if isinstance(metadata, dict) else 0.0

        try:
            candidate_model = candidate.router.route(
                messages=messages,
                strategy=routesmith_config.default_strategy,
                min_quality=routesmith_config.budget.quality_threshold,
            )
        except Exception as e:
            candidate_model = f"ERROR: {e}"

        is_match = candidate_model == original_model
        if is_match:
            matches += 1
        else:
            mismatches += 1

        total_original_cost += original_cost

        if original_quality is not None:
            total_original_quality += original_quality
            quality_samples += 1

        results.append({
            "request_id": rec.get("request_id", ""),
            "original_model": original_model,
            "candidate_model": candidate_model,
            "match": is_match,
            "original_quality": original_quality,
            "original_cost": original_cost,
        })

    # Summary
    total = matches + mismatches
    agreement = matches / total * 100 if total > 0 else 0.0

    print(f"{'Metric':<45} {'Value':<15}")
    print("-" * 60)
    print(f"{'Agreement (same model selected)':<45} {agreement:>6.1f}%  ({matches}/{total})")
    print(f"{'Total original cost':<45} ${total_original_cost:>8.5f}")
    print(f"{'Avg original quality':<45} {total_original_quality / quality_samples:>8.4f}" if quality_samples > 0 else "N/A")

    print()

    if mismatches > 0:
        print(f"\nMismatches ({mismatches}):")
        print(f"{'Request ID':<20} {'Original':<25} {'Candidate':<25}")
        print("-" * 70)
        for r in results[:20]:  # Show first 20 mismatches
            if not r["match"]:
                orig = r["original_model"][:24] if r["original_model"] else "?"
                cand = str(r["candidate_model"])[:24] if r["candidate_model"] else "?"
                print(f"{r['request_id'][:18]:<20} {orig:<25} {cand:<25}")
        if mismatches > 20:
            print(f"... and {mismatches - 20} more mismatches")

    return 0


def register_subparser(subparsers) -> None:
    """Register the evaluate subcommand."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a routing policy against historical feedback data",
        description=(
            "Load historical feedback records from a SQLite database and replay them "
            "through a candidate routing configuration. Shows agreement rate, cost "
            "impact, and model selection differences. Use this to validate config "
            "changes before deploying."
        ),
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="routesmith.yaml",
        help="Candidate config file (default: routesmith.yaml)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="routesmith_feedback.db",
        help="SQLite database path with historical feedback records (default: routesmith_feedback.db)",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="",
        help="Override routing strategy (e.g., direct, cascade, parallel, speculative)",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=1000,
        help="Max records to evaluate (default: 1000)",
    )
