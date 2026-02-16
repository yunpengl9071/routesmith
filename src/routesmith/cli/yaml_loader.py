"""YAML configuration file loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from routesmith.config import RouteSmithConfig, RoutingStrategy


def load_config_file(path: Path) -> tuple[RouteSmithConfig, list[dict[str, Any]]]:
    """
    Load RouteSmith configuration from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Tuple of (RouteSmithConfig, list of model registration dicts).

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        return RouteSmithConfig(), []

    # Parse routing config
    routing_config = data.get("routing", {})
    config = RouteSmithConfig(
        default_strategy=_parse_strategy(routing_config.get("strategy", "direct")),
        fallback_model=routing_config.get("fallback_model"),
    )

    # Parse budget config
    if "budget" in data:
        budget = data["budget"]
        config = config.with_budget(
            max_cost_per_request=budget.get("max_cost_per_request"),
            max_cost_per_minute=budget.get("max_cost_per_minute"),
            max_cost_per_hour=budget.get("max_cost_per_hour"),
            max_cost_per_day=budget.get("max_cost_per_day"),
            quality_threshold=budget.get("quality_threshold", 0.8),
        )

    # Parse cache config
    if "cache" in data:
        cache = data["cache"]
        config = config.with_cache(
            enabled=cache.get("enabled", False),
            similarity_threshold=cache.get("similarity_threshold", 0.95),
            ttl_seconds=cache.get("ttl_seconds", 3600),
        )

    # Parse feedback config
    if "feedback" in data:
        feedback = data["feedback"]
        config.feedback_enabled = feedback.get("enabled", True)
        if "sample_rate" in feedback:
            config.feedback_sample_rate = feedback["sample_rate"]

    # Parse models
    models: list[dict[str, Any]] = []
    for model_data in data.get("models", []):
        model: dict[str, Any] = {
            "model_id": model_data["id"],
            "cost_per_1k_input": model_data["cost_per_1k_input"],
            "cost_per_1k_output": model_data["cost_per_1k_output"],
            "quality_score": model_data.get("quality_score", 0.8),
        }
        # Optional fields
        if "latency_p50_ms" in model_data:
            model["latency_p50_ms"] = model_data["latency_p50_ms"]
        if "latency_p99_ms" in model_data:
            model["latency_p99_ms"] = model_data["latency_p99_ms"]
        if "context_window" in model_data:
            model["context_window"] = model_data["context_window"]
        models.append(model)

    return config, models


def _parse_strategy(strategy: str) -> RoutingStrategy:
    """
    Parse strategy string to enum.

    Args:
        strategy: Strategy name string.

    Returns:
        RoutingStrategy enum value.
    """
    mapping = {
        "direct": RoutingStrategy.DIRECT,
        "cascade": RoutingStrategy.CASCADE,
        "parallel": RoutingStrategy.PARALLEL,
        "speculative": RoutingStrategy.SPECULATIVE,
    }
    return mapping.get(strategy.lower(), RoutingStrategy.DIRECT)
