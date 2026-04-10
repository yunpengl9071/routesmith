"""YAML configuration file loader.

Supports two model specification styles:

  # Style A: full spec (prices embedded)
  models:
    - id: openai/gpt-4o
      cost_per_1k_input: 2.50
      cost_per_1k_output: 10.00

  # Style B: IDs only — prices fetched live from OpenRouter
  openrouter_models:
    - openai/gpt-4o
    - openai/gpt-4o-mini
    - anthropic/claude-3-5-sonnet

Both styles may coexist; openrouter_models is merged in after models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from routesmith.config import PredictorConfig, RouteSmithConfig, RoutingStrategy


def load_config_file(path: Path) -> tuple[RouteSmithConfig, list[dict[str, Any]]]:
    """Load RouteSmith configuration from YAML file.

    Returns:
        Tuple of (RouteSmithConfig, list of model registration dicts).
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        return RouteSmithConfig(), []

    # ── Routing / predictor config ────────────────────────────────────────────
    routing_cfg = data.get("routing", {})
    predictor_type = routing_cfg.get("predictor", "lints")

    predictor = PredictorConfig(
        lints_v_sq=routing_cfg.get("lints_v_sq", 1.0),
        linucb_alpha=routing_cfg.get("linucb_alpha", 1.5),
        linucb_cost_lambda=routing_cfg.get("linucb_cost_lambda", 0.15),
        seed=routing_cfg.get("seed", 42),
    )

    config = RouteSmithConfig(
        default_strategy=_parse_strategy(routing_cfg.get("strategy", "direct")),
        fallback_model=routing_cfg.get("fallback_model"),
        predictor_type=predictor_type,
        predictor=predictor,
    )

    # ── Budget ────────────────────────────────────────────────────────────────
    if "budget" in data:
        b = data["budget"]
        config = config.with_budget(
            max_cost_per_request=b.get("max_cost_per_request"),
            max_cost_per_minute=b.get("max_cost_per_minute"),
            max_cost_per_hour=b.get("max_cost_per_hour"),
            max_cost_per_day=b.get("max_cost_per_day"),
            quality_threshold=b.get("quality_threshold", 0.8),
        )

    # ── Cache ─────────────────────────────────────────────────────────────────
    if "cache" in data:
        c = data["cache"]
        config = config.with_cache(
            enabled=c.get("enabled", False),
            similarity_threshold=c.get("similarity_threshold", 0.95),
            ttl_seconds=c.get("ttl_seconds", 3600),
        )

    # ── Feedback ──────────────────────────────────────────────────────────────
    if "feedback" in data:
        fb = data["feedback"]
        config.feedback_enabled = fb.get("enabled", True)
        if "sample_rate" in fb:
            config.feedback_sample_rate = fb["sample_rate"]

    # ── Models: Style A — full spec ───────────────────────────────────────────
    models: list[dict[str, Any]] = []
    for m in data.get("models", []):
        models.append(_parse_model_entry(m))

    # ── Models: Style B — IDs only, fetch prices from OpenRouter ─────────────
    or_ids: list[str] = data.get("openrouter_models", [])
    if or_ids:
        models.extend(_fetch_openrouter_models(or_ids))

    return config, models


def _parse_model_entry(m: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "model_id": m["id"],
        "cost_per_1k_input": m["cost_per_1k_input"],
        "cost_per_1k_output": m["cost_per_1k_output"],
        "quality_score": m.get("quality_score", 0.8),
    }
    for key in ("latency_p50_ms", "latency_p99_ms", "context_window",
                "supports_vision", "supports_function_calling"):
        if key in m:
            entry[key] = m[key]
    return entry


def _fetch_openrouter_models(model_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch live pricing from OpenRouter for a list of model IDs."""
    print(f"Fetching prices from OpenRouter for {len(model_ids)} models...",
          file=sys.stderr)
    try:
        from routesmith.registry.openrouter import fetch_pricing
        pricing = fetch_pricing(model_ids)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch pricing from OpenRouter: {exc}\n"
            "Use the 'models:' key with explicit prices, or run "
            "'routesmith init' to generate a config with embedded prices."
        ) from exc

    results = []
    for mid in model_ids:
        if mid not in pricing:
            print(
                f"  WARNING: '{mid}' not found on OpenRouter — skipping.",
                file=sys.stderr,
            )
            continue
        p = pricing[mid]
        results.append({
            "model_id": p.id,
            "cost_per_1k_input": p.cost_per_1k_input,
            "cost_per_1k_output": p.cost_per_1k_output,
            "quality_score": p.quality_score,
            "context_window": p.context_window,
            "supports_vision": p.supports_vision,
            "supports_function_calling": p.supports_function_calling,
        })
        print(
            f"  {p.id}: ${p.cost_per_1k_input}/1k in, "
            f"${p.cost_per_1k_output}/1k out",
            file=sys.stderr,
        )
    return results


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
