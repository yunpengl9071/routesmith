"""Custom reward function support for RouteSmith routing feedback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, TypedDict

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


class RewardContext(TypedDict):
    """Context dict passed to custom reward functions."""

    quality: float          # User-provided quality score 0-1
    cost_usd: float         # Actual USD cost of the completion
    cost_normalized: float  # cost_per_1k_total / max model cost (0-1)
    latency_ms: float       # Wall-clock time for the completion call
    tokens_in: int          # Prompt token count
    tokens_out: int         # Completion token count
    model_id: str           # Model that was selected


def build_reward_context(
    model_id: str,
    quality: float,
    response: Any,
    latency_ms: float,
    registry: "ModelRegistry",
) -> RewardContext:
    """Build a RewardContext from a completion response and registry data."""
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0

    if hasattr(response, "usage") and response.usage is not None:
        tokens_in = getattr(response.usage, "prompt_tokens", None) or 0
        tokens_out = getattr(response.usage, "completion_tokens", None) or 0

    model_config = registry.get(model_id)
    if model_config and (tokens_in or tokens_out):
        cost_usd = (
            (tokens_in / 1000) * model_config.cost_per_1k_input
            + (tokens_out / 1000) * model_config.cost_per_1k_output
        )

    max_cost = max(
        (m.cost_per_1k_total for m in registry.list_models()),
        default=1.0,
    )
    cost_normalized = 0.0
    if model_config and max_cost > 0:
        cost_normalized = model_config.cost_per_1k_total / max_cost

    return RewardContext(
        quality=quality,
        cost_usd=cost_usd,
        cost_normalized=cost_normalized,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        model_id=model_id,
    )


def compile_reward_fn(expr: str) -> Callable[[RewardContext], float]:
    """Compile a reward expression string into a callable.

    Uses simpleeval (restricted expression evaluator) - no arbitrary code execution.
    Available variables: quality, cost_usd, cost_normalized, latency_ms,
    tokens_in, tokens_out, model_id.

    Raises:
        ImportError: If simpleeval is not installed.
        ValueError: If the expression has a syntax error or disallowed operation.
    """
    try:
        from simpleeval import EvalWithCompoundTypes, FeatureNotAvailable
    except ImportError as e:
        raise ImportError(
            "simpleeval is required for reward expressions. "
            "Install with: pip install simpleeval"
        ) from e

    # Safe built-in functions allowed in reward expressions.
    _safe_functions = {"min": min, "max": max, "abs": abs, "round": round}

    # Validate with dummy context at compile time (fail fast).
    _dummy: dict[str, Any] = {
        "quality": 0.8, "cost_usd": 0.001, "cost_normalized": 0.5,
        "latency_ms": 200.0, "tokens_in": 100, "tokens_out": 50,
        "model_id": "test/model",
    }
    _evaluator = EvalWithCompoundTypes(names=_dummy, functions=_safe_functions)
    try:
        _evaluator.eval(expr)
    except SyntaxError as e:
        raise ValueError(f"Invalid reward expression syntax: {e}") from e
    except FeatureNotAvailable as e:
        raise ValueError(f"Disallowed operation in reward expression: {e}") from e
    except Exception as e:
        raise ValueError(f"Could not evaluate reward expression with dummy values: {e}") from e

    def _fn(ctx: RewardContext) -> float:
        ev = EvalWithCompoundTypes(names=dict(ctx), functions=_safe_functions)
        try:
            result = ev.eval(expr)
        except FeatureNotAvailable as exc:
            raise ValueError(f"Disallowed operation in reward expression: {exc}") from exc
        return float(result)

    return _fn
