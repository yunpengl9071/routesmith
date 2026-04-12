"""Tests for custom reward function support."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(models=None):
    from routesmith.registry.models import ModelRegistry
    registry = ModelRegistry()
    if models is None:
        models = [
            ("openai/gpt-4o-mini", 0.15, 0.60),
            ("openai/gpt-4o", 2.50, 10.0),
        ]
    for mid, inp, out in models:
        registry.register(mid, cost_per_1k_input=inp, cost_per_1k_output=out, quality_score=0.8)
    return registry


def _make_response(prompt_tokens=100, completion_tokens=50):
    r = MagicMock()
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    return r


# ---------------------------------------------------------------------------
# build_reward_context
# ---------------------------------------------------------------------------

class TestBuildRewardContext:
    def test_quality_passed_through(self):
        from routesmith.feedback.reward import build_reward_context
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.9,
            response=_make_response(), latency_ms=200.0,
            registry=_make_registry(),
        )
        assert ctx["quality"] == 0.9

    def test_model_id_passed_through(self):
        from routesmith.feedback.reward import build_reward_context
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=_make_response(), latency_ms=100.0,
            registry=_make_registry(),
        )
        assert ctx["model_id"] == "openai/gpt-4o-mini"

    def test_latency_ms_passed_through(self):
        from routesmith.feedback.reward import build_reward_context
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=_make_response(), latency_ms=333.0,
            registry=_make_registry(),
        )
        assert ctx["latency_ms"] == 333.0

    def test_token_counts_from_usage(self):
        from routesmith.feedback.reward import build_reward_context
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=_make_response(prompt_tokens=200, completion_tokens=75),
            latency_ms=100.0, registry=_make_registry(),
        )
        assert ctx["tokens_in"] == 200
        assert ctx["tokens_out"] == 75

    def test_cost_usd_computed_from_tokens(self):
        from routesmith.feedback.reward import build_reward_context
        # gpt-4o-mini: $0.15/1k in, $0.60/1k out
        # 100 in + 50 out -> 0.1*0.15 + 0.05*0.60 = 0.015 + 0.030 = 0.045
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=_make_response(prompt_tokens=100, completion_tokens=50),
            latency_ms=100.0, registry=_make_registry(),
        )
        assert abs(ctx["cost_usd"] - 0.045) < 1e-9

    def test_cost_normalized_is_zero_to_one(self):
        from routesmith.feedback.reward import build_reward_context
        registry = _make_registry()
        ctx_cheap = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=_make_response(), latency_ms=100.0, registry=registry,
        )
        ctx_expensive = build_reward_context(
            model_id="openai/gpt-4o", quality=0.8,
            response=_make_response(), latency_ms=100.0, registry=registry,
        )
        assert 0.0 <= ctx_cheap["cost_normalized"] <= 1.0
        assert abs(ctx_expensive["cost_normalized"] - 1.0) < 1e-9
        assert ctx_cheap["cost_normalized"] < ctx_expensive["cost_normalized"]

    def test_no_usage_gives_zero_tokens_and_cost(self):
        from routesmith.feedback.reward import build_reward_context
        response = MagicMock()
        response.usage = None
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=response, latency_ms=100.0, registry=_make_registry(),
        )
        assert ctx["tokens_in"] == 0
        assert ctx["tokens_out"] == 0
        assert ctx["cost_usd"] == 0.0

    def test_single_model_registry_cost_normalized_is_one(self):
        from routesmith.feedback.reward import build_reward_context
        from routesmith.registry.models import ModelRegistry
        registry = ModelRegistry()
        registry.register("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                           cost_per_1k_output=0.60, quality_score=0.8)
        ctx = build_reward_context(
            model_id="openai/gpt-4o-mini", quality=0.8,
            response=_make_response(), latency_ms=100.0, registry=registry,
        )
        assert abs(ctx["cost_normalized"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compile_reward_fn
# ---------------------------------------------------------------------------

def _dummy_ctx():
    from routesmith.feedback.reward import RewardContext
    return RewardContext(
        quality=0.8, cost_usd=0.001, cost_normalized=0.5,
        latency_ms=200.0, tokens_in=100, tokens_out=50,
        model_id="openai/gpt-4o-mini",
    )


class TestCompileRewardFn:
    def test_returns_callable(self):
        from routesmith.feedback.reward import compile_reward_fn
        fn = compile_reward_fn("quality - 0.15 * cost_normalized")
        assert callable(fn)

    def test_simple_expression_evaluates_correctly(self):
        from routesmith.feedback.reward import compile_reward_fn
        fn = compile_reward_fn("quality - 0.15 * cost_normalized")
        ctx = _dummy_ctx()
        result = fn(ctx)
        assert abs(result - (0.8 - 0.15 * 0.5)) < 1e-9

    def test_latency_expression(self):
        from routesmith.feedback.reward import compile_reward_fn
        fn = compile_reward_fn("quality - 0.001 * latency_ms")
        ctx = _dummy_ctx()
        result = fn(ctx)
        assert abs(result - (0.8 - 0.001 * 200.0)) < 1e-9

    def test_quality_only_expression(self):
        from routesmith.feedback.reward import compile_reward_fn
        fn = compile_reward_fn("quality")
        from routesmith.feedback.reward import RewardContext
        ctx = RewardContext(
            quality=0.75, cost_usd=0.0, cost_normalized=0.0,
            latency_ms=0.0, tokens_in=0, tokens_out=0, model_id="x",
        )
        assert abs(fn(ctx) - 0.75) < 1e-9

    def test_syntax_error_raises_value_error(self):
        from routesmith.feedback.reward import compile_reward_fn
        with pytest.raises(ValueError, match="syntax"):
            compile_reward_fn("quality *** cost_normalized")

    def test_disallowed_import_raises_value_error(self):
        from routesmith.feedback.reward import compile_reward_fn
        with pytest.raises(ValueError):
            compile_reward_fn("__import__('os').getcwd()")

    def test_returns_float(self):
        from routesmith.feedback.reward import compile_reward_fn
        fn = compile_reward_fn("quality")
        result = fn(_dummy_ctx())
        assert isinstance(result, float)

    def test_expression_with_min_builtin(self):
        from routesmith.feedback.reward import compile_reward_fn
        fn = compile_reward_fn("min(quality, 0.9)")
        from routesmith.feedback.reward import RewardContext
        ctx = RewardContext(
            quality=1.0, cost_usd=0.0, cost_normalized=0.0,
            latency_ms=0.0, tokens_in=0, tokens_out=0, model_id="x",
        )
        assert abs(fn(ctx) - 0.9) < 1e-9
