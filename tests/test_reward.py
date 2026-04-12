"""Tests for custom reward function support."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
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


class TestRouteSmithConfigRewardFields:
    def test_reward_fn_defaults_to_none(self):
        from routesmith import RouteSmithConfig
        assert RouteSmithConfig().reward_fn is None

    def test_reward_expr_defaults_to_none(self):
        from routesmith import RouteSmithConfig
        assert RouteSmithConfig().reward_expr is None

    def test_reward_fn_accepts_callable(self):
        from routesmith import RouteSmithConfig
        fn = lambda ctx: ctx["quality"]
        assert RouteSmithConfig(reward_fn=fn).reward_fn is fn

    def test_reward_expr_accepts_string(self):
        from routesmith import RouteSmithConfig
        config = RouteSmithConfig(reward_expr="quality - 0.15 * cost_normalized")
        assert config.reward_expr == "quality - 0.15 * cost_normalized"

    def test_with_cache_preserves_reward_fn(self):
        from routesmith import RouteSmithConfig
        fn = lambda ctx: ctx["quality"]
        assert RouteSmithConfig(reward_fn=fn).with_cache(enabled=True).reward_fn is fn

    def test_with_budget_preserves_reward_expr(self):
        from routesmith import RouteSmithConfig
        config = RouteSmithConfig(reward_expr="quality").with_budget(max_cost_per_request=0.10)
        assert config.reward_expr == "quality"


def _make_linucb(models=None):
    from routesmith.predictor.linucb import LinUCBPredictor
    return LinUCBPredictor(registry=_make_registry(models))


def _make_adaptive(models=None):
    from routesmith.predictor.learner import AdaptivePredictor
    return AdaptivePredictor(registry=_make_registry(models))


def _make_lints(models=None):
    from routesmith.predictor.lints import LinTSPredictor
    return LinTSPredictor(registry=_make_registry(models))


_MSGS = [{"role": "user", "content": "Hello"}]


class TestPredictorRewardOverride:
    def test_linucb_accepts_reward_override_param(self):
        pred = _make_linucb()
        pred.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=0.5)

    def test_linucb_reward_override_changes_b_vector(self):
        """Two LinUCB predictors updated with same quality but different
        reward_override should diverge in their b vectors."""
        pred_default = _make_linucb()
        pred_override = _make_linucb()

        pred_default.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_override.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8,
                             reward_override=0.5)

        b_default = pred_default._arms["openai/gpt-4o-mini"]["b"]
        b_override = pred_override._arms["openai/gpt-4o-mini"]["b"]
        assert not np.allclose(b_default, b_override), (
            "b vectors should differ when reward_override is used"
        )

    def test_linucb_none_override_identical_to_default(self):
        pred_a = _make_linucb()
        pred_b = _make_linucb()

        pred_a.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_b.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=None)

        np.testing.assert_array_almost_equal(
            pred_a._arms["openai/gpt-4o-mini"]["b"],
            pred_b._arms["openai/gpt-4o-mini"]["b"],
        )

    def test_lints_accepts_reward_override_param(self):
        pred = _make_lints()
        pred.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=0.5)

    def test_lints_reward_override_changes_arm_state(self):
        pred_default = _make_lints()
        pred_override = _make_lints()

        pred_default.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_override.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8,
                             reward_override=0.3)

        idx_d = pred_default._arm_index["openai/gpt-4o-mini"]
        idx_o = pred_override._arm_index["openai/gpt-4o-mini"]
        assert not np.allclose(
            pred_default._router.arms[idx_d].b,
            pred_override._router.arms[idx_o].b,
        )

    def test_lints_none_override_identical_to_default(self):
        pred_a = _make_lints()
        pred_b = _make_lints()

        pred_a.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_b.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=None)

        idx_a = pred_a._arm_index["openai/gpt-4o-mini"]
        idx_b = pred_b._arm_index["openai/gpt-4o-mini"]
        np.testing.assert_array_almost_equal(
            pred_a._router.arms[idx_a].b,
            pred_b._router.arms[idx_b].b,
        )

    def test_embedding_none_override_identical_to_default(self):
        from routesmith.predictor.embedding import EmbeddingPredictor
        pred_a = EmbeddingPredictor(model_quality_priors={"m": 0.5})
        pred_b = EmbeddingPredictor(model_quality_priors={"m": 0.5})
        pred_a.update(_MSGS, "m", actual_quality=0.8)
        pred_b.update(_MSGS, "m", actual_quality=0.8, reward_override=None)
        assert pred_a.model_quality_priors["m"] == pred_b.model_quality_priors["m"]

    def test_embedding_reward_override_changes_prior(self):
        from routesmith.predictor.embedding import EmbeddingPredictor
        pred_a = EmbeddingPredictor(model_quality_priors={"m": 0.5})
        pred_b = EmbeddingPredictor(model_quality_priors={"m": 0.5})
        pred_a.update(_MSGS, "m", actual_quality=0.8)
        pred_b.update(_MSGS, "m", actual_quality=0.8, reward_override=0.1)
        assert pred_a.model_quality_priors["m"] != pred_b.model_quality_priors["m"]

    def test_adaptive_none_override_identical_to_default(self):
        pred_a = _make_adaptive()
        pred_b = _make_adaptive()
        pred_a.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.1)  # quality != initial prior
        pred_b.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.1, reward_override=None)
        # Verify update was effective (EMA moved from initial 0.8 toward 0.1)
        assert pred_a._ema_priors["openai/gpt-4o-mini"] < 0.8
        assert pred_a._ema_priors["openai/gpt-4o-mini"] == pred_b._ema_priors["openai/gpt-4o-mini"]

    def test_adaptive_reward_override_changes_ema(self):
        pred_a = _make_adaptive()
        pred_b = _make_adaptive()
        pred_a.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_b.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=0.1)
        assert pred_a._ema_priors["openai/gpt-4o-mini"] != pred_b._ema_priors["openai/gpt-4o-mini"]


# ---------------------------------------------------------------------------
# YAML loader reward support
# ---------------------------------------------------------------------------


class TestYamlLoaderReward:
    def test_feedback_reward_string_sets_reward_expr(self):
        from routesmith.cli.yaml_loader import load_config_file

        yaml_content = (
            "routing:\n  predictor: lints\n"
            "feedback:\n  reward: \"quality - 0.15 * cost_normalized\"\n"
            "models: []\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config, _ = load_config_file(Path(path))
            assert config.reward_expr == "quality - 0.15 * cost_normalized"
        finally:
            os.unlink(path)

    def test_no_feedback_section_leaves_reward_expr_none(self):
        from routesmith.cli.yaml_loader import load_config_file

        yaml_content = "routing:\n  predictor: lints\nmodels: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config, _ = load_config_file(Path(path))
            assert config.reward_expr is None
        finally:
            os.unlink(path)

    def test_feedback_without_reward_leaves_expr_none(self):
        from routesmith.cli.yaml_loader import load_config_file

        yaml_content = (
            "routing:\n  predictor: lints\n"
            "feedback:\n  enabled: true\n  sample_rate: 0.5\n"
            "models: []\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config, _ = load_config_file(Path(path))
            assert config.reward_expr is None
        finally:
            os.unlink(path)
