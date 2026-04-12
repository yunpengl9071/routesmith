# Custom Reward Function Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users define how the router weighs quality vs. cost vs. latency by supplying a reward expression string (YAML) or a Python callable, while keeping existing behavior unchanged when no reward function is configured.

**Architecture:** After each completion, `client.py` optionally builds a `RewardContext` dict, calls the user's `reward_fn`, and passes the resulting scalar to `predictor.update()` as `reward_override`. All four predictor implementations accept the new parameter; when `None`, each predictor's existing internal reward computation runs unchanged. The expression path uses `simpleeval` for safe evaluation (no arbitrary code execution).

**Tech Stack:** `simpleeval>=0.9.13` (new core dep), existing `linucb.py`/`lints.py`/`embedding.py`/`learner.py` predictors, existing `FeedbackRecord`, dataclass config pattern already in use.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/routesmith/feedback/reward.py` | `RewardContext` type, `build_reward_context()`, `compile_reward_fn()` |
| Modify | `src/routesmith/config.py` | Add `reward_fn` + `reward_expr` fields to `RouteSmithConfig`; propagate in `with_cache`/`with_budget` |
| Modify | `pyproject.toml` | Add `simpleeval>=0.9.13` to core deps |
| Modify | `src/routesmith/predictor/base.py` | Add `reward_override: float | None = None` to abstract `update()` |
| Modify | `src/routesmith/predictor/linucb.py` | Use `reward_override` in `update()` when provided |
| Modify | `src/routesmith/predictor/lints.py` | Use `reward_override` in `update()` when provided |
| Modify | `src/routesmith/predictor/embedding.py` | Accept `reward_override` in `update()`, apply to EMA target |
| Modify | `src/routesmith/predictor/learner.py` | Accept `reward_override` in `update()`, apply to EMA target |
| Modify | `src/routesmith/cli/yaml_loader.py` | Parse `feedback.reward` string → `reward_expr` on config |
| Modify | `src/routesmith/client.py` | Resolve `reward_fn` on init; build context + pass override in `record_outcome()` |
| Create | `tests/test_reward.py` | All reward function tests |

---

## Task 1: `reward.py` — RewardContext, build_reward_context, compile_reward_fn

**Files:**
- Create: `src/routesmith/feedback/reward.py`
- Create: `tests/test_reward.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reward.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/yliulupo/Apps/routesmith
.venv/bin/pytest tests/test_reward.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'build_reward_context' from 'routesmith.feedback.reward'`

- [ ] **Step 3: Create `src/routesmith/feedback/reward.py`**

```python
"""Custom reward function support for RouteSmith routing feedback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)

from typing import TypedDict


class RewardContext(TypedDict):
    """Context dict passed to custom reward functions."""
    quality: float
    cost_usd: float
    cost_normalized: float
    latency_ms: float
    tokens_in: int
    tokens_out: int
    model_id: str


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

    Uses simpleeval (restricted expression evaluator) — no arbitrary code execution.
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

    # Validate with dummy context at compile time (fail fast).
    _dummy: dict[str, Any] = {
        "quality": 0.8, "cost_usd": 0.001, "cost_normalized": 0.5,
        "latency_ms": 200.0, "tokens_in": 100, "tokens_out": 50,
        "model_id": "test/model",
    }
    _evaluator = EvalWithCompoundTypes(names=_dummy)
    try:
        _evaluator.eval(expr)
    except SyntaxError as e:
        raise ValueError(f"Invalid reward expression syntax: {e}") from e
    except FeatureNotAvailable as e:
        raise ValueError(f"Disallowed operation in reward expression: {e}") from e
    except Exception as e:
        raise ValueError(f"Could not evaluate reward expression with dummy values: {e}") from e

    def _fn(ctx: RewardContext) -> float:
        ev = EvalWithCompoundTypes(names=dict(ctx))
        try:
            result = ev.eval(expr)
        except FeatureNotAvailable as exc:
            raise ValueError(f"Disallowed operation in reward expression: {exc}") from exc
        return float(result)

    return _fn
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_reward.py::TestBuildRewardContext tests/test_reward.py::TestCompileRewardFn -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/feedback/reward.py tests/test_reward.py
git commit -m "feat(reward): add RewardContext, build_reward_context, compile_reward_fn"
```

---

## Task 2: Config fields + pyproject.toml

**Files:**
- Modify: `src/routesmith/config.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reward.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_reward.py::TestRouteSmithConfigRewardFields -v
```

Expected: `AttributeError: 'RouteSmithConfig' object has no attribute 'reward_fn'`

- [ ] **Step 3: Add `simpleeval` to `pyproject.toml` core deps**

Change the `dependencies` list from:

```toml
dependencies = [
    "litellm>=1.40.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
]
```

To:

```toml
dependencies = [
    "litellm>=1.40.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
    "simpleeval>=0.9.13",
]
```

- [ ] **Step 4: Install the new dependency**

```bash
uv pip install simpleeval
```

- [ ] **Step 5: Add `reward_fn` and `reward_expr` to `RouteSmithConfig`**

In `src/routesmith/config.py`, change the imports from:

```python
from typing import Any
```

To:

```python
from typing import Any, Callable
```

After the `litellm_params` field in `RouteSmithConfig`, add:

```python
    # Custom reward function for predictor updates.
    # reward_fn takes priority over reward_expr. When neither is set,
    # each predictor uses its internal default reward computation.
    reward_fn: Callable[..., float] | None = None
    reward_expr: str | None = None  # Expression string, compiled on RouteSmith init
```

- [ ] **Step 6: Propagate new fields in `with_cache` and `with_budget`**

In `with_cache`, the `return RouteSmithConfig(...)` call ends with `litellm_params=self.litellm_params,`.
Add after that line (before the closing paren):

```python
            reward_fn=self.reward_fn,
            reward_expr=self.reward_expr,
```

Do the same in `with_budget`.

- [ ] **Step 7: Run tests**

```bash
.venv/bin/pytest tests/test_reward.py::TestRouteSmithConfigRewardFields -v
.venv/bin/pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/routesmith/config.py pyproject.toml
git commit -m "feat(config): add reward_fn and reward_expr fields to RouteSmithConfig"
```

---

## Task 3: Predictor `update()` — add `reward_override` parameter

**Files:**
- Modify: `src/routesmith/predictor/base.py`
- Modify: `src/routesmith/predictor/linucb.py`
- Modify: `src/routesmith/predictor/lints.py`
- Modify: `src/routesmith/predictor/embedding.py`
- Modify: `src/routesmith/predictor/learner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reward.py`:

```python
import numpy as np

def _make_linucb(models=None):
    from routesmith.predictor.linucb import LinUCBPredictor
    return LinUCBPredictor(registry=_make_registry(models))

def _make_lints(models=None):
    from routesmith.predictor.lints import LinTSPredictor
    return LinTSPredictor(registry=_make_registry(models))

_MSGS = [{"role": "user", "content": "Hello"}]


class TestPredictorRewardOverride:
    def test_linucb_accepts_reward_override_param(self):
        pred = _make_linucb()
        pred.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=0.5)

    def test_linucb_reward_override_changes_b_vector(self):
        pred_default = _make_linucb()
        pred_override = _make_linucb()

        pred_default.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_override.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8,
                             reward_override=0.5)

        b_default = pred_default._arms["openai/gpt-4o-mini"]["b"]
        b_override = pred_override._arms["openai/gpt-4o-mini"]["b"]
        assert not np.allclose(b_default, b_override)

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
            pred_default._router._arms[idx_d].b,
            pred_override._router._arms[idx_o].b,
        )

    def test_lints_none_override_identical_to_default(self):
        pred_a = _make_lints()
        pred_b = _make_lints()

        pred_a.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8)
        pred_b.update(_MSGS, "openai/gpt-4o-mini", actual_quality=0.8, reward_override=None)

        idx_a = pred_a._arm_index["openai/gpt-4o-mini"]
        idx_b = pred_b._arm_index["openai/gpt-4o-mini"]
        np.testing.assert_array_almost_equal(
            pred_a._router._arms[idx_a].b,
            pred_b._router._arms[idx_b].b,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_reward.py::TestPredictorRewardOverride -v
```

Expected: `TypeError: update() got an unexpected keyword argument 'reward_override'`

- [ ] **Step 3: Update `src/routesmith/predictor/base.py`**

Change the abstract `update` signature from:

```python
    @abstractmethod
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
```

To:

```python
    @abstractmethod
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
    ) -> None:
```

- [ ] **Step 4: Update `src/routesmith/predictor/linucb.py`**

Change the `update` signature from:

```python
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
```

To:

```python
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
    ) -> None:
```

Change the reward computation from:

```python
        reward = actual_quality - self._cost_lambda * normalized_cost
```

To:

```python
        if reward_override is not None:
            reward = reward_override
        else:
            reward = actual_quality - self._cost_lambda * normalized_cost
```

- [ ] **Step 5: Update `src/routesmith/predictor/lints.py`**

Change `LinTSPredictor.update` signature from:

```python
    def update(
        self,
        messages: list[dict],
        model_id: str,
        actual_quality: float,
    ) -> None:
```

To:

```python
    def update(
        self,
        messages: list[dict],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
    ) -> None:
```

Change the body from:

```python
        arm_idx = self._arm_index.get(model_id)
        if arm_idx is None:
            return
        x = self._features(messages, model_id)
        self._router.update(arm=arm_idx, x=x, reward=actual_quality)
        self._total_updates += 1
```

To:

```python
        arm_idx = self._arm_index.get(model_id)
        if arm_idx is None:
            return
        x = self._features(messages, model_id)
        reward = reward_override if reward_override is not None else actual_quality
        self._router.update(arm=arm_idx, x=x, reward=reward)
        self._total_updates += 1
```

- [ ] **Step 6: Update `src/routesmith/predictor/embedding.py`**

Change the `update` signature from:

```python
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
```

To:

```python
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
    ) -> None:
```

Change the body from:

```python
        alpha = 0.1
        current = self.model_quality_priors.get(model_id, 0.5)
        self.model_quality_priors[model_id] = alpha * actual_quality + (1 - alpha) * current
```

To:

```python
        target = reward_override if reward_override is not None else actual_quality
        alpha = 0.1
        current = self.model_quality_priors.get(model_id, 0.5)
        self.model_quality_priors[model_id] = alpha * target + (1 - alpha) * current
```

- [ ] **Step 7: Update `src/routesmith/predictor/learner.py`**

Change the `update` signature from:

```python
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
```

To:

```python
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
    ) -> None:
```

Find the EMA update line `self._ema_priors[model_id] = alpha * actual_quality + (1 - alpha) * current`
and change it to:

```python
        target = reward_override if reward_override is not None else actual_quality
        self._ema_priors[model_id] = alpha * target + (1 - alpha) * current
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/pytest tests/test_reward.py::TestPredictorRewardOverride -v
.venv/bin/pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/routesmith/predictor/base.py src/routesmith/predictor/linucb.py \
        src/routesmith/predictor/lints.py src/routesmith/predictor/embedding.py \
        src/routesmith/predictor/learner.py
git commit -m "feat(predictor): add reward_override param to all update() implementations"
```

---

## Task 4: YAML loader — parse `feedback.reward`

**Files:**
- Modify: `src/routesmith/cli/yaml_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reward.py`:

```python
import tempfile
import os


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
            config, _ = load_config_file(path)
            assert config.reward_expr == "quality - 0.15 * cost_normalized"
        finally:
            os.unlink(path)

    def test_no_feedback_reward_leaves_reward_expr_none(self):
        from routesmith.cli.yaml_loader import load_config_file
        yaml_content = "routing:\n  predictor: lints\nmodels: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config, _ = load_config_file(path)
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
            config, _ = load_config_file(path)
            assert config.reward_expr is None
        finally:
            os.unlink(path)
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/pytest tests/test_reward.py::TestYamlLoaderReward -v
```

Expected: `AssertionError: assert None == "quality - 0.15 * cost_normalized"`

- [ ] **Step 3: Update `src/routesmith/cli/yaml_loader.py`**

In the `# ── Feedback ──` block, change from:

```python
    if "feedback" in data:
        fb = data["feedback"]
        config.feedback_enabled = fb.get("enabled", True)
        if "sample_rate" in fb:
            config.feedback_sample_rate = fb["sample_rate"]
```

To:

```python
    if "feedback" in data:
        fb = data["feedback"]
        config.feedback_enabled = fb.get("enabled", True)
        if "sample_rate" in fb:
            config.feedback_sample_rate = fb["sample_rate"]
        if "reward" in fb:
            config.reward_expr = fb["reward"]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_reward.py::TestYamlLoaderReward -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/cli/yaml_loader.py
git commit -m "feat(yaml): parse feedback.reward expression string into config.reward_expr"
```

---

## Task 5: `client.py` — resolve reward_fn, apply in `record_outcome`

**Files:**
- Modify: `src/routesmith/client.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reward.py`:

```python
from unittest.mock import patch


def _make_client(config=None):
    from routesmith import RouteSmith, RouteSmithConfig
    cfg = config or RouteSmithConfig()
    cfg.feedback_sample_rate = 1.0
    rs = RouteSmith(config=cfg)
    rs.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                      cost_per_1k_output=0.60, quality_score=0.8)
    rs.register_model("openai/gpt-4o", cost_per_1k_input=2.50,
                      cost_per_1k_output=10.0, quality_score=0.95)
    return rs


def _insert_record(rs, request_id="req-001"):
    from routesmith.feedback.collector import FeedbackRecord
    import time
    fake_response = MagicMock()
    fake_response.usage.prompt_tokens = 100
    fake_response.usage.completion_tokens = 50
    record = FeedbackRecord(
        request_id=request_id,
        messages=[{"role": "user", "content": "Hi"}],
        model_id="openai/gpt-4o-mini",
        response=fake_response,
        latency_ms=150.0,
        timestamp=time.time(),
    )
    rs.feedback._request_index[request_id] = record
    return record


class TestClientRewardFn:
    def test_no_reward_fn_stores_none(self):
        from routesmith import RouteSmith
        assert RouteSmith()._reward_fn is None

    def test_reward_fn_callable_stored(self):
        from routesmith import RouteSmith, RouteSmithConfig
        fn = lambda ctx: ctx["quality"]
        rs = RouteSmith(config=RouteSmithConfig(reward_fn=fn))
        assert rs._reward_fn is fn

    def test_reward_expr_compiled_on_init(self):
        from routesmith import RouteSmith, RouteSmithConfig
        rs = RouteSmith(config=RouteSmithConfig(
            reward_expr="quality - 0.15 * cost_normalized"
        ))
        assert callable(rs._reward_fn)

    def test_bad_reward_expr_raises_on_init(self):
        from routesmith import RouteSmith, RouteSmithConfig
        with pytest.raises(ValueError):
            RouteSmith(config=RouteSmithConfig(reward_expr="quality ***"))

    def test_reward_fn_passes_override_to_predictor(self):
        from routesmith import RouteSmithConfig

        called_with = {}
        def _spy(messages, model_id, actual_quality, reward_override=None):
            called_with["reward_override"] = reward_override

        rs = _make_client(RouteSmithConfig(reward_fn=lambda ctx: 0.42, feedback_sample_rate=1.0))
        rs.router.predictor.update = _spy
        _insert_record(rs, "req-001")

        rs.record_outcome("req-001", score=0.9)
        assert abs(called_with["reward_override"] - 0.42) < 1e-9

    def test_no_reward_fn_passes_none_override(self):
        called_with = {}
        def _spy(messages, model_id, actual_quality, reward_override=None):
            called_with["reward_override"] = reward_override

        rs = _make_client()
        rs.router.predictor.update = _spy
        _insert_record(rs, "req-002")

        rs.record_outcome("req-002", score=0.9)
        assert called_with["reward_override"] is None

    def test_reward_fn_exception_logs_warning_and_uses_none(self):
        from routesmith import RouteSmithConfig

        called_with = {}
        def _spy(messages, model_id, actual_quality, reward_override=None):
            called_with["reward_override"] = reward_override

        def _bad_fn(ctx):
            raise RuntimeError("boom")

        rs = _make_client(RouteSmithConfig(reward_fn=_bad_fn, feedback_sample_rate=1.0))
        rs.router.predictor.update = _spy
        _insert_record(rs, "req-003")

        with patch("routesmith.client.logger") as mock_logger:
            rs.record_outcome("req-003", score=0.9)
            mock_logger.warning.assert_called_once()

        assert called_with["reward_override"] is None

    def test_reward_expr_end_to_end(self):
        from routesmith import RouteSmithConfig

        called_with = {}
        def _spy(messages, model_id, actual_quality, reward_override=None):
            called_with["reward_override"] = reward_override

        rs = _make_client(RouteSmithConfig(reward_expr="quality", feedback_sample_rate=1.0))
        rs.router.predictor.update = _spy
        _insert_record(rs, "req-004")

        rs.record_outcome("req-004", score=0.85)
        # reward_expr="quality" -> reward_override == 0.85
        assert abs(called_with["reward_override"] - 0.85) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_reward.py::TestClientRewardFn -v
```

Expected: `AttributeError: 'RouteSmith' object has no attribute '_reward_fn'`

- [ ] **Step 3: Add `logger` to `src/routesmith/client.py`**

After the last existing import line, before the `@dataclass` line, add:

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 4: Resolve `reward_fn` in `RouteSmith.__init__`**

At the end of `__init__`, after `self._last_routing_metadata: RoutingMetadata | None = None`, add:

```python
        # Resolve reward_fn from config (fail fast on bad expressions).
        if self.config.reward_fn is not None:
            self._reward_fn = self.config.reward_fn
        elif self.config.reward_expr is not None:
            from routesmith.feedback.reward import compile_reward_fn
            self._reward_fn = compile_reward_fn(self.config.reward_expr)
        else:
            self._reward_fn = None
```

- [ ] **Step 5: Update `record_outcome` to pass `reward_override`**

Replace the predictor update block inside `record_outcome` from:

```python
        if quality is not None:
            record = self.feedback.get_record_by_id(request_id)
            if record is not None:
                self.router.predictor.update(
                    messages=record.messages,
                    model_id=record.model_id,
                    actual_quality=quality,
                )
```

To:

```python
        if quality is not None:
            record = self.feedback.get_record_by_id(request_id)
            if record is not None:
                reward_override = None
                if self._reward_fn is not None:
                    from routesmith.feedback.reward import build_reward_context
                    ctx = build_reward_context(
                        model_id=record.model_id,
                        quality=quality,
                        response=record.response,
                        latency_ms=record.latency_ms,
                        registry=self.registry,
                    )
                    try:
                        reward_override = float(self._reward_fn(ctx))
                    except Exception as e:
                        logger.warning(
                            "reward_fn raised an error, skipping reward override: %s", e
                        )
                self.router.predictor.update(
                    messages=record.messages,
                    model_id=record.model_id,
                    actual_quality=quality,
                    reward_override=reward_override,
                )
```

- [ ] **Step 6: Run client tests**

```bash
.venv/bin/pytest tests/test_reward.py::TestClientRewardFn -v
```

Expected: all pass.

- [ ] **Step 7: Run full suite**

```bash
.venv/bin/pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/routesmith/client.py
git commit -m "feat(client): resolve reward_fn on init and apply reward_override in record_outcome"
```

---

## Task 6: Final verification + push

- [ ] **Step 1: Run complete test suite with verbose output**

```bash
.venv/bin/pytest tests/test_reward.py -v
.venv/bin/pytest tests/ -q
```

Expected: all pass. `test_reward.py` should show ~35 passing tests.

- [ ] **Step 2: Run linter**

```bash
.venv/bin/python -m ruff check src/routesmith/feedback/reward.py \
    src/routesmith/config.py src/routesmith/client.py \
    src/routesmith/predictor/base.py src/routesmith/predictor/linucb.py \
    src/routesmith/predictor/lints.py src/routesmith/predictor/embedding.py \
    src/routesmith/predictor/learner.py src/routesmith/cli/yaml_loader.py
```

Expected: no errors.

- [ ] **Step 3: Commit plan and push**

```bash
git add docs/superpowers/plans/2026-04-12-custom-reward-fn.md \
        docs/superpowers/specs/2026-04-10-reward-function-design.md
git push -u origin feature/custom-reward-fn
```
