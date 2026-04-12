# Custom Reward Function Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to define how the routing system weighs quality vs. cost vs. latency, via YAML expression strings or Python callables.

**Architecture:** After each LLM completion, RouteSmith builds a `RewardContext` dict from the response metadata, evaluates the user-supplied `reward_fn`, and passes the scalar result to the predictor's `update()` method instead of the predictor's internal default. When no custom reward function is configured, each predictor keeps its existing internal reward computation (backward compatible).

**Tech Stack:** `simpleeval` (restricted expression evaluator for YAML path), Python callables (Python API path), existing `linucb.py` / `lints.py` predictor internals.

---

## Problem

LinUCB internally computes `reward = quality - 0.15 * cost_normalized`. LinTS uses raw `quality`. Neither is configurable — users who want to penalize latency, or weight cost more aggressively, have no lever.

## Solution

Introduce an optional `reward_fn: Callable[[RewardContext], float]` at the `RouteSmithConfig` level. When set:
1. After each completion, `client.py` builds a `RewardContext` from the response.
2. Calls `reward_fn(ctx)` to get a scalar.
3. Passes that scalar to `predictor.update(model_id, x, reward=scalar)` (bypassing the predictor's internal reward computation).

When **not** set, each predictor's existing internal behavior is unchanged.

---

## RewardContext

Typed dict available inside expressions and callables:

| Key | Type | Description |
|---|---|---|
| `quality` | float (0–1) | Quality score from `record_feedback()` |
| `cost_usd` | float | Actual cost in USD |
| `cost_normalized` | float (0–1) | `cost_usd / max_model_cost_per_1k` |
| `latency_ms` | float | Wall-clock time for the completion call |
| `tokens_in` | int | Prompt token count |
| `tokens_out` | int | Completion token count |
| `model_id` | str | Model that was selected, e.g. `"openai/gpt-4o-mini"` |

---

## User-Facing APIs

### YAML (expression string)

```yaml
feedback:
  reward: "quality - 0.15 * cost_normalized"
```

Parsed by `yaml_loader.py`, compiled via `simpleeval`. Restricted to arithmetic operators and the variables above — no `import`, `exec`, arbitrary attribute access, or function calls (except `min`/`max`/`abs`).

### Python API (callable)

```python
config = RouteSmithConfig(
    reward_fn=lambda ctx: 0.7 * ctx["quality"] - 0.3 * ctx["cost_normalized"],
)
```

Any Python callable that accepts a `RewardContext` dict and returns a float.

### No reward_fn (default — backward compatible)

Each predictor retains its current internal reward computation:
- LinUCB: `quality - linucb_cost_lambda * cost_normalized`
- LinTS: `quality`
- Adaptive/embedding: unchanged

---

## Components

### New: `src/routesmith/feedback/reward.py`

```python
RewardContext = TypedDict(...)  # typed dict definition

def build_reward_context(model_id, quality, response, latency_ms, registry) -> RewardContext:
    """Build RewardContext from a completion response."""
    ...

def compile_reward_fn(expr: str) -> Callable[[RewardContext], float]:
    """Compile a simpleeval expression string into a callable."""
    ...
```

`compile_reward_fn` raises `ValueError` on syntax error (at compile time), and `ValueError` on evaluation error (at call time). This way bad expressions fail fast on startup (YAML path) rather than silently at runtime.

### Modified: `src/routesmith/config.py`

Add two optional fields to `RouteSmithConfig`:

```python
reward_fn: Callable[[RewardContext], float] | None = None
reward_expr: str | None = None  # used by yaml_loader; compiled to reward_fn on init
```

`RouteSmith.__init__()` resolves priority: `reward_fn` > `reward_expr` (compiled) > `None`.

`reward_expr` is the YAML-loaded string; it never coexists with `reward_fn` at runtime (the client resolves to one callable or `None`).

### Modified: `src/routesmith/cli/yaml_loader.py`

Read `feedback.reward` string from YAML, store as `reward_expr` on the config.

### Modified: `src/routesmith/client.py`

In `record_feedback()` (or wherever `predictor.update()` is called):

```python
if self._reward_fn is not None:
    ctx = build_reward_context(model_id, quality, response, latency_ms, self._registry)
    reward = self._reward_fn(ctx)
    self._predictor.update(model_id, x, reward=reward, skip_internal_reward=True)
else:
    self._predictor.update(model_id, x, reward=quality)  # predictor uses internal logic
```

The `skip_internal_reward=True` flag (or equivalent) tells the predictor to use the supplied scalar directly rather than recomputing its internal reward.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Bad YAML expression (syntax) | `ValueError` at `RouteSmith` init (fail fast) |
| Expression evaluates to non-float | `TypeError` logged; reward skipped for that call |
| Callable raises | Exception logged as warning; reward skipped for that call |
| `simpleeval` disallowed operation | `simpleeval.FeatureNotAvailable` → caught, re-raised as `ValueError` at compile time |

---

## Testing Plan

- Unit tests for `compile_reward_fn`: valid expressions, syntax errors, disallowed operations (e.g. `import`, `__class__`).
- Unit tests for `build_reward_context`: correct field computation (cost_normalized edge cases when max_cost=0).
- Integration tests: LinUCB predictor receives custom reward scalar when `reward_fn` is set; uses internal reward when not set.
- YAML round-trip test: parse YAML with `feedback.reward` string → `RouteSmith` init → verify reward callable works.
- Backward compat test: existing `RouteSmithConfig()` with no `reward_fn` produces same routing behavior as before.
