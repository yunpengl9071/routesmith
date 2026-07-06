# Phase 0 — Credibility: make every documented feature real (or undocumented)

**Why first:** RouteSmith's competitive pitch against OpenRouter's black-box Auto Router is
transparency and trust. That pitch dies if the README advertises features that are dead code.
This phase makes the smallest set of changes so that everything documented actually runs.

**Exit criteria:** G2 (`scripts/check_claims.sh` green), G7 (cache), G8 (budget) pass;
README quickstart runs verbatim.

Task order: P0.0 → P0.1 → P0.2 → P0.3 → P0.4 → P0.5 → P0.6. (P0.3/P0.4/P0.5 are independent
of each other.)

---

## Task P0.0 — Shared test fixture for fake LiteLLM responses  (size: S)

Every later task mocks `litellm`. Build the helper once.

**Files:** `tests/helpers.py` (new), `tests/conftest.py` (edit)

**Spec:** Create `tests/helpers.py`:

```python
"""Shared test helpers. No routesmith imports at module level."""
from types import SimpleNamespace


def fake_response(
    content: str = "ok",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
):
    """Build an object that quacks like a litellm ModelResponse."""
    message = SimpleNamespace(content=content, tool_calls=None, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    resp = SimpleNamespace(choices=[choice], usage=usage, model=model, id="chatcmpl-test")
    resp.model_dump = lambda: {
        "id": resp.id,
        "model": resp.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


def make_rs(**config_kwargs):
    """RouteSmith with two standard test models registered.

    gpt-4o:      expensive, quality 0.95
    gpt-4o-mini: cheap,     quality 0.85
    """
    from routesmith import RouteSmith
    from routesmith.config import RouteSmithConfig

    rs = RouteSmith(config=RouteSmithConfig(**config_kwargs))
    rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                      cost_per_1k_output=0.015, quality_score=0.95)
    rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                      cost_per_1k_output=0.0006, quality_score=0.85)
    return rs
```

**Tests:** `tests/test_helpers.py::test_fake_response_shape` — assert
`fake_response().usage.total_tokens == 30` and `fake_response().model_dump()["choices"][0]["message"]["content"] == "ok"`.

**Acceptance:** helper importable from any test as `from tests.helpers import fake_response, make_rs`
(add `tests/` package import path handling in `tests/conftest.py` if needed).

---

## Task P0.1 — Make LinTS the default predictor  (size: S)

README says "LinTS-27d (recommended, default)"; code default is `"adaptive"` (random forest),
which does nothing until 100 samples arrive. Make the code match the docs.

**Files:** `src/routesmith/config.py`, `README.md`, any failing tests

**Spec:**
1. In `config.py`, find `predictor_type: str = "adaptive"` (line ~83). Change to
   `predictor_type: str = "lints"`. Update the trailing comment to
   `# lints (default), linucb, adaptive, embedding`.
2. Grep tests for assumptions: `grep -rn '"adaptive"' tests/`. Where a test asserts the
   *default* is adaptive, update it to `"lints"`. Where a test *explicitly configures*
   adaptive, leave it alone.

**Tests:** `tests/test_config.py::test_default_predictor_is_lints` (new) —
`assert RouteSmithConfig().predictor_type == "lints"`.
Also `tests/test_router.py::test_default_router_uses_lints` (new) — build a `Router` with a
default config and 2-model registry, `assert type(router.predictor).__name__ == "LinTSPredictor"`.

**Acceptance:** both new tests pass; full suite green.

---

## Task P0.2 — Fix README/tutorial drift  (size: S)

**Files:** `README.md`, `docs/tutorial.md`, `scripts/check_claims.sh` (new)

**Spec:** Apply these corrections everywhere they appear (grep each pattern):

| Grep for | Replace with |
|---|---|
| `27-dim`, `27-dimensional`, `LinTS-27d`, `LinUCB-27d` | 35-dim / `LinTS-35d` / `LinUCB-35d` (feature vector is 35-dim: `features.py` docstring "Produces a 35-dimensional feature vector") |
| `rs.complete(` | `rs.completion(` |
| `response.request_id` | `response._routesmith_request_id` |
| Claims that cascade/parallel/speculative "execute", "escalate", "run multiple models" | Reword to "cascade execution lands in Phase 2 (see ROADMAP.md); today all strategies select a single model" — UNTIL Task P2.4 ships, then restore |
| "enforces per-request and daily cost limits" | Keep only after P0.4 is merged; if P0.4 is already merged, leave as-is |
| Cache claims ("reuses responses", "a hit skips the model call") | Keep only after P0.5 is merged |

Create `scripts/check_claims.sh`:

```bash
#!/usr/bin/env bash
# Fails if known-false claims reappear in docs. Extend when new claims ship.
set -e
fail=0
check_absent() {
  if grep -rn "$1" README.md docs/ --include='*.md' 2>/dev/null | grep -v ROADMAP | grep -v plans/; then
    echo "STALE CLAIM FOUND: $1"; fail=1
  fi
}
check_absent "27-dimensional"
check_absent "LinTS-27d"
check_absent "rs.complete("
check_absent "response.request_id"
exit $fail
```

**Tests:** `bash scripts/check_claims.sh` exits 0.

**Acceptance:** script green; README quickstart code block is copy-paste runnable
(verified properly in P5.6).

---

## Task P0.3 — Implement `fallback_model` + failure retry  (size: M)

`config.fallback_model` (config.py line ~80, `fallback_model: str | None = None`) is parsed
but read nowhere. On any model failure `completion()` re-raises immediately
(client.py: `except Exception as e:` following `response = litellm.completion(`).

**Files:** `src/routesmith/client.py`, `tests/test_fallback.py` (new)

**Spec:** In `RouteSmith.completion()` AND `acompletion()`:

1. Wrap the existing `litellm.completion(...)` call. On `Exception as primary_error`:
   - Let `fb = self.config.fallback_model`.
   - If `fb` is truthy AND `fb != selected_model` AND `self.registry.get(fb) is not None`:
     log a warning `f"Primary model {selected_model} failed ({primary_error}); retrying with fallback {fb}"`,
     set `selected_model = fb`, and retry the litellm call once with identical kwargs.
     If the retry succeeds, continue the normal success path (cost, feedback, metadata) with
     the fallback model; add `"fallback_from": <original model id>` to the routing metadata dict.
   - Otherwise (or if the fallback also raises): preserve the CURRENT behavior exactly —
     record failure via the existing `self.feedback.record_outcome(... success=False ...)`
     call, then re-raise the ORIGINAL `primary_error` (`raise primary_error from None` when the
     fallback also failed, so the user sees the primary cause).
2. Do not add retries of the same model (LiteLLM's own `num_retries` can be passed by users
   through `litellm_params`; out of scope here).

**Tests** (`tests/test_fallback.py`, mock litellm per P0.0):
- `test_fallback_used_on_primary_failure` — primary raises `RuntimeError("boom")`, fallback
  configured + registered; call succeeds; `response.model == fallback`; metadata contains
  `fallback_from`.
- `test_no_fallback_configured_reraises` — no fallback set → `pytest.raises(RuntimeError)`.
- `test_fallback_not_registered_reraises` — fallback set but not registered → re-raise.
- `test_fallback_same_as_selected_reraises` — fallback == selected model → re-raise (no loop).
- `test_fallback_failure_reraises_primary` — both raise; assert the raised exception is the
  primary one.
- Async duplicates for `acompletion` (`test_async_fallback_used_on_primary_failure`).

**Acceptance:** all 6+ tests pass; no change to the success path when no error occurs.

---

## Task P0.4 — Enforce budgets for real  (size: M)

`BudgetConfig.max_cost_per_minute/hour/day` (config.py lines ~47–49) are parsed and never
read. Only `quality_threshold` is honored.

**Files:** `src/routesmith/budget.py` (new), `src/routesmith/client.py`,
`src/routesmith/__init__.py`, `src/routesmith/proxy/server.py`, `tests/test_budget.py` (new),
`tests/test_proxy_budget.py` (new)

**Spec — new module `src/routesmith/budget.py`:**

```python
"""Rolling-window budget enforcement."""
from __future__ import annotations

import threading
import time
from collections import deque

WINDOWS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}


class BudgetExceededError(RuntimeError):
    """Raised pre-flight when a spend window is exhausted."""

    def __init__(self, window: str, limit: float, spent: float) -> None:
        self.window = window
        self.limit = limit
        self.spent = spent
        super().__init__(
            f"Budget exceeded: spent ${spent:.4f} of ${limit:.4f} in the last {window}"
        )


class BudgetTracker:
    """Tracks spend in rolling minute/hour/day windows. Thread-safe."""

    def __init__(self, budget_config) -> None:  # BudgetConfig
        self._config = budget_config
        self._events: deque[tuple[float, float]] = deque()  # (timestamp, cost)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        # Drop events older than the largest window
        cutoff = now - WINDOWS["day"]
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def spent(self, window: str, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            cutoff = now - WINDOWS[window]
            return sum(c for ts, c in self._events if ts >= cutoff)

    def check(self, now: float | None = None) -> None:
        """Raise BudgetExceededError if any configured window is exhausted."""
        now = time.time() if now is None else now
        limits = {
            "minute": self._config.max_cost_per_minute,
            "hour": self._config.max_cost_per_hour,
            "day": self._config.max_cost_per_day,
        }
        for window, limit in limits.items():
            if limit is not None:
                s = self.spent(window, now)
                if s >= limit:
                    raise BudgetExceededError(window, limit, s)

    def record(self, cost: float, now: float | None = None) -> None:
        if cost <= 0:
            return
        now = time.time() if now is None else now
        with self._lock:
            self._events.append((now, cost))
            self._prune(now)
```

**Wiring in `client.py`:**
1. In `RouteSmith.__init__`, after config is set:
   `self._budget = BudgetTracker(self.config.budget)`.
2. At the TOP of `completion()` and `acompletion()` (before routing):
   `self._budget.check()`.
3. `max_cost_per_request` becomes a pre-flight price filter: before calling
   `self.router.route(...)`, if `max_cost` (the per-call argument) is `None` and
   `self.config.budget.max_cost_per_request` is set, derive:
   ```python
   est_tokens = sum(len(m.get("content", "")) for m in messages) / 4.0 \
                + float(kwargs.get("max_tokens") or 1024)
   max_cost = self.config.budget.max_cost_per_request / est_tokens * 1000.0
   ```
   and pass that as the router's `max_cost` (it filters by `cost_per_1k_total`).
4. On the success path, where `self._total_cost += actual_cost` already happens
   (client.py: `self._total_cost += actual_cost`), also call `self._budget.record(actual_cost)`.

**Wiring in the proxy (`proxy/server.py`):** in `_handle_completion`, catch
`BudgetExceededError` and respond with status **429** and OpenAI-style error body
`{"error": {"message": str(e), "type": "budget_exceeded", "code": 429}}`.
Add `429: "Too Many Requests"` to the status-text map in `_send_json`.

**Export:** add `BudgetExceededError` to `routesmith/__init__.py` `__all__`.

**Tests** (`tests/test_budget.py` — use explicit `now=` values, never sleep):
- `test_tracker_spent_windows` — record 3 costs at t=0, 100, 4000; assert
  `spent("minute", now=4010) == cost3` and `spent("day", now=4010) == sum`.
- `test_check_raises_when_day_exhausted` — limit day=1.0, record 1.0 at t=0 →
  `check(now=10)` raises; `check(now=86401)` passes.
- `test_check_noop_when_unconfigured` — all limits None → never raises.
- `test_completion_blocked_when_over_budget` — `make_rs(budget=BudgetConfig(max_cost_per_day=0.000001))`,
  seed tracker via `rs._budget.record(1.0)`; `pytest.raises(BudgetExceededError)`; assert
  the mocked litellm was NOT called.
- `test_completion_records_spend` — one successful completion; `rs._budget.spent("day") > 0`.
- `test_max_cost_per_request_filters_models` — set `max_cost_per_request` low enough that
  gpt-4o's price is filtered; assert routed model is gpt-4o-mini.

`tests/test_proxy_budget.py`: drive `RouteSmithProxyServer._handle_completion` with a fake
writer (see existing `tests/test_proxy.py` for the pattern); over-budget request → response
contains `429` and `"budget_exceeded"`.

**Acceptance:** G8 green; docs updated: README "budget" section describes real behavior
including the 429 proxy response.

---

## Task P0.5 — Wire the semantic cache into the completion path  (size: M)

`SemanticCache` (cache/semantic.py) is complete but instantiated nowhere; `client.py` never
imports it. `CacheConfig.enabled` defaults to `False` (config.py line ~35) — wiring is
opt-in, so default behavior is unchanged.

**Files:** `src/routesmith/client.py`, `src/routesmith/cache/semantic.py`,
`tests/test_cache_wiring.py` (new)

**Spec:**
1. **Read `cache/semantic.py` first.** If its constructor cannot operate without
   `sentence-transformers` installed, add a degraded mode: constructor param
   `embedding_fn: Callable[[str], "np.ndarray"] | None = None`; when
   sentence-transformers is unavailable AND no `embedding_fn` is injected, the cache runs
   **exact-match only** (hash lookup; semantic search skipped) and logs one warning. It must
   never raise ImportError at construction.
2. In `RouteSmith.__init__`: when `self.config.cache.enabled` is true, build
   `self._cache = SemanticCache(similarity_threshold=..., ttl_seconds=..., max_entries=...,
   embedding_model=...)` from `CacheConfig` fields; else `self._cache = None`.
3. In `completion()` (sync; skip `acompletion` in this task, mirror later) — BEFORE routing:
   ```python
   if self._cache is not None and not kwargs.get("tools") and not kwargs.get("stream"):
       hit = self._cache.get(messages)
       if hit is not None:
           self._cache_hits += 1
           return copy.deepcopy(hit)   # never hand out a mutable shared object
   ```
   (`self._cache_hits = 0` initialized in `__init__`.)
4. AFTER a successful completion (same guard conditions): `self._cache.put(messages, response)`.
   Match the actual method names in `semantic.py` — if they differ (`lookup`/`store`), use the
   existing names; do not rename the cache's public API.
5. Add `"cache_hits": self._cache_hits` to the `stats` property dict.
6. Restore the README/tutorial cache claims removed in P0.2, now truthful, documenting:
   opt-in via `CacheConfig(enabled=True)`, exact-match works without extras, semantic
   matching requires `pip install routesmith[cache]`.

**Tests** (`tests/test_cache_wiring.py`, litellm mocked; force exact-match mode by injecting
`embedding_fn=None` and no sentence-transformers):
- `test_cache_disabled_by_default` — `make_rs()`; two identical completions → litellm called twice.
- `test_cache_hit_skips_model_call` — cache enabled; two identical requests → litellm called
  ONCE; second response content equals first; `rs.stats["cache_hits"] == 1`.
- `test_cache_hit_returns_copy` — mutate first response; second request unaffected.
- `test_cache_bypassed_for_tools` — pass `tools=[...]` kwarg → litellm called twice.
- `test_cache_hit_latency` — cached lookup completes in < 10 ms (`time.perf_counter` around
  the second call; generous bound, no embedding in exact mode).

**Acceptance:** G7 green; default-off confirmed by `test_cache_disabled_by_default`.

---

## Task P0.6 — OpenRouter registry fixes  (size: S)

Two known bugs in `src/routesmith/registry/openrouter.py`:

**Spec:**
1. `supports_function_calling` is currently read from
   `top_provider.is_supported_in_playground` (line ~79) — unrelated to tool calling. Fix: use
   `"tools" in model_data.get("supported_parameters", [])` when the field exists; default
   `True` when absent.
2. Zero-priced (free) models are silently dropped (line ~71). Keep dropping them (zero price
   breaks cost-aware routing) but count and log once:
   `logger.info(f"Skipped {n_free} free models (zero pricing)")`.

**Tests** (`tests/test_registry_openrouter.py`, new — feed `fetch_models` a canned JSON dict,
no network; if the function fetches internally, refactor minimally to accept
`_response_json: dict | None = None` test seam):
- `test_tools_support_from_supported_parameters` — model with `supported_parameters: ["tools"]`
  → `supports_function_calling is True`; with `["temperature"]` → `False`; absent → `True`.
- `test_free_models_skipped_and_counted` — 2 free of 5 → 3 registered, log contains "Skipped 2".

**Acceptance:** tests pass; no behavior change for priced models.
