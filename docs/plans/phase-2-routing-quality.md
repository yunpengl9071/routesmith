# Phase 2 — Routing quality: neutralize NotDiamond's head start, then pass it

NotDiamond's one real advantage is offline pre-training: it routes sensibly on request #1.
RouteSmith's cold start is a price-rank heuristic ("expensive = good",
`registry/openrouter.py`: `0.60 + 0.40 * log1p(rank)/log1p(n)`) or a hardcoded 0.8
(`client.py register_model quality_score: float = 0.8`). This phase ships benchmark-grounded
warm starts, semantic query understanding, verified cascade execution, and product-code
benchmark numbers we can publish — attacking the #1 criticism of `openrouter/auto`
(no benchmark data of its own) with reproducible transparency.

**Exit criteria:** G3 (`tests/test_perf_routing.py`) and G5 (`make bench-product` APGR ≥ 0.55) met.

Task order: P2.1 → P2.2 → P2.3 → P2.4 → P2.5. (P2.1 and P2.2 independent; P2.4 needs P1.3/P1.4.)

---

## Task P2.1 — Benchmark-grounded warm-start priors  (size: M)

**Files:** `src/routesmith/registry/priors.py` (new),
`src/routesmith/registry/data/default_priors.json` (new), `src/routesmith/registry/openrouter.py`,
`src/routesmith/client.py`, `pyproject.toml`, `tests/test_priors.py` (new)

**Spec — data file** `src/routesmith/registry/data/default_priors.json` (curated from public
benchmark aggregates; maintainers update quarterly — scores are relative quality in [0,1],
consistent scale, NOT raw benchmark numbers). Seed with exactly this content:

```json
{
  "version": 1,
  "updated": "2026-07",
  "priors": {
    "openai/gpt-4o": 0.92,
    "openai/gpt-4o-mini": 0.82,
    "openai/gpt-4.1": 0.93,
    "openai/gpt-4.1-mini": 0.84,
    "openai/o3": 0.96,
    "anthropic/claude-opus-4": 0.96,
    "anthropic/claude-sonnet-4": 0.92,
    "anthropic/claude-3.5-haiku": 0.80,
    "anthropic/claude-haiku-4.5": 0.86,
    "google/gemini-2.5-pro": 0.94,
    "google/gemini-2.5-flash": 0.85,
    "google/gemini-2.0-flash": 0.81,
    "meta-llama/llama-3.1-70b-instruct": 0.80,
    "meta-llama/llama-3.1-8b-instruct": 0.68,
    "meta-llama/llama-4-maverick": 0.84,
    "deepseek/deepseek-chat": 0.83,
    "deepseek/deepseek-r1": 0.90,
    "mistralai/mistral-large": 0.84,
    "mistralai/mistral-small": 0.74,
    "qwen/qwen-2.5-72b-instruct": 0.81
  }
}
```

**Spec — `src/routesmith/registry/priors.py`:**

```python
"""Benchmark-grounded quality priors for cold-start routing."""
from __future__ import annotations

import json
from importlib import resources


def load_default_priors() -> dict[str, float]:
    """Load the packaged prior table. Returns {} if the data file is missing."""
    try:
        ref = resources.files("routesmith.registry.data").joinpath("default_priors.json")
        return json.loads(ref.read_text())["priors"]
    except Exception:
        return {}


def lookup_prior(model_id: str, priors: dict[str, float]) -> float | None:
    """Match order: exact id -> id without provider prefix -> unique substring.

    'openrouter/openai/gpt-4o' matches key 'openai/gpt-4o' via suffix.
    Returns None when no unambiguous match exists.
    """
    if model_id in priors:
        return priors[model_id]
    for key, score in priors.items():
        if model_id.endswith("/" + key) or key.endswith("/" + model_id):
            return score
    tail = model_id.rsplit("/", 1)[-1]
    hits = [s for k, s in priors.items() if k.rsplit("/", 1)[-1] == tail]
    return hits[0] if len(hits) == 1 else None
```

Create `src/routesmith/registry/data/__init__.py` (empty) and add package-data so the JSON
ships in wheels — in `pyproject.toml`, following the existing build config style (check for
`[tool.setuptools]`; add `package-data = {"routesmith.registry.data" = ["*.json"]}` or the
equivalent for the build backend actually in use).

**Wiring:**
1. `openrouter.py fetch_models()`: for each model, `prior = lookup_prior(model_id, priors)`;
   use it when not None, else keep the existing cost-rank heuristic. Load priors once per call.
2. `client.py register_model()`: signature gains nothing; but when the caller leaves
   `quality_score` at the default `0.8`, check `lookup_prior` first and prefer it. To detect
   "caller left default", change the parameter default to `quality_score: float | None = None`
   and resolve `None → lookup_prior(...) or 0.8` inside. Grep callers/tests for positional
   passing before changing.

**Tests** (`tests/test_priors.py`):
- `test_load_default_priors_has_20_entries`.
- `test_lookup_exact` / `test_lookup_openrouter_prefixed`
  (`"openrouter/openai/gpt-4o"` → 0.92) / `test_lookup_tail_unique`
  (`"gpt-4o-mini"` → 0.82) / `test_lookup_ambiguous_returns_none` /
  `test_lookup_unknown_returns_none`.
- `test_register_model_uses_prior` — `rs.register_model("openai/gpt-4o", ...)` without
  quality_score → registry entry has `quality_score == 0.92`.
- `test_register_model_explicit_score_wins` — pass `quality_score=0.5` → 0.5 kept.

**Acceptance:** tests pass; README cold-start section explains the prior table, its source
policy, and how to override (`quality_score=` always wins).

---

## Task P2.2 — Optional embedding features for the bandit  (size: L)

Keyword overlap cannot distinguish "write a haiku" from "write a distributed lock". Add an
opt-in 8-dim semantic block to the feature vector.

**Depends on:** PR #22 (`extract_message_and_context` / `extract_for_model` split) — verify
present with `grep -n "extract_message_and_context" src/routesmith/predictor/features.py`;
cherry-pick `02e7e02` if absent.

**Files:** `src/routesmith/predictor/features.py`, `src/routesmith/predictor/lints.py`,
`src/routesmith/predictor/linucb.py`, `src/routesmith/config.py`, `pyproject.toml`,
`tests/test_embedding_features.py` (new)

**Spec:**
1. `config.py` `PredictorConfig`: add
   `embedding_features: bool = False` and
   `embedding_model: str = "all-MiniLM-L6-v2"`.
2. `features.py` — `FeatureExtractor.__init__` gains
   `use_embeddings: bool = False, embedding_model: str = "all-MiniLM-L6-v2"`. When True:
   - Lazy-load `sentence_transformers.SentenceTransformer(embedding_model)` on first use; on
     ImportError log ONE warning and permanently disable (fall back to 35-dim).
   - Build a fixed projection matrix once:
     `P = np.random.default_rng(42).normal(0, 1, (384, 8)) / np.sqrt(384.0)` (seed MUST be 42
     for reproducibility across processes).
   - New message-level step: embed the LAST USER message (`encode(text, normalize_embeddings=True)`),
     project: `sem = (vec @ P).tolist()` → 8 floats appended AFTER the context block
     (indices 35–42). Feature names `sem_0..sem_7` appended to `ALL_FEATURE_NAMES` only when
     enabled.
   - Add property `dim` returning 43 when active else 35. Compute the embedding ONCE per
     `extract_message_and_context()` call (it is model-independent) — never per candidate.
   - Cache: memoize the last `(text_hash → sem)` pair to make repeated turns cheap.
3. `lints.py` / `linucb.py`: replace hardcoded `d = 35` / `fv.features[:35]` slicing with
   `d = self._extractor.dim` and `[:d]`. `load_state` already cold-starts on dimension
   mismatch (`if stored_d != self._router.d: return`) — verify and keep; document that
   enabling embeddings resets learned state.
4. `pyproject.toml`: add extra `embeddings = ["sentence-transformers>=2.2.0"]` (dedupe with
   the existing `predictor` extra if it already pins it — inspect and reuse if identical).

**Tests** (`tests/test_embedding_features.py` — inject a FAKE encoder; never load real
models in CI):
- Fake: `class FakeEncoder: def encode(self, text, normalize_embeddings=True): return
  deterministic_384_vector(text)` (hash-seeded RNG). Inject via a constructor test seam
  `_encoder_override` param on FeatureExtractor.
- `test_dim_43_when_enabled` / `test_dim_35_when_disabled`.
- `test_semantic_dims_differ_across_topics` — "solve this integral" vs "write a poem" →
  `sem` blocks differ; same text twice → identical (cache).
- `test_embedding_once_per_predict` — encoder call count == 1 after a 5-candidate
  `LinTSPredictor.predict()`.
- `test_state_cold_start_on_dim_change` — save 35-dim state, load into 43-dim predictor →
  state ignored, no exception.
- `test_missing_dependency_falls_back` — force ImportError → dim 35, one warning.

**Acceptance:** tests pass; G3 perf test (P2.5) still passes with embeddings OFF (default);
README documents the tradeoff (semantic routing vs ~10–30 ms encode on first turn).

---

## Task P2.3 — Benchmark the PRODUCT router, not research reimplementations  (size: M)

`benchmark/strategies/` contains standalone LinTS/LinUCB copies; published numbers therefore
don't validate shipped code. Add an adapter so experiments exercise `routesmith.Router`.

**Files:** `benchmark/strategies/product_router.py` (new), `Makefile` (add `bench-product`),
`tests/test_benchmark_product.py` (new)

**Spec:**
1. Read `benchmark/harness.py` and one existing strategy (e.g. `benchmark/strategies/lints.py`)
   to learn the exact Strategy interface (constructor, `select(question) -> model_id`,
   `update(question, model_id, reward)` or equivalent — mirror precisely).
2. Implement `ProductRouterStrategy` satisfying that interface by delegating to REAL product
   classes: build `ModelRegistry` + `RouteSmithConfig(predictor_type="lints")` + `Router`;
   `select()` calls `router.route(messages=[{"role":"user","content":question}],
   strategy=RoutingStrategy.DIRECT, min_quality=0.0)`; `update()` calls
   `router.predictor.update(...)`. No reimplementation of any math.
3. `Makefile` target `bench-product`: runs the existing exp1 entry point with
   `--strategy product_router --dataset mmlu --n 600` (match the real CLI of
   `benchmark/experiments/exp1_binary.py` — read it first) and prints the APGR line.
4. CI smoke `tests/test_benchmark_product.py::test_product_strategy_runs_offline` — 10
   synthetic questions with scripted rewards (no API): select → update loop completes, and
   selections are drawn from the registered model set.

**Measurable target (G5):** `make bench-product` on MMLU-600 reports **APGR ≥ 0.55**
(research LinTS scored 0.593; the product path must land within ~10% of it). Record the
number in `benchmark/BENCHMARK_STATE.md`. If below target, the gap is a routing bug —
investigate feature plumbing (most likely: context/min_quality defaults differing from the
research harness) before tuning hyperparameters.

**Acceptance:** smoke test green in CI; one full manual run recorded in BENCHMARK_STATE.md
with command, commit hash, and APGR.

---

## Task P2.4 — Real cascade execution with verification  (size: L)

Today all strategies collapse to one call (`client.py` single `litellm.completion`).
Verified cascade — cheap first, check, escalate — is a categorically stronger cost saver
than single-shot prediction and has no OpenRouter equivalent.

**Depends on:** P1.3 (implicit signals), P1.4 (judge; optional at runtime).

**Files:** `src/routesmith/client.py`, `src/routesmith/config.py`,
`tests/test_cascade_execution.py` (new), README/tutorial (restore cascade claims)

**Spec:**
1. `config.py`: add to `RouteSmithConfig`:
   `cascade_max_tiers: int = 3` and
   `cascade_accept_threshold: float = 0.6`
   `# Judge score below this escalates to the next tier (judge enabled only)`.
2. `client.py completion()`: where the strategy is resolved, branch:

   ```python
   if effective_strategy == RoutingStrategy.CASCADE:
       return self._completion_cascade(messages, min_quality, required_capabilities,
                                       context, request_id, **kwargs)
   ```
3. New private method `_completion_cascade` — exact algorithm:
   1. `tiers = self.router.get_cascade_models(min_quality=..., max_tiers=self.config.cascade_max_tiers,
      required_capabilities=...)`; if empty, fall back to normal direct routing.
   2. For each `tier_model` in order: call litellm (reuse the same call/error handling as the
      direct path — extract a `_execute_model()` helper if needed to avoid duplicating the
      fallback logic from P0.3). Accumulate cost for EVERY attempt (honest accounting).
   3. **Verify** the response:
      - Extract implicit signals (same extractor the collector uses). Any hard-negative
        (`error`/`refusal`/`empty`) → REJECT tier.
      - Else if `self._judge is not None`: judge-score the response; score
        `< cascade_accept_threshold` → REJECT tier. Judge failure (None) → ACCEPT (fail-open).
      - Else ACCEPT.
   4. On ACCEPT: record feedback for the accepted model (and negative outcomes
      `record_outcome(tier_request_id, score=0.1)` for each rejected tier so the bandit
      learns from escalations), attach metadata:
      `{"strategy": "cascade", "tiers_tried": [...], "escalations": n}` and return.
   5. If ALL tiers rejected: return the LAST response anyway with metadata
      `"cascade_exhausted": True` (never fail a request that produced output).
4. `acompletion()`: same via an async twin (share verification logic in a small pure helper).

**Tests** (`tests/test_cascade_execution.py`, litellm mocked with scripted per-model
responses):
- `test_cascade_accepts_first_tier_when_clean` — 1 call total.
- `test_cascade_escalates_on_refusal` — tier1 returns refusal text, tier2 clean → 2 calls,
  final model == tier2, metadata `escalations == 1`, `tiers_tried` length 2.
- `test_cascade_escalates_on_judge_reject` — judge enabled, scores [0.3, 0.9] → escalates once.
- `test_cascade_exhausted_returns_last` — all tiers refuse → response returned,
  `cascade_exhausted` set.
- `test_cascade_cost_accumulates_all_attempts` — 2-tier escalation → `stats` total cost equals
  sum of both attempts.
- `test_cascade_rejected_tiers_get_negative_feedback` — predictor received an update ≤ 0.1
  for the rejected tier's model.

**Acceptance:** tests pass; README cascade section rewritten to describe the REAL algorithm
(with the verification ladder and the note that parallel/speculative remain selection-only
until scheduled — remove them from the feature table until then).

---

## Task P2.5 — Routing-overhead performance guardrail (G3)  (size: S)

**Files:** `tests/test_perf_routing.py` (new), `pyproject.toml` (register `perf` marker)

**Spec:** With a 5-model registry and LinTS predictor (no embeddings), measure
`router.route(...)` over 200 iterations after 20 warmup iterations, one representative
2-message conversation. Compute p50/p99 from `time.perf_counter` deltas.
`assert p50 < 0.005 and p99 < 0.015`. Mark `@pytest.mark.perf`; register the marker;
run in CI on a quiet job (allow 2× headroom via env var `PERF_MULTIPLIER` read as float,
default 1.0, so flaky runners can loosen without editing the test).

**Acceptance:** passes locally at 1.0 multiplier on the dev branch.
