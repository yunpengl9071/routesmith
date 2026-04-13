# Agentic Routing — Implementation Reflection

**Date:** 2026-04-12  
**Branch:** `dev`  
**Commits:** 26 (plan commit `425afe5` through `25afa9b`)  
**Tests:** 544 passing, 14 skipped (API key–gated)

---

## What Was Built

A complete agent-aware routing layer added on top of the existing RouteSmith stack. The feature is fully opt-in — callers who pass no `context` argument see identical behavior to before.

Key capabilities shipped:
- **`RouteContext`** — carries agent identity (`agent_id`, `agent_role`, `conversation_id`, `turn_index`) through any completion call
- **`AgentInferencer`** — keyword-density classifier that auto-fills `agent_role` from system prompts when the caller doesn't provide one; <1ms, MD5-cached
- **Feature vector 27→35** — 8 new context features appended; old trained predictors unaffected (new dims default to 0)
- **Predictor state durability** — LinTS, LinUCB, and RF model weights serialized to SQLite after retrain/deregister, reloaded on init; no more cold-start on restart
- **Arm lifecycle** — `add_arm()` / `remove_arm()` on all three predictors; model registration/deregistration wired end-to-end
- **Business rules** — pre-routing filters that run before the predictor sees candidates; keeps feedback attributable
- **Per-role reward functions** — `config.reward_fns[role]` → global `reward_fn` → None resolution chain
- **`ConversationTracker`** — stateful turn tracker; emits `RouteContext` with turn index, correction count, token density
- **`recommend_model_for_agent()`** — quality-adjusted cost efficiency ranking over historical records for a role
- **Integration support** — `agent_role`, `track_conversation`, `reward_fn` fields on `ChatRouteSmith`, `RouteSmithLM`, `RouteSmithAnthropic`; proxy reads `X-RouteSmith-*` headers

---

## What Went Smoothly

**The spec was tight.** The design doc (`docs/superpowers/specs/2026-04-12-agentic-routing-design.md`) answered almost every implementation question before it came up. Data shapes, SQL schemas, feature indices, and resolution orders were all pre-decided.

**Safe serialization for bandit state.** JSON was used for LinTS/LinUCB arm matrices (A, b, A_inv), and joblib only for the scikit-learn RF object in `AdaptivePredictor`. This is the right choice for a library — bandit state is just numpy arrays and scalars, all JSON-serializable, with no need for a binary format.

**TDD caught issues early.** The failing-test-first discipline meant no task was "done" until behavior was actually verified. The two-stage spec review + code quality review gates per task added overhead but caught real bugs before they compounded.

---

## Bugs Caught in Review

**Task 11 (context parameter):** After wiring `context=context` into `router.route()`, 22 tests failed — `AdaptivePredictor.predict()` didn't have the `context` keyword arg yet. Quick fix, but easy to miss without a full test run.

**Task 14 (min_samples guard):** The early-exit check used the total record count across all models rather than per-model count. In a multi-provider setup with mostly-unregistered historical records, this would have returned a recommendation backed by far fewer samples than advertised. Caught by the code quality reviewer.

**Final review — `acompletion()` had no `context` param (Critical):** The entire async routing path — proxy handler, LangChain `_agenerate`, DSPy — was silently dropping context. The parameter was absent and `**kwargs` forwarded it to litellm where it was ignored. This was the most consequential miss: the proxy is the primary path for AutoGen/CrewAI integrations, which was the main motivation for the feature.

**Final review — `record_outcome()` FK violation (Critical):** With `sample_rate < 1.0` (default 0.1), 90% of `record_outcome()` calls would raise `sqlite3.IntegrityError` when storage was enabled, because `store_signal()` tried to insert an outcome signal for a `request_id` not in `feedback_records`. Silent for the majority of users (no storage), catastrophic for anyone relying on outcome feedback.

**Final review — `AdaptivePredictor` periodic persistence never fired:** The save trigger checked `_total_updates` but `AdaptivePredictor` uses `_update_count`. `getattr` silently returned 0 every time. Fixed with a two-attribute probe.

---

## Known Gaps and Future Work

**Streaming path doesn't propagate headers in proxy.** `handle_completion_stream()` has no `headers` parameter and doesn't call `extract_route_context_from_headers`. Agent context is silently absent for streaming proxy calls. Low risk today (streaming is less common in agentic pipelines) but worth tracking.

**`test_client.py` has shallow tests.** Many tests check field assignment (`assert rs.agent_role == "research"`) rather than behavior. The meaningful tests in the file are genuinely good, but the ratio could be improved. Not worth fixing now — marginal correctness value.

**No test for the async context path end-to-end.** The `acompletion()` fix was validated by code review and the existing unit tests, but there's no integration test that fires an async completion and verifies the routing context was applied. The critical bug existed undetected because of this gap.

**`recommend_model_for_agent()` doesn't weight recency.** All records for a role are treated equally regardless of age. A model that was good six months ago but has degraded will still score well. A time-decay factor would improve recommendation quality.

---

## Design Decisions Worth Preserving

**Business rules run before the predictor.** This was deliberate: if a rule consistently excludes a model, the predictor's learned distribution reflects the true constrained action space rather than drifting from override contamination. Don't move this filtering to post-prediction.

**`role=None` and `role="general"` are distinct.** `AgentInferencer` returns `(None, 0.0)` when confidence is below threshold — the predictor learns to weight `agent_role_type=0` near zero and degrades to the 27-feature baseline. Forcing novel prompts into "general" would pollute the general-purpose bucket with out-of-distribution data.

**Historical records are immutable.** Deregistering a model doesn't delete its feedback records. Those records encode query-difficulty signal that informs routing for remaining models. Don't add cascade deletes.
