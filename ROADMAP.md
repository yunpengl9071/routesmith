# RouteSmith Roadmap

**Mission:** OpenRouter's Auto Router (powered by NotDiamond) answers *"which model is best on
average?"* once, at training time, for everyone. RouteSmith answers *"which model is best for
THIS workload?"* continuously, from live feedback — self-hosted, on any provider, including
OpenRouter itself.

**Positioning:** We do not compete with OpenRouter's marketplace. We *use* it as a provider
layer (one key, 400+ models) and replace its static, black-box `openrouter/auto` router with a
transparent, learning one. Our structural advantages that NotDiamond cannot copy:

1. **Online learning from the user's own traffic** (bandit updates from explicit + implicit + judged feedback)
2. **Self-hosted** — prompts never transit a third-party router
3. **Custom model pools, reward functions, and per-agent-role policies**
4. **Transparent decisions** — every routing choice is auditable with per-candidate scores
5. **Budget governance** — hard spend caps, which OpenRouter only offers at account-credit level

---

## Current state (honest)

A full gap audit was performed on 2026-07-06 (anchor commit `a1e5e5c`). Summary:

| Area | Status |
|---|---|
| Bandit core (LinTS / LinUCB, 35-dim features) | ✅ Working |
| SQLite persistence, state snapshots | ✅ Working |
| OpenAI-compatible proxy | ✅ Working (no auth, no feedback endpoint) |
| Framework integrations (LangChain, DSPy, CrewAI, AutoGen, Anthropic) | ✅ Working |
| Offline benchmark harness (MMLU/GSM8K/MBPP) | ✅ Exists, but tests research reimplementations, not product code |
| Cascade / parallel / speculative strategies | ❌ **Selection only — never executed** (single call in `client.py`) |
| Semantic cache | ❌ **Dead code — never instantiated** |
| Daily/hourly budget enforcement | ❌ **Parsed, never enforced** |
| `fallback_model` | ❌ **Parsed, never read** |
| LLM-as-judge | ❌ **Hook only, no implementation** |
| Implicit signals → predictor updates | ❌ **Extracted but never fed to learning** |
| Proxy feedback endpoint | ❌ **Does not exist** — proxy users cannot close the loop |
| Embedding-based query features | ❌ `EmbeddingPredictor` is a TODO stub |
| Metrics export / auth / multi-replica state | ❌ None |

---

## Measurable goals

Every goal has a concrete measurement procedure. A goal is MET only when its command passes.

| ID | Goal | Target | How measured |
|----|------|--------|--------------|
| G1 | Time-to-first-value | Fresh venv → first routed request in ≤ 6 shell commands | `bash scripts/verify_quickstart.sh` exits 0 |
| G2 | Zero dead claims | Every documented feature has runtime behavior | `bash scripts/check_claims.sh` exits 0 |
| G3 | Routing overhead | p50 < 5 ms, p99 < 15 ms per routing decision, 5 candidate models | `pytest tests/test_perf_routing.py -m perf` passes |
| G4 | Learning convergence | ≥ 80% optimal-arm selection within 150 feedback events (synthetic 2-arm workload, seeded) | `pytest tests/test_convergence.py` passes |
| G5 | Product-code benchmark | APGR ≥ 0.55 on MMLU-600 using the REAL `Router` (not research strategies) | `make bench-product` report |
| G6 | E2E loop via HTTP | completion → feedback → posterior update, proven over the proxy | `pytest tests/test_proxy_feedback_e2e.py` passes |
| G7 | Cache effectiveness | Cache-hit path < 10 ms, cost recorded as $0 | `pytest tests/test_cache_wiring.py` passes |
| G8 | Budget safety | Over-limit request rejected before any provider call; proxy returns 429 | `pytest tests/test_budget.py tests/test_proxy_budget.py` passes |
| G9 | Integration coverage | 8 agentic platforms with runnable examples + CI import-smoke tests | `pytest tests/test_examples_smoke.py` passes |
| G10 | Proxy hardening | Bearer auth + `/metrics` endpoint, `/health` open | `pytest tests/test_proxy_auth.py tests/test_proxy_metrics.py` passes |

---

## Phases

Execute in order. Within a phase, tasks are ordered by dependency; independent tasks may be
done in any order. Each task in the phase plans has files, specs, tests, and acceptance criteria.

| Phase | Plan | Theme | Goals unlocked |
|-------|------|-------|----------------|
| 0 | [docs/plans/phase-0-credibility.md](docs/plans/phase-0-credibility.md) | Stop shipping falsehoods: docs drift, fallback, budgets, cache wiring | G1, G2, G7, G8 |
| 1 | [docs/plans/phase-1-learning-loop.md](docs/plans/phase-1-learning-loop.md) | Close the learning loop: proxy feedback, implicit signals, LLM judge | G4, G6 |
| 2 | [docs/plans/phase-2-routing-quality.md](docs/plans/phase-2-routing-quality.md) | Beat NotDiamond on quality: warm-start priors, embeddings, real cascade, product benchmarks | G3, G5 |
| 3 | [docs/plans/phase-3-differentiation.md](docs/plans/phase-3-differentiation.md) | Moats: conversation stickiness, per-role policies, decision audit log | — |
| 4 | [docs/plans/phase-4-production-proxy.md](docs/plans/phase-4-production-proxy.md) | Production proxy: auth, metrics, shared state, real HTTP server | G10 |
| 5 | [docs/plans/phase-5-integrations-dx.md](docs/plans/phase-5-integrations-dx.md) | Ease of use + agentic platforms: examples, Anthropic-native endpoint, quickstart CLI | G1, G9 |

Suggested release mapping: v0.2 = Phase 0 · v0.3 = Phase 1 · v0.4 = Phase 2 · v0.5 = Phase 3 · v0.6 = Phases 4–5.

---

## Execution protocol (READ THIS FIRST if you are an AI agent executing tasks)

1. **Anchors.** File:line references in the plans refer to commit `a1e5e5c`. Lines drift —
   ALWAYS locate code by the quoted snippet first (`grep -n "<snippet>" <file>`), line number
   second. If a snippet cannot be found, STOP and re-read the file before editing; do not guess.
2. **Prerequisite.** PR #22 (`fix/message-features-once`, perf: single message-feature
   extraction per `predict()`) may or may not be merged. Task P2.2 depends on it; if unmerged,
   cherry-pick commit `02e7e02` first.
3. **One task = one commit.** Conventional-commit style, task ID in the body, e.g.
   `feat(budget): enforce rolling spend windows [P0.4]`.
4. **Tests are the contract.** Each task lists test files and named test functions with the key
   assertions. Write the tests exactly as named. A task is DONE only when
   (a) its listed tests pass, and (b) the full suite passes:
   ```bash
   python -m pytest tests/ -q \
     --ignore=tests/manual \
     --ignore=tests/test_anthropic_integration.py \
     --ignore=tests/test_langchain_integration.py \
     --ignore=tests/test_crewai_integration.py \
     --ignore=tests/test_autogen_integration.py \
     --ignore=tests/test_dspy_integration.py
   ```
   (The ignored files need optional extras; run them too if the extras are installed.)
5. **Mock `litellm` in unit tests.** Never make real API calls in tests. Pattern:
   `monkeypatch.setattr("routesmith.client.litellm.completion", fake_completion)`.
   Build fake responses with the helper specified in Task P0.0.
6. **Scope discipline.** Do not refactor beyond the task spec. Do not rename public APIs.
   Do not change a test to make it pass unless the task explicitly says the test is wrong.
7. **Docs move with code.** If a task changes behavior described in README.md or
   docs/TUTORIAL.md, update those lines in the same commit.
8. **When a spec under-determines a choice**, pick the simplest option consistent with the
   acceptance criteria and record the choice in the commit body. Do not block.
