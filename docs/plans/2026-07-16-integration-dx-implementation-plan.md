# Integration DX Overhaul — Implementation & Test Plan

**Date:** 2026-07-16
**Spec:** `docs/plans/2026-07-16-integration-dx-spec.md` (referenced below as "spec §…").
**Audience:** an autonomous **executor agent** (Part A) and an autonomous **tester agent**
(Part B). Both parts are self-contained; no further instructions will be provided.
Requirement IDs (R1–R6) and acceptance criteria (AC-1…AC-15) refer to the spec.

---

## Part A — Executor

### A0. Workflow & ground rules

- **Branch:** create `feature/integration-dx` off `origin/dev`. All implementation
  commits go there. One commit per phase (P1–P6 below), message prefix `feat(dx):` /
  `docs(dx):` matching repo convention (`git log --oneline` for examples).
- **Related branch:** `claude/release-social-visibility-g9t0jx` (unmerged at time of
  writing) fixes the PyPI package name (`routesmith` → `routesmith-llm`) across docs and
  error strings, bumps `__version__` to 0.8.0, and adds a launch kit. If it has merged
  into `dev` by the time you start, you inherit those fixes; if not, do **not** duplicate
  them — your docs changes should still write `routesmith-llm` in any *new or rewritten*
  text (spec R6), and merge conflicts on `docs/integrations/claude-code.md` are expected
  and yours-wins for the sections you rewrite.
- **Do not** modify: `paper/`, `benchmark/`, `docs/plans/` (except nothing), `README.md`
  (out of scope per spec §2 non-goals), predictor/bandit code under
  `src/routesmith/predictor/` and `src/routesmith/strategy/` (except the single decision-
  point unification in P3 if it touches strategy dispatch — keep it minimal).
- **Version:** bump `pyproject.toml` and `src/routesmith/__init__.py` to `0.9.0` in the
  final phase (this is a feature release).
- **After each phase:** run the full local gate (see §CI below) before committing.

### CI & local gate (run before every commit)

The GitHub workflow is `.github/workflows/test.yml`. Reproduce locally:

```bash
PERF_MULTIPLIER=3 python -m pytest tests/ -q \
  --ignore=tests/manual \
  --ignore=tests/test_anthropic_integration.py \
  --ignore=tests/test_langchain_integration.py \
  --ignore=tests/test_crewai_integration.py \
  --ignore=tests/test_autogen_integration.py \
  --ignore=tests/test_dspy_integration.py
bash scripts/check_claims.sh
```

CI also enforces **mypy** (no escape hatch) and runs **ruff** and **bandit** — run
`mypy src/routesmith` and `ruff check src tests` locally; keep both clean. Baseline as of
this plan: 709 passed, 8 skipped.

`scripts/check_claims.sh` greps `README.md docs/ --include='*.md'`, excluding lines
matching `ROADMAP` or `plans/`, for forbidden strings. Current list: `27-dimensional`,
`LinTS-27d`, `rs.complete(`, `response.request_id`. You will extend it in P6 (spec R6.5).
Never introduce those strings in docs you touch.

### Codebase map (verified anchors)

| Concern | Where |
|---|---|
| Proxy request handling, `AUTO_MODELS`, routing-vs-passthrough decision | `src/routesmith/proxy/handler.py` (`AUTO_MODELS` at :149; decision at :180 non-stream, :236 stream; `ChatCompletionRequest.extra_kwargs` passes unknown fields through) |
| Raw HTTP server, Anthropic path wiring | `src/routesmith/proxy/server.py` (`_handle_completion` :262; Anthropic translation call :300–:323) |
| Anthropic ⇄ OpenAI translation + SSE builder | `src/routesmith/proxy/anthropic_compat.py` (text-only guard at :75; `AnthropicSSEStream` :141) |
| Core client, litellm dispatch, capability detection | `src/routesmith/client.py` (`_detect_required_capabilities` :299 — `tools` → `tool_calling`, images → `vision`; `litellm.completion` calls :627, :666, :724; async :1155) |
| Model registry & capabilities | `src/routesmith/registry/models.py` (`ModelConfig.capabilities` :27; `filter_by_capabilities` :173) |
| OpenRouter live fetch (reuse for refresh) | `src/routesmith/registry/openrouter.py` (`fetch_models`, `_MODELS_URL`) |
| Existing discovery + curated fallback | `src/routesmith/registry/discovery.py` |
| Quality priors (packaged JSON) | `src/routesmith/registry/priors.py`, `src/routesmith/registry/data/default_priors.json` |
| CLI wiring | `src/routesmith/cli/main.py` (argparse subparsers: init :29, serve :50, quickstart :88, openclaw :106, stats :125, audit :164, roles :200, dashboard :243, evaluate :257) |
| Quickstart defaults to replace | `src/routesmith/cli/quickstart.py` (`_detect_provider`, `_DEFAULT_MODELS`) |
| YAML config load/save | `src/routesmith/cli/yaml_loader.py`, `src/routesmith/config.py` (`RouteSmithConfig` :131) |
| Test fake for litellm responses | `tests/helpers.py::fake_response` (SimpleNamespace that quacks like `ModelResponse`; extend it — see B2) |
| Existing endpoint tests | `tests/test_proxy.py` (46 tests), `tests/test_anthropic_endpoint.py` (16), `tests/test_cli_quickstart.py`, `tests/test_discovery.py` |

### P1 — Provider-aware catalog (spec R1) — foundations first

**New files**
- `src/routesmith/registry/catalog.py`:
  - `detect_providers() -> list[str]`
  - `load_catalog(provider: str) -> dict` (via `importlib.resources`, same pattern as
    `priors.load_default_priors`)
  - `build_default_pool(providers: list[str] | None = None) -> list[dict]` — returns
    dicts shaped for the YAML `models:` list; implements the dedupe rule (spec R1.3):
    canonical key = `model_id.split("/")[-1]`, direct provider beats `openrouter/`.
  - `NoProviderDetectedError(RouteSmithError)` — add to `src/routesmith/exceptions.py`
    following existing exception style.
- `src/routesmith/registry/data/catalogs/{anthropic,openai,openrouter,groq}.json` —
  schema per spec R1.2. Content requirements:
  - anthropic: Haiku/Sonnet/Opus current tiers, all `default: true`, bare `claude-*`
    LiteLLM ids, `supports_tools: true`, `supports_vision: true`, 200k context.
  - openai: `gpt-4o-mini` + `gpt-4o` as defaults plus a small-reasoning entry; bare ids.
  - openrouter: 4–6 spread-across-vendors defaults, `openrouter/<vendor>/<model>` ids.
  - groq: 2 defaults with `groq/` prefix (see `routesmith.yaml.example` for current ids).
  - Prices: fill from providers' public pricing pages at implementation time; these are
    curated values and the refresh path (P2) is the correction mechanism. Quality scores:
    take from `default_priors.json` via `lookup_prior` when present, else set a value
    consistent with neighboring models in `routesmith.yaml.example`.
- Ensure package data ships: check `pyproject.toml` `[tool.setuptools.package-data]` (or
  equivalent) already includes `registry/data/*.json`; extend the glob to
  `registry/data/**/*.json` if needed. Verify with a wheel build:
  `python -m build --wheel && unzip -l dist/*.whl | grep catalogs`.

**Modified files**
- `src/routesmith/cli/quickstart.py`: replace `_DEFAULT_MODELS`/`_detect_provider` usage
  with `catalog.build_default_pool()`; add `--provider` (append action); write
  `catalog:` stamp and `routing.intercept: all` (field exists after P3 — in P1 write the
  stamp only, add the intercept line in P3's commit) into generated YAML.
- `src/routesmith/cli/init.py`: same for its non-interactive path; interactive
  OpenRouter picker untouched.

**Tests (new `tests/test_catalog.py` + updates)**
- `detect_providers` under monkeypatched env combinations (none/one/many) — AC-3 partial.
- `build_default_pool` dedupe: anthropic+openrouter env → no `openrouter/anthropic/*`
  entries (AC-2); each single-provider env → single-provider pool (AC-1 partial).
- Catalog JSON schema validation test: every packaged catalog file parses, every entry
  has all required keys, LiteLLM id shape per provider (regex per prefix rule in spec
  R2.3), and at least one `default: true` entry per file.
- `tests/test_cli_quickstart.py`: update per spec §9.3 — assert provider-matched pool and
  `catalog.refreshed_at` presence (AC-1).

### P2 — Refreshable model list (spec R2)

**New files**
- `src/routesmith/cli/models_cmd.py`: `models list` and `models refresh` per spec R2.2–
  R2.3. Refresh engine:
  - `infer_provider(model_id: str) -> str | None` (prefix rules, spec R2.3).
  - Per-provider fetchers: reuse `registry/openrouter.py::fetch_models` for OpenRouter;
    new thin `fetch_openai_model_ids()` (`GET {OPENAI_BASE_URL|https://api.openai.com}/v1/models`,
    bearer key) and `fetch_anthropic_model_ids()` (`GET https://api.anthropic.com/v1/models`,
    `x-api-key` + `anthropic-version` headers) returning id sets; both with 10s timeout
    and the same `ssl` handling as `openrouter.py`. No new runtime dependency — use
    `urllib.request` like `openrouter.py` does (note `discovery.py` imports `requests`
    lazily; do not copy that pattern).
  - Update rules exactly as spec R2.3 (pinned, quality_score untouched, warn-don't-delete,
    `--add-new`, `--dry-run`, atomic temp-file+rename write, `refreshed_at` stamp,
    `sort_keys=False`).
- Wire into `src/routesmith/cli/main.py` as a subparser with nested `list|refresh`.

**Modified files**
- `src/routesmith/cli/serve.py`: single INFO staleness line per spec R2.4.

**Tests (new `tests/test_models_refresh.py`)**
- All of AC-4 and AC-5 with monkeypatched fetchers (no network in unit tests): stale
  price updated, pinned untouched, vanished warned, dry-run leaves file identical
  (byte-compare), atomic write (patch `os.replace` to raise after temp write; original
  file must be intact), offline fallback exit codes.
- Staleness hint: serve startup with old/absent `refreshed_at` logs exactly one INFO line
  (use `caplog`).

### P3 — Intercept routing (spec R3)

**Modified files**
- `src/routesmith/config.py`: add `intercept: str = "auto"` and
  `passthrough_models: list[str]` to the routing config dataclass; validate value in
  `{"all","auto"}`.
- `src/routesmith/cli/yaml_loader.py`: parse the new fields + `catalog` block +
  per-model `pinned` (tolerate absence).
- `src/routesmith/proxy/handler.py`: replace the two inline `AUTO_MODELS` checks (:180,
  :236) with one decision function
  `resolve_routing(request, headers, config) -> RoutingDecision` returning
  `(route: bool, model_override: str | None, passthrough_reason: str | None)`;
  implement escape hatches and counters per spec R3.3–R3.4; attach
  `routesmith_metadata` (spec R5.5) to OpenAI-format responses (non-stream: top-level
  key; stream: include in the final chunk before `[DONE]`).
- `src/routesmith/proxy/server.py`: route the Anthropic path (:300–:323) through the
  same `resolve_routing` (spec R3.5); expose counters in the `/v1/stats` payload
  (handler already owns stats surface at :288).
- `src/routesmith/cli/stats.py`: render the two counters + by-reason breakdown.
- `src/routesmith/cli/quickstart.py` + `init.py`: now write `routing.intercept: all`
  (deferred from P1).

**Tests (extend `tests/test_proxy.py`, new `tests/test_intercept.py`)**
- Full AC-6 and AC-7 matrix on both endpoints: intercept all/auto × concrete-model/auto ×
  header/passthrough-list/unregistered. Use `tests/helpers.py::fake_response` with
  monkeypatched `litellm` as the existing proxy tests do.
- `requested_model` lands in audit log (see `cli/audit.py` for the read side) and in
  `routesmith_metadata`.
- Passthrough-warning log after 5 consecutive passthroughs under `auto` (caplog).
- Regression: every pre-existing `tests/test_proxy.py` case passes unmodified (default
  `auto` preserves behavior; the only allowed edit is response-shape assertions per spec
  §9.4).

### P4 — `/v1/messages` agent support (spec R4)

**Modified files**
- `src/routesmith/proxy/anthropic_compat.py`: implement the full translation table (spec
  R4.1–R4.2). Suggested structure: `_convert_tools`, `_convert_tool_choice`,
  `_convert_content_blocks(role, blocks) -> list[openai_messages]`,
  `_tool_calls_to_blocks`. Keep `anthropic_to_internal` / `internal_to_anthropic`
  signatures; they now also return/accept tool fields via the kwargs dict (add
  `kwargs["tools"]`, `kwargs["tool_choice"]`).
- `AnthropicSSEStream`: tool-use streaming per spec R4.3. The current implementation
  collects chunks then emits (`iter_chunks(chunks: list)`); preserve that call shape but
  handle interleaved text/tool blocks with correct `index` bookkeeping.
- `src/routesmith/proxy/server.py`: 400s from translation use the Anthropic error
  envelope (spec R4.4); confirm `tools` in kwargs reaches `RouteSmith.acompletion` so
  `_detect_required_capabilities` (client.py:299) filters the pool; a pool with no
  qualifying model must surface as a 4xx Anthropic-envelope error, not a 500 (AC-10 —
  check what `routesmith.acompletion` raises for an empty candidate set, see
  `src/routesmith/exceptions.py`, and map it).

**Also in this phase (spec R4.5/R4.6):**
- Protocol-native fast path in `proxy/server.py`: selected model is `claude-*` and
  inbound is `/v1/messages` → forward original body (model swapped, RouteSmith headers
  stripped, `anthropic-beta` headers forwarded) and relay the raw response/SSE; routing
  still happens first (the fast path is about *transport*, not selection). Audit records
  `cache_control_stripped` on crossings.
- `POST /v1/messages/count_tokens` route: verbatim forward to Anthropic when
  `ANTHROPIC_API_KEY` is set, chars/4 estimate otherwise.

**Tests (extend `tests/test_anthropic_endpoint.py`, new `tests/test_anthropic_tools.py`)**
- AC-20: cache_control byte-preservation on native path; stripped+audited on crossing;
  count_tokens both modes, never 404.
- AC-8 round-trip (non-streaming), asserting the exact downstream message shapes the mock
  received.
- AC-9 SSE grammar for tool deltas; text+tool interleave; malformed `arguments` JSON →
  `input: {}` + warning.
- AC-10: capability filtering (vision/tool_calling), unsupported block → 400 envelope,
  no-qualifying-model → 4xx envelope.
- `thinking` block dropped silently; text-only requests byte-compatible with previous
  responses modulo metadata (spec §5).
- Update the ≤16 existing tests asserting text-only `ValueError` per spec §9.1 (comment
  each edit with `# changed per 2026-07-16 integration-dx spec §9.1`).

### P5 — `routesmith connect` (spec R5)

**New files**
- `src/routesmith/cli/connect.py`: tool table exactly per spec R5.2; `--apply` per R5.3
  (temp-`$HOME`-safe: resolve `Path.home()` at call time, not import time); `--verify`
  per R5.4 using `urllib.request` against `--url`; `--proxy-api-key` /
  `ROUTESMITH_API_KEY` support (spec §6).
- Wire subparser in `cli/main.py`; make `connect openclaw` delegate to the existing
  generator in `cli/openclaw.py` (import and call, don't shell out).

**Tests (new `tests/test_cli_connect.py`)**
- Snippet emission for all eight tools contains the `--url` value and never contains a
  real key value (AC-11, spec §6).
- `--apply` for codex under monkeypatched `HOME`: writes, refuses overwrite, `--yes`
  overwrites.
- `--verify` against an in-process proxy with mocked litellm: exit 0 + routed model
  printed under `intercept: all`; exit 1 + fix text under `auto` (AC-11). Follow the
  in-process server pattern used by `tests/test_proxy_feedback_e2e.py`.

### P5b — Proxy lifecycle + auto-stickiness (spec R7, R8)

**New files**
- `src/routesmith/cli/run.py`: `routesmith run <command> [args...]` per spec R7.1 —
  family detection by command basename (`claude` → Anthropic env; everything else →
  OpenAI env; `--family` override), config resolution order, daemon-ensure, then
  `os.execvpe`. Windows: fall back to `subprocess` + exit-code propagation when `execvpe`
  is unavailable.
- Daemon primitives in `src/routesmith/cli/serve.py` (or a small `daemon.py` helper):
  `--daemon` flag (detach, pidfile + logfile under `~/.routesmith/`), plus `status` and
  `down` subcommands per spec R7.2, including stale-pidfile cleanup (pid alive AND
  `/health` responding).

**Modified files**
- `src/routesmith/proxy/handler.py`: when `routing.sticky == "auto"` and no
  `x-routesmith-conversation-id` header, compute the fingerprint per spec R8.1 and pass
  it as `conversation_id` in the `RouteContext` (the header path at `handler.py:23`
  already plumbs this — reuse it, both endpoints).
- `src/routesmith/client.py`: cap `_conversation_models` (:122) with an LRU + 24h TTL per
  spec R8.2 (a small ordered-dict helper is fine; no new dependency).
- `src/routesmith/config.py` / `cli/yaml_loader.py`: `routing.sticky` field, default
  `"header"`; quickstart/init write `"auto"`.
- `cli/connect.py`: claude-code output gains the `ANTHROPIC_AUTH_TOKEN` line, the
  OAuth/subscription caveat, and the `routesmith run claude` footer (spec R5.2 table, as
  amended).

**Tests (new `tests/test_cli_run.py`, `tests/test_sticky_auto.py`)**
- AC-16: stub child script dumps env → assert injected vars, unmodified parent env, exit
  code propagation; no-config path exits 1 naming quickstart.
- AC-17: daemon lifecycle under temp `$HOME`; stale pidfile (write a dead pid) cleaned by
  both `status` and `run`.
- AC-18: same-fingerprint turn 1/turn 2 → same model + stickiness routing_reason; changed
  first message → independent; header overrides fingerprint. Both endpoints.
- AC-19: LRU eviction at 1,000; TTL expiry with injected clock; sticky turns still record
  outcomes (assert the feedback path is hit, mirroring the header-based stickiness tests
  if present — see `tests/test_conversation_tracker.py`).

### P6 — Docs + claims guard + version (spec R6)

- Rewrite `docs/integrations/claude-code.md`; create `docs/integrations/hermes.md`;
  update all eight guides to lead with `connect` and verify with `connect --verify`
  (spec R6.1–R6.3).
- Strip fabricated dollar tables and "zero quality loss" from `docs/integrations/*`
  (R6.4); where savings are cited, use the paper numbers with the "paper experiments"
  label.
- Append `check_absent "Codex plugin"` and `check_absent "zero quality loss"` to
  `scripts/check_claims.sh` (R6.5) — then run it; it must pass.
- `docs/quickstart.md`: add refresh section (R6.6) and make `routesmith run <tool>` the
  headline daily workflow; document `serve --daemon` / `status` / `down` and
  `routing.sticky` in `docs/cli.md`.
- Bump version to `0.9.0` (`pyproject.toml` + `__init__.py`); add a `## [0.9.0]`
  CHANGELOG entry summarizing R1–R6 in the established format.

### Executor definition of done

1. AC-1 … AC-14 and AC-16 … AC-20 all covered by passing automated tests.
2. Local gate (§CI) fully green: pytest (existing 709 + new), check_claims, mypy, ruff.
3. `python -m build --wheel` succeeds and the wheel contains `registry/data/catalogs/*`.
4. Seven commits (P1–P5, P5b, P6) on `feature/integration-dx`, pushed with
   `git push -u origin feature/integration-dx`.
5. Do **not** open a PR and do not merge; the tester agent goes next.

---

## Part B — Tester agent

You verify the executor's work **independently** on branch `feature/integration-dx`.
Do not fix implementation bugs yourself; produce a findings report (see B5).

### B1. Environment

```bash
git fetch origin feature/integration-dx && git checkout feature/integration-dx
pip install -e ".[proxy,all]" || pip install -e ".[proxy]"
```

No provider keys are required for B2–B4 (mocks only). If `OPENROUTER_API_KEY` is present,
also run B4-live.

### B2. Gate reproduction

Run the exact CI gate from Part A §CI. Record: total passed/failed/skipped vs. the
709/8 baseline, plus mypy/ruff/check_claims status. Any pre-existing test that now fails
and is **not** enumerated in spec §9 is automatically a **critical finding**.

### B3. Acceptance-criteria audit

For each of AC-1 … AC-14 and AC-16 … AC-20: identify the test(s) that cover it (by file::test name), run
them in isolation, and mark COVERED / PARTIAL / MISSING. An AC with no covering test is a
**major finding** even if the feature "looks implemented."

### B4. Adversarial scenarios (beyond the executor's tests)

Run each; mocks per `tests/helpers.py`:

1. **Config back-compat:** load `routesmith.yaml.example` (which has none of the new
   fields) through the yaml loader and start the proxy — must behave exactly as `auto`
   intercept, no warnings other than the staleness hint.
2. **Claude Code first-contact simulation:** send to `/v1/messages` a request modeled on
   real Claude Code traffic: `system` array with two text blocks, 8 tool definitions,
   `tool_choice {type:"auto"}`, a history containing text → tool_use → tool_result →
   text, `stream: true`, concrete model `claude-sonnet-4-5`, `max_tokens: 8096`. With
   `intercept: all` and a tool-capable mocked pool: assert a grammatically valid SSE
   stream and routed metadata. With a pool where **no** model has `tool_calling`: assert
   4xx Anthropic envelope, not 500 or hang.
3. **Interleaving torture (streaming):** mocked chunks: text delta, tool-call delta,
   text delta, finish `tool_calls` — assert block indices are consistent and every
   `content_block_start` has a matching `content_block_stop`.
4. **Refresh corruption attempt:** run `models refresh` while the YAML contains a
   comment, a pinned model, and an unknown extra field — file must remain loadable
   afterward, pinned entry byte-identical, unknown field preserved or removal explicitly
   noted in output.
5. **Refresh with wrong key:** monkeypatched 401 from a provider — refresh degrades per
   spec R2.3 offline rules and the config file is untouched.
6. **Passthrough counters under concurrency:** fire 20 parallel requests (10 routed, 10
   passthrough-by-header) at an in-process server; `/v1/stats` totals must equal 10/10.
7. **`connect --verify` false-positive check:** with `intercept: auto` and a proxy that
   returns 200 passthrough responses, verify MUST exit 1 (this is the whole point of
   R5.4 — a `curl /health`-equivalent pass is a failure).
8. **No-key quickstart:** with all four env vars unset, `routesmith quickstart --yes`
   exits 1 and names all four env vars (AC-3).
9. **Wheel data check:** `python -m build --wheel`; catalogs present in the wheel;
   `pip install dist/*.whl` into a scratch venv; `routesmith models list` works there
   (catches importlib.resources path bugs that editable installs mask).
10. **`run` wrapper hygiene:** `routesmith run <stub>` where the stub dumps env — parent
    shell env unmodified afterward; stub exit code 42 propagates; running the stub
    *without* the wrapper sees no RouteSmith env (UX corollary: stopped proxy can't brick
    tools).
11. **Subagent fan-out simulation:** against one in-process proxy with `sticky: auto` +
    `intercept: all`, drive three interleaved "conversations" (distinct system prompt +
    first message, multi-turn each, mocked upstream) mimicking a main agent plus two
    subagents — each conversation must be internally sticky (one model throughout) while
    conversations route independently; counters and audit entries must attribute all
    turns correctly.
12. **Stale pidfile recovery:** write a pidfile pointing at a dead pid, then
    `routesmith run <stub>` — must clean up, start the daemon, and proceed (no crash, no
    duplicate daemon).
13. **Cache-economics regression tripwire:** an `/v1/messages` request with
    `cache_control` blocks and a pool containing both `claude-*` and non-Anthropic
    models, forced (via mocked bandit choice) to the Anthropic arm — assert the mocked
    upstream received `cache_control` intact (spec R4.5). If this fails, Claude Code
    users silently lose prompt caching: file it as **critical**.

### B4-live (only if `OPENROUTER_API_KEY` is set)

`routesmith quickstart --yes` → `routesmith serve` (background) → one real completion
with a **concrete** model name → assert `routesmith_metadata.routed == true` → 
`routesmith connect openai-sdk --verify` exits 0 → `routesmith models refresh --dry-run`
exits 0 and prints a summary. Kill the server; scenario must not leave orphan processes.

### B5. Report format

Write `docs/plans/2026-07-16-integration-dx-test-report.md` on the same branch,
commit it (`test(dx): verification report`), push. Structure:

```
## Summary            — PASS / FAIL overall, counts vs baseline
## Gate results       — B2 table
## AC coverage matrix — AC-1..AC-15 × COVERED/PARTIAL/MISSING × test refs
## Adversarial results— B4 scenarios 1–9 (+ B4-live) each PASS/FAIL with repro commands
## Findings           — numbered; severity critical/major/minor; file:line; repro steps
```

Verdict rule: overall PASS requires zero critical findings, zero MISSING ACs, and gate
green. Otherwise FAIL with the findings list — the executor agent picks the report up
from this file path.

---

## Shared appendix — things that will bite you

1. **`discovery.py` swallows exceptions** (`except Exception: pass`) and imports
   `requests` lazily; don't extend it — the new catalog/refresh code lives in
   `catalog.py`/`models_cmd.py` with explicit error handling and `urllib`.
2. **The proxy is a hand-rolled asyncio HTTP server** (`proxy/server.py`), not
   FastAPI/Starlette — no TestClient; follow the in-process pattern in
   `tests/test_proxy_feedback_e2e.py` for E2E tests.
3. **`fake_response` in `tests/helpers.py` has `tool_calls=None` hardcoded** — extend it
   with an optional `tool_calls` parameter (list of SimpleNamespace with
   `id`/`function.name`/`function.arguments`) rather than building a second fake.
4. **check_claims is line-based grep** — a forbidden string in a code comment or test
   fixture under `docs/` fails CI; keep fixtures under `tests/`.
5. **mypy is enforced** — the translation code is dict-heavy; type the block-translation
   helpers with `dict[str, Any]` in/out and keep `TypedDict`s optional, matching the
   existing style in `anthropic_compat.py`.
6. **CI installs GPU-heavy extras** (torch etc.) — never add a dependency to make the
   catalog work; stdlib only for the new fetchers.
7. **Anthropic `max_tokens` is required** on `/v1/messages` (validated at
   `anthropic_compat.py:36`) — keep that validation; Claude Code always sends it.
8. **`routesmith_explanation` already exists on core responses** (see README) — the new
   `routesmith_metadata` must not collide with or replace it; metadata is additive.
