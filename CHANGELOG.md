# Changelog

All notable changes to RouteSmith will be documented in this file.

## [Unreleased]

### Added
- **Propensity-logging research instrument** (`routesmith.research`): wraps a
  `LinTSRouter` to make each routing decision propensity-logged under an enforced
  exploration floor and exactly replayable from the logged posterior state + a
  per-decision seed. Thompson-sampling propensities are Monte-Carlo estimated via a
  univariate score reduction (no d-dimensional draws), floored into the mixture
  `p(a|x) = (1-ε)·p_TS(a|x) + ε/K` (positivity for off-policy estimators). Includes a
  `DecisionLog` SQLite sink (its own `routing_decisions` table; core storage untouched)
  and `shadow_replay`. Purely additive — production routing hot path is unchanged.
  Substrate for the off-policy-evaluation study (Proposal B v4.1, WU-0.2).

### Changed
- **`routesmith.RouteSmith` / `RoutingMetadata` now load lazily** (PEP 562 `__getattr__`).
  These live in `routesmith.client`, which imports `litellm` (a heavy, native-built
  dependency); loading them eagerly meant importing *any* routesmith submodule pulled in
  the full LLM-calling stack. They now resolve on first access, so lightweight submodules
  — notably `routesmith.research` — import standalone without `litellm`. `from routesmith
  import RouteSmith` and attribute access are unchanged. (Verified: the research + LinTS
  test suites pass with `litellm` absent.)

## [0.9.3] — 2026-07-16

Supersedes 0.9.2 on PyPI (0.9.2 was tagged but never published there — this
release includes everything in it).

### Fixed
- **`pip install "routesmith-llm[all]"` installed a stranger's package.** The
  self-referential `all` extra in `pyproject.toml` still said
  `routesmith[proxy,...]` — the old distribution name, which on PyPI belongs
  to an unrelated project — so every `[all]` install since 0.8.0 silently
  pulled that third-party `routesmith` package into the environment (visible
  as installer warnings like `The package routesmith==0.1.8 does not have an
  extra named 'anthropic'`). Now correctly `routesmith-llm[...]`.
- **PyPI sidebar links pointed at a nonexistent repository**
  (`github.com/routesmith/routesmith`) in `[project.urls]`; same wrong URL in
  `mkdocs.yml`. Both now point at the real repo.

### Added
- `tests/test_packaging.py`: parses `pyproject.toml` and fails CI if any
  dependency (including self-referential extras) resolves to the bare
  `routesmith` PyPI name, if the `all` extra drops a user-facing extra, or if
  project URLs point at the wrong GitHub org.

## [0.9.2] — 2026-07-16

Critical fixes found by actually running the 0.9.1 CLI end-to-end (quickstart →
serve → route a real request) rather than relying on unit tests alone — several
of the flagship one-command flows this project exists for were broken in ways
no existing test caught, because on-disk config generation and config loading
were never tested together.

### Fixed
- **`routesmith serve` crashed on any config from `quickstart`, `models
  refresh`, or `init --provider`** with `Error loading config: 'id'`. Those
  three commands wrote the internal catalog schema (key `model_id`) straight
  into `routesmith.yaml`; the loader requires the on-disk schema (key `id`,
  matching `routesmith.yaml.example`). The flagship `routesmith quickstart`
  one-liner could not get past its own generated config.
- **Every request routed to an Anthropic model failed**, on both
  `/v1/chat/completions` and `/v1/messages`, with
  `litellm.UnsupportedParamsError`. The proxy unconditionally sent
  `frequency_penalty`/`presence_penalty` (even at their 0.0 default) to
  whatever model was selected; Anthropic's API rejects both params outright.
  This is the same class of issue an earlier 0.9.1 fix addressed for
  `temperature`/`top_p` but didn't extend to the penalty params — the single
  most common path (Claude Code or any client → RouteSmith → an Anthropic
  model) was unusable.
- **Tool-calling capability silently dropped** on any catalog-generated
  config: the catalog schema's `supports_tools` key was never translated to
  the loader's `supports_function_calling` key. Currently masked by both
  defaulting to `True`, but relying on that coincidence was fragile — fixed
  with an explicit, tested translation.
- **`routesmith --version` was hardcoded to `0.1.0`**, disconnected from the
  actual package version since the CLI was first scaffolded.
- **`routesmith connect <tool> --verify` crashed with a raw Python traceback**
  on any downstream failure (e.g. a missing/invalid provider API key — the
  most common first-run mistake) instead of the clean pass/fail signal the
  command exists to provide. Now prints the actual upstream error message and
  exits 1.
- **`routesmith models list` displayed a blank Model ID column** against any
  real config, and **`routesmith models refresh`/`models list` subcommands
  documented in every integration guide didn't exist** — only `models
  --refresh`/`--json` flags did. Added the documented `models list` / `models
  refresh` subcommands (flags remain as a back-compat alias) and fixed the
  display to read the correct on-disk key.

### Added
- Round-trip regression tests for `quickstart`, `models refresh`, and `init
  --provider`: each now asserts the generated config actually loads via
  `load_config_file()` and registers tool-capable models — the exact
  generate-then-load path that was broken and that no test previously
  exercised end-to-end.

## [0.9.1] — 2026-07-16

Follow-up fixes to the 0.9.0 integration DX overhaul: the PyPI package name was
still wrong in most docs, and the documented setup paths hadn't caught up to the
new one-command UX.

### Fixed
- **PyPI package name**: every remaining `pip install routesmith` (README, all
  integration guides, quickstart, example scripts, and in-package error/install
  messages) now correctly reads `pip install routesmith-llm` — `routesmith` on
  PyPI is a different, unrelated package.
- **`docs/integrations/claude-code.md`** rewritten around the actual v0.9.0 setup
  (`routesmith run claude` / `routesmith connect claude-code --verify`); removed
  the deprecated "Codex plugin bridge" as the recommended path, the "zero quality
  loss" claim, and the fabricated Before/After dollar table.
- **`docs/integrations/opencode.md`, `codex.md`, `pi.md`, `openclaw.md`** updated
  to lead with `routesmith quickstart` / `routesmith run <tool>` and verify with
  `routesmith connect <tool> --verify` instead of a bare `curl /health` (which
  reports healthy even when requests are silently passing through unrouted).
- **`scripts/check_claims.sh`**: the 0.9.0 PR added the wrong forbidden strings;
  now correctly guards against `"Codex plugin"` and `"zero quality loss"`
  reappearing in docs.
- **Conversation stickiness is now bounded.** The in-memory sticky-model map
  (`RouteSmith._conversation_models`) had no cap or expiry — a long-running
  `routesmith serve` process (the default has no persistent storage configured)
  would grow it without bound over the life of the process. It's now an LRU
  capped at 1,000 conversations with a 24h TTL, enforced independently of
  whether SQLite persistence is configured; the persisted table is bounded the
  same way on every write.

### Added
- **`docs/integrations/hermes.md`**: Hermes was in the `routesmith connect` tool
  list but had no integration guide.
- Tests for sticky-map LRU eviction, TTL expiry, and that sticky (non-routed)
  turns still record feedback to the bandit.

## [0.9.0] — 2026-07-16

Integration DX overhaul — one-command setup for coding agents, provider-aware
model catalogs, and full agentic tool-use support on the Anthropic-native proxy
endpoint.

### Added
- **`routesmith run <tool>`**: launches Claude Code, Codex, or OpenCode with the
  right provider env injected for that session only; starts the proxy first if
  needed. `routesmith serve --daemon` / `status` / `down` for lifecycle
  management.
- **`routesmith connect <tool> [--verify]`**: prints exact setup for Claude Code,
  Codex, OpenCode, OpenClaw, pi, Hermes, OpenAI SDK, and Anthropic SDK;
  `--verify` does a live round-trip and confirms requests are actually being
  routed, not just that the proxy is reachable.
- **`routing.intercept: all`**: routes every proxy request through the bandit
  regardless of the concrete model name the client requested — the previous
  behavior only routed requests explicitly asking for `"auto"`, so real coding-
  tool traffic was silently passed through unrouted.
- **Full tool-use support on `POST /v1/messages`**: tool definitions,
  `tool_use`/`tool_result` blocks, images, and streaming tool-call deltas are
  now translated end-to-end (previously text-only). A protocol-native fast path
  forwards Anthropic-to-Anthropic requests verbatim so prompt caching
  (`cache_control`) and other Anthropic-specific fields survive routing.
  Added `POST /v1/messages/count_tokens`.
- **Provider-aware model catalogs**: `routesmith quickstart`/`init` now detect
  which provider API keys are present (Anthropic, OpenAI, OpenRouter, Groq) and
  build a matching default model pool instead of a hardcoded OpenAI-only one.
- **`routesmith models list` / `models refresh`**: re-sync prices, context
  windows, and available models from providers without losing pinned or
  hand-tuned entries.
- **Automatic conversation stickiness**: routing stays on the same model across
  turns of one conversation (including subagent conversations) even when the
  client never sends a conversation-id header, by fingerprinting the system
  prompt and first user message.

## [0.8.0] — 2026-07-13

First release published to PyPI, as [`routesmith-llm`](https://pypi.org/project/routesmith-llm/)
(the `routesmith` name was taken). The import name is unchanged: `import routesmith`.

### Added
- **PyPI package**: `pip install routesmith-llm`.
- **Anthropic-native proxy endpoint**: `POST /v1/messages`, non-streaming and streaming.
- **Agent-framework examples**: OpenAI SDK, Pydantic AI, LlamaIndex, plus an
  `examples/` directory with a CI smoke harness.
- **Per-project cost stats, decision audit log, per-role policy CLI.**

## [0.5.0-beta] — 2026-06-20

### Added
- **NeuralUCB predictor**: Shallow neural network with UCB exploration bonus for nonlinear reward modeling
- **REINFORCE predictor**: Policy-gradient predictor with entropy-regularized stochastic exploration
- **WarmStart LinUCB predictor**: LinUCB initialized from benchmark quality priors
- **Paper**: ICML-style research paper with benchmark results (compiles to 10-page PDF)
- **Benchmark runner scripts**: `run_linucb_27d.py`, `run_linucb_fast.py` for reproducible experiments
- **Latency micro-benchmark**: Verified <0.5ms P99 routing overhead for 5-arm deployment

### Changed
- **Feature vector**: Simplified from 35-dim to 27-dim (removed 8 context features per paper-validated results)
- **LinUCB API**: Simplified — removed `context`, `reward_override`, `add_arm`, `remove_arm`, `serialize_state`, `load_state`
- **Router**: Added `neural_ucb`, `reinforce`, `warmstart_linucb` predictor types
- **CI**: mypy now enforced (no `|| true` escape), added bandit security scan

### Fixed
- Two conflicting paper versions consolidated into single `main.tex`
- Table symbols fixed for proper LaTeX compilation

## [0.4.4] — 2026-06-19

### Added
- **Trust-but-verify**: Shadow execution that compares routed model against alternatives
- **CLI stats**: `routesmith stats --local --watch` with live updating
- **Dashboard TUI**: `routesmith dashboard` terminal UI for real-time routing analytics

## [0.4.3] — 2026-06-18

### Added
- **Quality polls**: Adaptive sampling for collecting user feedback on model quality
- **`rs.answer_poll()`**: Explicit feedback mechanism to close the quality loop
- **Recommendations engine**: `rs.recommendations()` for proactive model selection advice

## [0.4.2] — 2026-06-17

### Added
- **Auto-registration**: `RouteSmith.with_auto()` zero-config entry point with OpenRouter model discovery
- **Tradeoff parameter**: Per-request `tradeoff` (0-10) controlling cost vs quality
- **RouteSmith explanation**: `routesmith_explanation` field on every response with routing rationale
- **Conversation stickiness**: Model persistence within a conversation scope

## [0.3.0] — 2026-06-16

### Added
- **Enterprise features**: CostModel enum (ON_DEMAND, PROVISIONED, SELF_HOSTED), capacity tracking, provisioned-first routing
- **Compliance routing**: Tag-based filtering (HIPAA, SOC2, PCI) with `required_compliance` parameter
- **Budget enforcement**: FAIL, FALLBACK, QUEUE behaviors when budget exhausted
- **Multi-project isolation**: Per-project cost allocation and stats via `project` parameter on `RouteSmith.__init__()`

## [0.2.0] — 2026-06-15

### Added
- **Production hardening**: Prometheus metrics, Docker image, health/liveness/readiness endpoints
- **Circuit breaker**: Per-model failure detection with automatic backoff
- **Structured logging**: JSON-format log output for observability
- **Comprehensive documentation**: mkdocs site with setup guides, API reference, integration docs
- **Nightly live tests**: Automated smoke tests against real APIs

## [0.1.0] — 2026-04-06

### Added
- **Core routing engine**: Model Registry, DIRECT/CASCADE/PARALLEL/SPECULATIVE strategies
- **Local proxy server**: OpenAI-compatible HTTP API via `routesmith serve`
- **CLI**: `routesmith serve`, `routesmith stats`, `routesmith init`
- **Cost tracking**: Real-time cost estimation with counterfactual comparison
- **Response metadata**: Per-response routing decisions (which model, why, what cost)
- **Quality prediction**: Embedding-based, classifier-based, random forest, and adaptive predictors
- **Semantic cache**: Embedding-based response caching for similar queries
- **Feedback collection**: Quality signal extraction, SQLite storage, predictor online learning
- **LangChain integration**: `ChatRouteSmith(BaseChatModel)` with tool calling and streaming
- **DSPy adapter**: Proxy mode and native `RouteSmithLM`
- **CrewAI adapter**: Proxy mode and native chat model
- **AutoGen integration**: `routesmith_autogen_agents()` proxy-based agent pair
- **Anthropic/OpenClaw integration**: Drop-in replacement for `anthropic.Anthropic`