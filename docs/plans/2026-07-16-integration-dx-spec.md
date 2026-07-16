# Integration DX Overhaul — Functional Specification

**Date:** 2026-07-16
**Status:** Approved for implementation
**Audience:** Autonomous executor agent (implementation) and tester agent (verification).
This document plus `2026-07-16-integration-dx-implementation-plan.md` are intended to be
sufficient to complete and verify the work with **no further instructions**. Where a design
decision was open, it has been made here — do not re-litigate decisions, implement them.

**Companion document:** `docs/plans/2026-07-16-integration-dx-implementation-plan.md`
(phasing, file-level changes, test plan, CI constraints, definition of done).

---

## 0. Target UX (the thing every requirement serves)

This is the experience the whole spec exists to produce. When a design question is not
answered explicitly below, resolve it in favor of this narrative.

```bash
# One-time setup (~2 minutes)
pip install "routesmith-llm[proxy]"
routesmith quickstart            # detects the user's API keys, builds a matching pool

# Daily workflow — this is everything the user does
routesmith run claude            # or: routesmith run codex / opencode <args...>
```

The user opens Claude Code (or Codex, or OpenCode) and simply works. The tool sends its
completely normal traffic — concrete model names, tool definitions, streaming. RouteSmith,
invisibly: routes every request to the best model for it, keeps the **same** model within
a conversation so multi-turn agent sessions stay coherent, and enforces budgets. The user
never selects a model, never edits the tool's model setting, and never babysits a server
terminal. RouteSmith is visible in exactly one place: `routesmith stats` showing routing
decisions and savings.

Corollaries that follow from this narrative (binding):

- **Nothing may require the user to change their tool's model configuration.** That is
  what `routing.intercept: all` (R3) exists for.
- **A stopped proxy must never brick the user's tool.** The `routesmith run` wrapper (R7)
  injects env only into the wrapped session — if the user launches the tool normally,
  it works normally.
- **Multi-turn coherence is part of correctness, not a nice-to-have.** Auto conversation
  stickiness (R8) is required because no coding tool sends the
  `x-routesmith-conversation-id` header.
- **Setup must be verifiable in one command** (`connect --verify`, R5), and the
  verification must fail loudly when requests are being silently passed through.

## 1. Problem statement

RouteSmith's promise is "point your AI coding tool at `http://localhost:9119` and stop
thinking about model IDs." An audit on 2026-07-16 found five gaps between that promise and
the shipped behavior. All file/line references below were verified against the repo state
at that date (branch `dev`, after PR #45).

| # | Gap | Evidence |
|---|-----|----------|
| G1 | **Proxy only routes when the client requests model `"auto"`.** Coding tools (Claude Code, Codex, OpenCode, pi, Hermes) send concrete model IDs (`gpt-4o`, `claude-sonnet-4-5`, …) by default, so their requests are silently passed through with **no routing, no savings, and no visible indication**. | `src/routesmith/proxy/handler.py:149` — `AUTO_MODELS = {"auto", "routesmith", "routesmith/auto"}`; `handler.py:180` and `:236` — explicit model bypasses routing. |
| G2 | **Default model pool ignores which provider the user actually has.** `routesmith quickstart` detects `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` but always writes an OpenAI-only pool (`openai/gpt-4o-mini`, `openai/gpt-4o`). A user with only `ANTHROPIC_API_KEY` gets a config whose every request fails on auth. | `src/routesmith/cli/quickstart.py` — `_detect_provider()` vs. hardcoded `_DEFAULT_MODELS`. |
| G3 | **Model lists are static.** There is no way to refresh prices/context windows/available models after config generation; prices drift and stale configs mis-route. Only OpenRouter discovery exists (`routesmith init`), and only at generation time. | `src/routesmith/registry/openrouter.py`, `src/routesmith/registry/discovery.py`, `src/routesmith/cli/init.py`. |
| G4 | **The Anthropic-native `/v1/messages` endpoint is text-only.** It raises `ValueError` on any non-text content block and never forwards tool definitions, so Claude Code (tool-driven from its first request) and any agentic Anthropic-SDK client cannot use the natural `ANTHROPIC_BASE_URL` setup. Docs work around this with a convoluted "Codex plugin bridge". | `src/routesmith/proxy/anthropic_compat.py:75` — "text-only supported in v1"; no `tools` in forwarded kwargs (`anthropic_to_internal`, lines 84–99). |
| G5 | **Per-tool setup is bespoke and partly unverifiable.** Only `routesmith openclaw-config` exists as a generator. Each integration doc has a different manual procedure; verification is `curl /health`, which passes even when every request is passed through unrouted (G1). Hermes has no integration path at all. | `src/routesmith/cli/main.py` subcommands; `docs/integrations/*.md`. |

## 2. Goals

1. **Provider-aware model catalog**: the default model pool is derived from the LLM
   provider(s) the user actually has credentials for.
2. **Refreshable catalog**: a first-class CLI verb re-syncs model lists, pricing, and
   context windows from providers, without clobbering user customizations.
3. **Zero-cumbersome-setup with the named clients**: Claude Code, Codex, OpenCode,
   OpenClaw, pi, Hermes, OpenAI SDK, Anthropic SDK each connect with at most one command
   plus one env var / config snippet, and **routing actually happens** for their real
   (tool-using, concrete-model-name) traffic.
4. **No regression of core functionality**: routing/bandit learning, cascade/parallel/
   speculative strategies, semantic cache, budget enforcement, per-project stats, audit
   log, feedback, circuit breaker, Prometheus metrics, and the existing public Python API
   all keep working. The full existing test suite keeps passing (see §9 for the short list
   of intentionally-changed behaviors).

### Non-goals

- No new routing algorithms or predictor changes.
- No hosted/telemetry service; everything stays local/self-hosted.
- No support promises for clients beyond those listed in Goal 3.
- No change to the PyPI package name (`routesmith-llm`) or import name (`routesmith`).
- Marketing/README copy is out of scope except where §R5 explicitly touches
  `docs/integrations/`.

---

## 3. Requirements

### R1 — Provider-aware model catalog

**R1.1 Provider detection.** New function `detect_providers() -> list[str]` in
`src/routesmith/registry/catalog.py`. It returns every provider whose key is present, in
this canonical order: `["anthropic", "openai", "openrouter", "groq"]`, based on env vars
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`. (Order is
cosmetic — all detected providers contribute to the pool.)

**R1.2 Packaged per-provider catalogs.** New packaged data files
`src/routesmith/registry/data/catalogs/<provider>.json`, one per provider above. Schema:

```json
{
  "provider": "anthropic",
  "generated_at": "2026-07-16",
  "models": [
    {
      "model_id": "claude-haiku-4-5-20251001",
      "display_name": "Claude Haiku 4.5",
      "cost_per_1k_input": 0.001,
      "cost_per_1k_output": 0.005,
      "quality_score": 0.80,
      "context_window": 200000,
      "latency_p50_ms": 300,
      "supports_tools": true,
      "supports_vision": true,
      "default": true
    }
  ]
}
```

- `model_id` must be a valid **LiteLLM** identifier (bare `claude-*` for Anthropic,
  bare `gpt-*`/`o*` for OpenAI, `openrouter/<vendor>/<model>` for OpenRouter,
  `groq/<model>` for Groq) — dispatch happens via `litellm.completion(model=model_id)`
  (`src/routesmith/client.py:627` et al.).
- `default: true` marks the 2–4 models per provider included in a generated config;
  others are available to `models add` / `--all`.
- Populate each catalog with current public pricing at implementation time. Quality
  scores: reuse `src/routesmith/registry/data/default_priors.json` via
  `priors.lookup_prior()` where a prior exists; otherwise use the judgment values in the
  catalog file. Anthropic catalog must include at least Haiku, Sonnet, and Opus tiers so
  the "route across Anthropic's own tiers" configuration works out of the box.

**R1.3 Pool construction.** New function
`build_default_pool(providers: list[str] | None = None) -> list[dict]`:
- `providers=None` → use `detect_providers()`.
- Merge `default: true` entries from each detected provider's catalog.
- **Dedupe rule:** canonicalize by stripping any provider prefix from `model_id`
  (`openrouter/anthropic/claude-x` → `claude-x`); if a direct-provider entry and an
  OpenRouter entry collide, keep the **direct provider** entry (cheaper, no middleman).
- Zero providers detected → raise a typed error `NoProviderDetectedError` whose message
  lists the four supported env vars. CLI catches it and prints setup instructions,
  exit code 1.

**R1.4 Generation uses the pool.** `routesmith quickstart` and `routesmith init`
(non-interactive path) call `build_default_pool()` instead of the hardcoded
`_DEFAULT_MODELS`. Both gain `--provider <name>` (repeatable) to force provider selection
regardless of env. The interactive OpenRouter picker in `init` is unchanged. Generated
configs get a new top-level stamp:

```yaml
catalog:
  refreshed_at: "2026-07-16T12:00:00Z"
  providers: [anthropic]
```

### R2 — Refreshable model list

**R2.1 CLI surface.** New subcommand group in `src/routesmith/cli/main.py`:

```
routesmith models list   [--config routesmith.yaml]
routesmith models refresh [--config routesmith.yaml] [--dry-run] [--add-new] [--yes]
```

**R2.2 `models list`** prints a table of the config's registered models: id, input/output
cost, quality, context window, capabilities, and a staleness note if
`catalog.refreshed_at` is older than 30 days or absent.

**R2.3 `models refresh`** re-syncs each model entry from its provider:
- Provider inference per entry: `openrouter/` prefix → OpenRouter; `groq/` prefix → Groq;
  `claude*` → Anthropic; `gpt*`/`o1*`/`o3*`/`o4*` → OpenAI; anything else → skipped with a
  notice.
- **Sources of truth:**
  - OpenRouter: `GET https://openrouter.ai/api/v1/models` (already implemented in
    `registry/openrouter.py::fetch_models`; reuse it) — provides pricing + context window.
  - OpenAI: `GET /v1/models` (existence only; API returns no pricing) + packaged catalog
    for pricing/context.
  - Anthropic: `GET https://api.anthropic.com/v1/models` (existence, requires
    `x-api-key` header) + packaged catalog for pricing/context.
  - Groq: packaged catalog only (no reliable public pricing endpoint) — refresh updates
    from the packaged file.
- **Update rules (deterministic):**
  1. An entry with `pinned: true` is never modified (new optional per-model YAML field).
  2. Otherwise refresh may update: `cost_per_1k_input`, `cost_per_1k_output`,
     `context_window`, `latency_p50_ms`.
  3. `quality_score` is **never** touched by refresh (it is user/bandit territory).
  4. A model no longer listed upstream → print a `WARNING: <id> not found upstream`
     line; do not remove it.
  5. `--add-new` additionally appends upstream models marked `default: true` in the
     packaged catalog that are missing from the config.
  6. Default is to apply changes and print a unified summary of old→new values;
     `--dry-run` prints the summary without writing. If the file would change and stdin
     is a TTY, confirm before writing unless `--yes`.
  7. On success, update `catalog.refreshed_at`.
- **Offline behavior:** any network failure per provider degrades to the packaged catalog
  values with a one-line notice; exit code stays 0 unless *nothing* could be refreshed
  (then 2). Never leave a half-written YAML file: write to a temp file and atomically
  rename.
- **YAML fidelity:** comments in user configs may be lost on rewrite — acceptable, but the
  refresh summary must say so the first time (`note: comments in routesmith.yaml are not
  preserved`). Key order should be kept stable (dump with `sort_keys=False`).

**R2.4 Staleness hint at serve time.** `routesmith serve` logs exactly one INFO line at
startup if `catalog.refreshed_at` is absent or >30 days old:
`model catalog last refreshed <date|never> — run 'routesmith models refresh'`.
Additionally, if `routing.intercept` is `"auto"` or absent (legacy configs) and the
config was *generated* by `quickstart`/`init` rather than hand-written, log one INFO:
`routing.intercept is 'auto' — set routing.intercept: all to route all requests,
not just 'auto' model names`. Hand-written configs are detected by the absence of a
`catalog` block (they never had one); skip the hint for those.

### R3 — Intercept routing (fixes G1)

**R3.1 Config.** New field under `routing:` in YAML and `RouteSmithConfig`:

```yaml
routing:
  intercept: all        # "all" | "auto"  (default: "auto")
  passthrough_models: []  # optional list of model ids never intercepted
```

- `"auto"` (default) = current behavior: only `AUTO_MODELS` names are routed. Library
  and existing-config back-compat is preserved by this default.
- `"all"` = every proxy request is routed through the bandit **regardless of the
  requested model name**, except the escape hatches in R3.3.
- `routesmith quickstart` and `routesmith init` write `intercept: all` into generated
  configs — new users get the just-works behavior; existing configs keep old semantics.

**R3.2 Semantics of `intercept: all`.** The requested model name is treated as metadata,
not a constraint: the router selects from the full registered pool (capability filtering
per `client.py::_detect_required_capabilities` still applies — tool-calling requests only
route to `tool_calling`-capable models, vision to `vision`). The originally requested
model id is recorded as `requested_model` in the decision audit log and in
`routesmith_metadata` on the response.

**R3.3 Escape hatches (both endpoints):**
1. Request header `X-RouteSmith-Passthrough: true` → forward verbatim to the named model.
2. Requested model listed in `routing.passthrough_models` → forward verbatim.
3. Requested model **not registered** in the pool → forward verbatim (do not 404; the
   model may be valid upstream), and count it (R3.4).
   Exception: `AUTO_MODELS` names always route.

**R3.4 Visibility.** The handler maintains counters `routed_requests` and
`passthrough_requests` (with a `by_reason` breakdown: `explicit_header`,
`passthrough_list`, `unregistered_model`, `intercept_auto`). Counters are shared between
streaming and non‑streaming code paths — increment in the single `resolve_routing`
function, not in each path separately. Exposed in `GET /v1/stats` JSON and in
`routesmith stats` output. When `intercept: auto` and ≥5 consecutive passthroughs occur,
log one WARNING:
`N requests passed through unrouted — set routing.intercept: all to route them`.

**R3.5 Both endpoints.** Interception applies identically to `/v1/chat/completions` and
`/v1/messages` (the Anthropic path in `proxy/server.py:300–323` must flow through the same
decision point — unify, don't duplicate).

### R4 — Full agent support on `/v1/messages` (fixes G4)

Rewrite `src/routesmith/proxy/anthropic_compat.py` from "text-only v1" to full
bidirectional translation. Target: **Claude Code works end-to-end against
`ANTHROPIC_BASE_URL=http://localhost:9119`** including tool use and streaming.

**R4.1 Request translation (Anthropic → OpenAI-internal):**

| Anthropic input | OpenAI-internal output |
|---|---|
| `tools: [{name, description, input_schema}]` | `tools: [{type:"function", function:{name, description, parameters:input_schema}}]` |
| `tool_choice: {type:"auto"}` / `{type:"any"}` / `{type:"tool", name}` | `tool_choice: "auto"` / `"required"` / `{type:"function", function:{name}}` |
| assistant message w/ `tool_use` block `{id, name, input}` | assistant message w/ `tool_calls: [{id, type:"function", function:{name, arguments: json.dumps(input)}}]` |
| user message w/ `tool_result` block `{tool_use_id, content}` | `{role:"tool", tool_call_id, content:<flattened text>}` message (one per block; a user message mixing `tool_result` and `text` blocks splits into tool message(s) followed by a user text message) |
| `image` block `{source:{type:"base64", media_type, data}}` | content part `{type:"image_url", image_url:{url:"data:<media_type>;base64,<data>"}}` (message content becomes the OpenAI parts-array form) |
| `text` block | as today |
| `thinking`/`redacted_thinking` blocks in assistant history | **dropped silently** (they are not replayable downstream) |
| any other block type | HTTP 400 with Anthropic-style error body `{type:"error", error:{type:"invalid_request_error", message:"unsupported content block type '<t>'"}}` — never a 500 |

Presence of `tools` must trigger the existing `tool_calling` capability filter (verify the
kwarg reaches `RouteSmith.acompletion`); image parts must trigger `vision`.

**R4.2 Response translation (OpenAI-internal → Anthropic):** `tool_calls` on the choice →
`tool_use` content blocks `{type:"tool_use", id, name, input: json.loads(arguments)}`
(malformed JSON arguments → `input: {}` plus a logged warning, never a crash);
`finish_reason: "tool_calls"` → `stop_reason: "tool_use"` (mapping exists in
`_STOP_REASON_MAP`). Mixed text+tool responses emit text block(s) first, then tool_use
blocks, matching Anthropic ordering.

**R4.3 Streaming.** Extend `AnthropicSSEStream` to emit tool-use streams:
`content_block_start` with `content_block: {type:"tool_use", id, name, input:{}}`,
`content_block_delta` events with `{type:"input_json_delta", partial_json}` accumulated
from OpenAI `tool_calls[].function.arguments` deltas, `content_block_stop`, correct block
`index` management when text and tool blocks interleave, and `message_delta` with
`stop_reason:"tool_use"`. Usage in `message_delta` must use real token counts from the
final chunk when the upstream provides them, falling back to the current estimate.

**R4.4 Error shape.** All 4xx from this endpoint use the Anthropic error envelope (see
R4.1 table), because Anthropic SDK clients parse it.

**R4.5 Protocol-native fast path (critical for prompt caching).** When the *selected*
model's provider protocol matches the inbound endpoint's protocol — an `/v1/messages`
request routed to an Anthropic (`claude-*`) model — the proxy must forward the **original
request body** with only the `model` field replaced (RouteSmith headers stripped), and
relay the raw response/SSE untouched. The OpenAI-internal translation (R4.1–R4.3) runs
only on protocol crossings. This preserves everything translation cannot represent:
`cache_control` blocks (Claude Code depends on prompt caching — losing it can cost more
than routing saves), `thinking` blocks, fine-grained tool-choice options, and
`anthropic-beta` headers (forward these to Anthropic targets; drop them on crossings).
On crossings, `cache_control` is stripped silently and the audit entry records
`cache_control_stripped: true`. The symmetric case (OpenAI-format inbound → OpenAI-family
target) is naturally near-lossless but must follow the same forward-verbatim principle
where the internal representation would drop fields.

**R4.6 `POST /v1/messages/count_tokens`.** Claude Code calls this endpoint for context
management; a 404 breaks the session. Implement it: if `ANTHROPIC_API_KEY` is set,
forward the request verbatim to `https://api.anthropic.com/v1/messages/count_tokens` and
relay the response; otherwise return the estimate `{"input_tokens": ceil(total_chars/4)}`
computed over serialized message text. Never 404, never 500 on well-formed input.

### R5 — `routesmith connect <tool>` + honest verification (fixes G5)

**R5.1 CLI.** New file `src/routesmith/cli/connect.py`, subcommand:

```
routesmith connect <tool> [--url http://localhost:9119] [--apply] [--verify] [--yes]
```

`<tool>` ∈ `claude-code | codex | opencode | openclaw | pi | hermes | openai-sdk |
anthropic-sdk`. Unknown tool → list valid ones, exit 2.

**R5.2 Default behavior: print.** Emits the exact, copy-pasteable configuration for that
tool against `--url`:

| tool | emitted setup |
|---|---|
| `claude-code` | `export ANTHROPIC_BASE_URL=<url>` **and** `export ANTHROPIC_API_KEY=<proxy --api-key value, or "routesmith" when the proxy runs keyless>` — without an API key Claude Code falls into its OAuth login flow instead of using the base URL. **Implementation note:** verify the exact env var name Claude Code checks for API-key auth (`ANTHROPIC_API_KEY` is the standard Anthropic SDK var; Claude Code may use the same or require `ANTHROPIC_AUTH_TOKEN`). If neither works, document the discovered name in a comment and match it. (+ note: or put both in the `env` block of `~/.claude/settings.json`). Must print the caveat that this applies to API-key usage; subscription (OAuth) Claude Code sessions cannot be re-routed. Requires R4. Recommended footer on output: "or just use: routesmith run claude" (R7). |
| `codex` | `export OPENAI_BASE_URL=<url>/v1` + `~/.codex/config.yaml` snippet defining a custom provider with `base_url` **and `wire_api = "chat"`** — Codex defaults to the OpenAI Responses API, which RouteSmith does not serve; the chat wire API must be selected explicitly |
| `opencode` | JSON `providers.routesmith` snippet with `base_url: <url>/v1` |
| `openclaw` | delegate to the existing generator (`cli/openclaw.py`) — same output as `routesmith openclaw-config` |
| `pi` | same OpenClaw-compatible provider JSON, labeled for pi |
| `hermes` | generic OpenAI-compatible setup: `export OPENAI_BASE_URL=<url>/v1` + note that any Hermes provider config accepting an OpenAI-compatible `base_url` works |
| `openai-sdk` | Python snippet: `OpenAI(base_url="<url>/v1", api_key=os.environ.get("OPENAI_API_KEY", "routesmith"))` |
| `anthropic-sdk` | Python snippet: `Anthropic(base_url="<url>")` |

If the proxy was started with `--api-key`, snippets must include the key placeholder
(`<your --api-key value>`) in the right field per tool. Never print an actual key value.

**R5.3 `--apply`.** Writes the config file only for tools with an unambiguous location:
`codex` (`~/.codex/config.yaml`), `openclaw`/`pi` (`./routesmith-provider.json`). Existing
file → refuse unless `--yes`; on apply, print what was written where. All other tools:
`--apply` prints the snippet and a note that manual placement is required (exit 0).

**R5.4 `--verify`.** Performs a live end-to-end check against `--url`:
0. `GET /health` — if unreachable (connection refused, timeout), print the exact command
   to start the proxy (`routesmith serve --daemon` or `routesmith run`), exit 1.
1. `GET /health` must return ok.
2. Send a minimal completion through the endpoint family the tool uses
   (`/v1/messages` for `claude-code`/`anthropic-sdk`, `/v1/chat/completions` otherwise)
   with a **concrete model name** (not `auto`), e.g. `{"model": "gpt-4o", ...}`.
3. Assert the response carries `routesmith_metadata` indicating the request was
   **routed** (not passed through). If it was passed through, print the exact fix
   (`set routing.intercept: all in routesmith.yaml, restart routesmith serve`) and exit 1.
4. Print the routed-to model and estimated cost from the metadata on success.

This makes "it's configured but silently not routing" impossible to miss — the failure
mode G1 created.

**R5.5 Verification requires a routed marker on responses.** Ensure both endpoints attach
`routesmith_metadata` (already emitted on the Anthropic path, `anthropic_compat.py:137`)
including at minimum: `routed: bool`, `selected_model`, `requested_model`,
`passthrough_reason` (null when routed). Add to the OpenAI-format response as a top-level
`routesmith_metadata` key (OpenAI clients ignore unknown keys).

### R6 — Documentation overhaul

1. **Rewrite `docs/integrations/claude-code.md`** around the direct path:
   `pip install "routesmith-llm[proxy]"` → `routesmith quickstart` →
   `routesmith connect claude-code --verify`. Remove the "Codex plugin bridge" as the
   recommended path (keep OpenRouter-custom-provider as a short "alternative" note or
   delete entirely — implementer's choice; do not keep the Codex-plugin instructions).
2. **New `docs/integrations/hermes.md`** following the structure of `opencode.md`
   (setup → configure → verify → advanced), using the generic OpenAI-compatible path.
3. **All eight integration docs lead with `routesmith connect <tool>`** and use
   `--verify` as the verification step instead of bare `curl /health`.
4. **Remove fabricated numbers:** the Before/After dollar tables (e.g. "$8.40 → $3.20")
   and the phrase "zero quality loss" must be removed from every file under
   `docs/integrations/`. Where a savings claim is wanted, cite the paper's experimental
   results (45% cost savings at 71% accuracy, 5-arm) and label them as paper experiments.
5. **Claims guard:** add `"Codex plugin"` and `"zero quality loss"` to
   `scripts/check_claims.sh` `check_absent` list. Note the script's scope is `README.md
   docs/ --include='*.md'` excluding `ROADMAP` and `plans/`; README currently contains
   "Zero quality loss" (title case) — the grep is case-sensitive, so the lowercase
   pattern will not fail on README, and README copy is out of scope here. Verify
   `bash scripts/check_claims.sh` passes before every commit.
6. `docs/quickstart.md` gains a short "Refreshing your model list" section
   (`routesmith models refresh`).

### R7 — Proxy lifecycle: `routesmith run` and daemon mode (serves UX corollary 2)

The foreground `routesmith serve` terminal is acceptable for debugging, not for the daily
workflow. Two additions:

**R7.1 `routesmith run <command> [args...]`** — the recommended daily entry point:

```
routesmith run claude
routesmith run codex --some-codex-flag
routesmith run -- opencode .
```

Behavior:
1. Ensure the proxy is running (R7.2 `status` check); if not, start it as a daemon with
   the default config resolution (`./routesmith.yaml`, then `~/.routesmith/routesmith.yaml`).
   If no config exists, print the `routesmith quickstart` hint and exit 1 — `run` must not
   silently generate config.
2. Inject the correct env **only into the child process**: for commands recognized as
   Anthropic-family (`claude`) set `ANTHROPIC_BASE_URL` + an API-key env var (same var
   verified in R5.2 as the one Claude Code reads — see that note); for all
   others set `OPENAI_BASE_URL` (+ `OPENAI_API_KEY=routesmith` if unset — some tools
   refuse to start without one). Recognition is by basename of the command with a
   `--family anthropic|openai` override flag. The parent shell env is never modified.
3. `exec` the tool (`os.execvpe` on POSIX), so signals/exit codes pass through and no
   wrapper process lingers.
4. Because env lives only inside the wrapped session, launching the tool *without*
   `routesmith run` uses the provider directly — a stopped proxy can never brick the
   user's tools.

**R7.2 Daemon management.** `routesmith serve --daemon` (double-fork/detach, pidfile +
log file under `~/.routesmith/`), `routesmith status` (running? port? config path? pool
size? counters summary), `routesmith down` (SIGTERM by pidfile, wait, report). `run`
uses these primitives. Stale-pidfile handling: `status`/`run` verify the pid is alive
*and* `/health` responds; otherwise clean up and treat as stopped. Windows: `--daemon`
may fall back to a detached subprocess (`CREATE_NEW_PROCESS_GROUP`); document the
limitation, do not gate the feature on service-manager integration (launchd/systemd
integration is explicitly out of scope for this iteration).

### R8 — Automatic conversation stickiness (serves UX corollary 3)

Conversation stickiness currently activates only via the `x-routesmith-conversation-id`
header (`proxy/handler.py:23` → `client.py:512–515`). No coding tool sends it, so under
`intercept: all` every turn of an agent session would be routed independently — mid-task
model switching, incoherent multi-turn behavior. Fix:

**R8.1 Fingerprinting.** New config:

```yaml
routing:
  sticky: auto     # "auto" | "header" | "off"  (default "header" = current behavior;
                   #  quickstart/init write "auto")
```

Under `auto`, when no conversation header is present the proxy derives
`conversation_id = sha256(normalized(system_prompt) + "\x00" + normalized(first_user_message))[:16]`
where `normalized` = strip whitespace, take first 2,000 chars. Rationale: agent tools
resend the full message history every turn, so this fingerprint is stable across turns of
one session and distinct across sessions. An explicit header always wins. Applies to both
endpoints (`/v1/chat/completions` and `/v1/messages`).

**R8.2 Bounded memory.** The stickiness map (`client.py:122` `_conversation_models` —
currently unbounded) becomes an LRU capped at 1,000 entries with 24h TTL; document both
constants in the config reference as non-configurable defaults for this iteration.

**R8.3 Stickiness vs. learning.** Sticky turns bypass per-request model selection but
must still record outcomes/feedback for the bandit (verify the existing header-based path
already does this — mirror it). The audit log entry for a sticky decision keeps
`routing_reason: "conversation stickiness …"` as today.

---

## 4. Config schema — consolidated delta

```yaml
# NEW — written by quickstart/init, optional in hand-written configs
catalog:
  refreshed_at: "2026-07-16T12:00:00Z"
  providers: [anthropic, openai]

routing:
  strategy: direct            # unchanged
  fallback_model: gpt-4o-mini # unchanged
  intercept: all              # NEW: "all" | "auto" (default "auto")
  passthrough_models: []      # NEW: optional
  sticky: auto                # NEW: "auto" | "header" | "off" (default "header")

models:
  - id: claude-haiku-4-5-20251001
    cost_per_1k_input: 0.001
    cost_per_1k_output: 0.005
    quality_score: 0.80
    context_window: 200000
    pinned: false             # NEW: optional, refresh never touches pinned entries
```

Unknown-field tolerance: the YAML loader (`cli/yaml_loader.py`) must accept configs
without any of the new fields (all defaults preserve current behavior).

## 5. Backward compatibility contract

- Default `routing.intercept` is `"auto"` → existing proxy behavior for existing configs.
- `RouteSmith` Python API signatures are unchanged; new config fields have defaults.
- Existing YAML configs load unchanged.
- `/v1/messages` accepts a strict superset of previously-valid requests; previously-valid
  (text-only) requests produce byte-compatible responses modulo the richer
  `routesmith_metadata`.
- `routesmith init` interactive flow unchanged; only its non-interactive default pool and
  the new `catalog` stamp differ.
- `routesmith openclaw-config` remains (now also reachable as `connect openclaw`).

## 6. Security & privacy constraints

- Never write API key **values** into generated configs or `connect` output; reference
  env vars only.
- `models refresh` sends keys only to the matching provider's official endpoint
  (Anthropic key → api.anthropic.com only, etc.).
- Proxy `--api-key` continues to gate all endpoints, including new verification calls
  (`connect --verify` must accept `--proxy-api-key` or read `ROUTESMITH_API_KEY`).

## 7. Acceptance criteria

Each criterion must be demonstrable by an automated test (unit or integration) unless
marked [manual].

**AC-1 (R1):** With only `ANTHROPIC_API_KEY` set, `routesmith quickstart --yes` produces a
config whose every model is an Anthropic model (incl. Haiku/Sonnet/Opus tiers), with
`routing.intercept: all` and a `catalog.refreshed_at` stamp. Same for OpenAI-only,
OpenRouter-only, Groq-only.
**AC-2 (R1):** With both `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` set, the pool
contains direct Anthropic ids and no `openrouter/anthropic/*` duplicates.
**AC-3 (R1):** With no provider keys, `quickstart` exits 1 with a message listing all four
supported env vars.
**AC-4 (R2):** `models refresh --dry-run` against a config with one pinned model, one
stale-priced model, and one upstream-vanished model: prints planned price update for the
stale one, warning for the vanished one, nothing for the pinned one; file unchanged.
Without `--dry-run` the file updates atomically and `refreshed_at` advances.
**AC-5 (R2):** `models refresh` with all provider endpoints unreachable exits 0 (packaged
catalog fallback) with a notice; a completely-unrefreshable run exits 2. Config file never
left in a corrupt state (test by injecting failure mid-write).
**AC-6 (R3):** With `intercept: all`, a `/v1/chat/completions` request for `"gpt-4o"` is
routed (response `routesmith_metadata.routed == true`, `requested_model == "gpt-4o"`);
with `intercept: auto` the same request is passed through and the passthrough counter
increments. Same pair of assertions on `/v1/messages` with `"claude-sonnet-4-5"`.
**AC-7 (R3):** `X-RouteSmith-Passthrough: true` header and `passthrough_models` entries
bypass routing under `intercept: all`; unregistered model ids pass through with
`passthrough_reason == "unregistered_model"`. `GET /v1/stats` exposes both counters and
the by-reason breakdown.
**AC-8 (R4):** A non-streaming `/v1/messages` request containing `tools`, an assistant
`tool_use` turn, and a user `tool_result` turn round-trips: downstream (mocked litellm)
receives correctly-shaped OpenAI `tools`/`tool_calls`/`role:"tool"` messages; a mocked
`tool_calls` response comes back as `tool_use` blocks with `stop_reason: "tool_use"`.
**AC-9 (R4):** Streaming: mocked OpenAI chunks containing tool-call argument deltas
produce a valid Anthropic SSE sequence (`message_start` → `content_block_start[tool_use]`
→ `input_json_delta`+ → `content_block_stop` → `message_delta[stop_reason=tool_use]` →
`message_stop`), parseable by the `anthropic` Python SDK's stream reassembly (test with
the SDK if it is an available dev dependency, otherwise assert the event grammar
directly).
**AC-10 (R4):** An image block routes only to `vision`-capable models; a `tools` request
routes only to `tool_calling`-capable models; an unsupported block type returns HTTP 400
in the Anthropic error envelope, and a request with *no* pool model satisfying required
capabilities returns a clear 4xx (not a 500).
**AC-11 (R5):** `connect <tool>` for all eight tools prints a snippet containing the
`--url` value; `connect claude-code --verify` against a running proxy (mocked upstream)
with `intercept: all` exits 0 and prints the routed model; with `intercept: auto` exits 1
and prints the `routing.intercept: all` fix. `connect codex --apply` writes
`~/.codex/config.yaml` (under a temp `$HOME` in tests) and refuses to overwrite without
`--yes`.
**AC-12 (R6):** `bash scripts/check_claims.sh` passes; no file under `docs/integrations/`
contains `"zero quality loss"`, `"Codex plugin"`, or a Before/After dollar table;
`docs/integrations/hermes.md` exists; all eight guides reference `routesmith connect`.
**AC-13 (regression):** The entire pre-existing test suite passes unmodified except for
tests listed in §9. CI command: see implementation plan §CI.
**AC-14 (regression):** Perf guard: existing `tests/perf` suite passes with
`PERF_MULTIPLIER=3` (routing overhead budget unchanged).
**AC-16 (R7):** `routesmith run <cmd>` with a running proxy execs the child with the
correct env family injected (Anthropic vars for `claude`, OpenAI vars otherwise) and the
parent environment unmodified; with a stopped proxy and a valid config it daemonizes
first; with no config it exits 1 naming `routesmith quickstart`. Child exit code is
propagated. (Test with a stub child script that dumps its env and exit code.)
**AC-17 (R7):** `serve --daemon` writes a pidfile under `~/.routesmith/` (temp `$HOME` in
tests); `status` reports running with port/config/pool; `down` terminates and cleans up;
a stale pidfile (dead pid) is detected and cleaned by both `status` and `run`.
**AC-18 (R8):** With `sticky: auto`, two `/v1/messages` requests sharing system prompt +
first user message (turn 1 and turn 2 of a simulated agent session) route to the same
model with `routing_reason` indicating stickiness on turn 2; a request with a different
first user message routes independently; an explicit `x-routesmith-conversation-id`
header overrides the fingerprint. Same on `/v1/chat/completions`.
**AC-19 (R8):** The stickiness map evicts beyond 1,000 entries (LRU) and after 24h TTL
(inject a fake clock); sticky turns still record feedback/outcomes for the bandit.
**AC-20 (R4.5/R4.6):** An `/v1/messages` request containing `cache_control` blocks routed
to a `claude-*` model reaches the (mocked) upstream byte-identical except the `model`
field; the same request routed to a non-Anthropic model arrives translated with no
`cache_control` and the audit entry has `cache_control_stripped: true`.
`POST /v1/messages/count_tokens` returns `{"input_tokens": <int>}` both with a mocked
Anthropic upstream and in estimate mode (no key), never 404.
**AC-21 (plumbing gate):** With a hand-written single-model pool and `intercept: all`,
the full simulated Claude Code session (tools, streaming, `cache_control`,
`count_tokens`) and an OpenCode-style chat-completions session complete with responses
equivalent to direct API use (same content path, valid SSE grammar, caching markers
intact via the native fast path). This is the exit gate for the plumbing milestone — see
the implementation plan's "Gate G-M1".
**AC-15 [manual, tester agent]:** End-to-end smoke with a real key if
`OPENROUTER_API_KEY` is available in the environment (CI provides it for smoke tests —
see `.github/workflows`): `routesmith quickstart --yes` → `serve` → one real completion
via `/v1/chat/completions` with a concrete model name → assert routed metadata. Skip
cleanly when the key is absent.

## 8. Explicitly decided (do not reopen)

1. `intercept` default is `"auto"` in the library, `"all"` in *generated* configs — this
   preserves back-compat and gives new users the just-works path.
2. Under `intercept: all` the requested model is a hint only (audit metadata), not a
   quality floor. Rationale: users adopt RouteSmith precisely to delegate the choice;
   a floor would collapse routing to always-strongest for tools that default to frontier
   models.
3. Unregistered requested models pass through rather than 404.
4. `models refresh` never changes `quality_score` and never deletes entries.
5. `thinking` blocks are dropped, not errored, on `/v1/messages`.
6. Hermes is served via the generic OpenAI-compatible path; no Hermes-specific code.
7. Catalog pricing lives in packaged JSON (curated), refreshed from live APIs where the
   provider offers one (OpenRouter fully; OpenAI/Anthropic existence-only + curated
   pricing).
8. `routesmith run` injects env per-session and execs the tool; globally exporting
   `*_BASE_URL` is documented as an alternative, never the recommended path (a stopped
   proxy must not brick tools).
9. Stickiness fingerprint is the sha256 of normalized system prompt + first user message
   (first 2,000 chars each); `sticky` defaults to `"header"` in the library and `"auto"`
   in generated configs — same back-compat pattern as `intercept`.
10. No launchd/systemd/Windows-service integration in this iteration; `--daemon` +
    pidfile is the ceiling.

## 8b. Risks and rollout guardrails (read before implementing, cite in the PR)

1. **Quality risk on agentic traffic is the top product risk.** The paper's savings
   numbers come from MMLU/GSM8K-style single-shot evals, not long-horizon agent sessions;
   a cheap model that fumbles tool-call formatting can wreck an agent loop. Guardrails:
   catalog files must set `default: true` only on models with dependable tool-calling;
   capability filtering (R3.2/AC-10) is mandatory, and conversation stickiness (R8)
   prevents mid-session thrash. Do not weaken any of these to "make routing more
   aggressive."
2. **Prompt-cache economics.** R4.5 exists because losing Anthropic prompt caching on a
   cache-heavy Claude Code session can exceed routing savings. Any future change that
   forces Anthropic→Anthropic traffic through the translation layer is a regression.
3. **Release gate:** before this ships in a release, a human (or the tester agent with a
   real key, B4-live) should run one real Claude Code session and one real OpenCode
   session through the proxy and skim `routesmith audit` for the session. Automated tests
   validate the protocol; only a real session validates the experience.

## 9. Known intentional behavior changes (tests that may be updated)

Only these existing behaviors change; any test asserting them may be *minimally* edited,
with a comment referencing this spec:

1. `anthropic_compat.anthropic_to_internal` no longer raises on `tool_use`/`tool_result`/
   `image` blocks (affects `tests/test_anthropic_endpoint.py` cases asserting the
   text-only `ValueError`).
2. Non-Anthropic-envelope errors from `/v1/messages` become Anthropic-envelope errors.
3. `quickstart` default pool is provider-dependent (affects `tests/test_cli_quickstart.py`
   assertions on `_DEFAULT_MODELS` contents).
4. Proxy responses gain a `routesmith_metadata` top-level key on the OpenAI endpoint.
5. Generated configs gain `catalog` + `routing.intercept` keys (affects config-shape
   assertions).

Everything else asserting current behavior must pass **unmodified**.
