# Claude Code Integration

RouteSmith sits between Claude Code and the model provider, routing every request
(including subagent calls) to the best model for that specific task — cheap models
for simple edits, frontier models for complex refactors.

## Setup

### 1. Install and configure RouteSmith

```bash
pip install "routesmith-llm[proxy]"

# Detects your provider API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY,
# OPENROUTER_API_KEY, GROQ_API_KEY) and builds a matching model pool
routesmith quickstart
```

### 2. Run Claude Code through RouteSmith

```bash
routesmith run claude
```

This starts the RouteSmith proxy (if it isn't already running) and launches Claude
Code with `ANTHROPIC_BASE_URL` pointed at it, for this session only — your shell
environment and any other terminal are untouched. Claude Code behaves exactly as
normal; RouteSmith just picks the model on every turn, including subagent calls,
which inherit the same routing.

That's the whole setup. Everything below is for manual configuration, CI, or
understanding what's happening under the hood.

## Manual configuration (alternative to `routesmith run`)

If you'd rather not use the wrapper — e.g. you want RouteSmith on for every Claude
Code session in a given shell — print the exact env/config with:

```bash
routesmith connect claude-code
```

This sets `ANTHROPIC_BASE_URL` (and an `ANTHROPIC_API_KEY`/auth-token placeholder if
your proxy runs with `--api-key`) either as exports or in the `env` block of
`~/.claude/settings.json`. Requires RouteSmith's native `/v1/messages` endpoint
(supports full tool use and streaming; ships from v0.9.0).

> **Note:** this works for API-key-based Anthropic usage. Claude Code sessions
> authenticated via a Pro/Max subscription (OAuth) cannot be rerouted — RouteSmith
> pays for downstream calls with your provider keys, and OAuth sessions don't have
> a key to swap. Use `routesmith run claude` for everything else.

## What Happens

RouteSmith extracts features from every prompt — message complexity, tool presence,
conversation depth — and routes accordingly:

| Query type | Routed to | Why |
|------------|-----------|-----|
| Simple edits (typos, formatting) | gpt-4o-mini / Claude Haiku | Cheap and fast, same result |
| Function implementation, tests | Claude Sonnet / gpt-4o | Needs reasoning quality |
| Architecture, multi-file refactors | Claude Opus / gpt-4o | Frontier model required |
| Subagent fan-out (Explore, grep-heavy tasks) | Cheapest capable model | Routed independently per subagent |
| Repeated prompts | Semantic cache hit | Zero cost, sub-ms response |

Within one conversation (including a subagent's own conversation), RouteSmith keeps
routing to the same model turn-to-turn — so a session doesn't switch models mid-task.
A new conversation gets a fresh routing decision.

For measured cost/quality tradeoffs on real benchmarks (not this table), see
[Research](../../README.md#research) in the main README.

## Verify

```bash
routesmith connect claude-code --verify
```

This does a live round-trip through the proxy and confirms the response was actually
**routed** — not just that the server is reachable. If it reports requests are
passing through unrouted, it prints the exact config fix.

```bash
# Check routing stats after a session
routesmith stats

# See individual routing decisions
routesmith audit
```

## Advanced

### Budget Enforcement

Add to your `routesmith.yaml`:

```yaml
budget:
  max_cost_per_day: 15.00
```

When the daily budget is exceeded, RouteSmith falls back to the cheapest qualifying
model rather than failing.

### Refreshing your model catalog

Prices and context windows drift. Re-sync from providers without losing your
customizations:

```bash
routesmith models refresh
```

### Semantic Cache

Enable caching to skip repeated or similar queries:

```yaml
cache:
  enabled: true
  similarity_threshold: 0.92
```

Requests with 92%+ semantic similarity to a cached response return instantly with
zero API cost. Claude Code sessions often repeat similar patterns — initialization
checks, file reads, project structure queries — making caching particularly
effective here.
