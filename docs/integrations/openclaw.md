# OpenClaw Integration

RouteSmith optimizes your OpenClaw agent workflows — route every agent call through
intelligent model selection for better quality at lower cost.

## Setup

```bash
pip install "routesmith-llm[proxy]"

# Detects your provider API keys and builds a matching model pool
routesmith quickstart
```

Or generate a catalog interactively (browse OpenRouter's model list, pick 3–10
models):

```bash
routesmith init --output routesmith.yaml
```

Start the proxy:

```bash
routesmith serve --config routesmith.yaml
```

The proxy starts at `http://localhost:9119/v1` and speaks the OpenAI-compatible
format that OpenClaw already understands.

## Configure OpenClaw

Generate the provider config:

```bash
routesmith connect openclaw
```

(equivalent to `routesmith openclaw-config`). This creates a provider entry that
adds `routesmith/auto` as an available model. Add the generated config to your
OpenClaw settings. Make sure RouteSmith is running (`routesmith serve --config
routesmith.yaml`) before starting OpenClaw.

> **Note:** If you're using the `--config` flag with OpenClaw, you can pass the
> generated file directly: `openclaw --config routesmith-provider.json`

## What Happens

RouteSmith analyzes every prompt before sending it to a model. It extracts 19
features (message complexity, tools presence, conversation depth, etc.) and
routes accordingly:

| Query type | Routed to | Why |
|------------|-----------|-----|
| Simple completions | gpt-4o-mini / Llama 3.1 8B | Cheap and fast, more than capable |
| Tool-calling agents | Claude Sonnet / gpt-4o | Needs structured output quality |
| Complex reasoning | gpt-4o / Claude Opus | Frontier model required |
| Multi-agent coordination | Cascade: cheap → check → escalate | Don't pay unless the first answer is wrong |
| Repeated queries | Semantic cache hit | Same meaning, zero cost, sub-ms response |

## Verify

```bash
routesmith connect openclaw --verify
```

Does a live round-trip through the proxy and confirms the response was actually
routed — not just that the server is reachable. A bare `curl /health` check will
report OK even when every request is silently passing through unrouted.

## Advanced

### Budget Enforcement

Add to your `routesmith.yaml`:

```yaml
budget:
  max_cost_per_day: 10.00
```

When the daily budget is exceeded, new requests fail with a clear error.
Combine with cascade routing (`routing: {strategy: cascade}`) to reduce
costs before hitting the cap.

### Cost Tracking

Monitor spending after a session:

```bash
routesmith stats
```

The proxy also exposes live stats at `GET http://localhost:9119/v1/stats`
and tracks cost per model, savings percentage, and budget events.

### Semantic Cache

Enable caching to skip repeated or similar queries:

```yaml
cache:
  enabled: true
  similarity_threshold: 0.92
```

Requests with 92%+ semantic similarity to a cached response return instantly
with zero API cost.
