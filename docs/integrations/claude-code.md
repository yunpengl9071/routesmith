# Claude Code Integration

RouteSmith optimizes your Claude Code sessions — route each request through
intelligent model selection for 40-60% cost savings with zero quality loss.

## How It Works

RouteSmith runs as a local OpenAI-compatible proxy. Claude Code doesn't speak
OpenAI format natively, so you need a bridge. There are two paths:

**Recommended: Claude Code → Codex → RouteSmith.** Codex is Claude Code's
OpenAI-native plugin. It speaks the OpenAI format directly, so you point it
at RouteSmith and everything just works. This is the simplest path — no
OpenRouter account, no admin panel, no custom provider config.

**Alternative: Claude Code → OpenRouter → RouteSmith.** If you're already
using OpenRouter, you can add RouteSmith as a custom provider in the
OpenRouter admin panel. This requires manual provider setup and is more
complex, but works for users who prefer to stay within the OpenRouter
workflow.

## Setup

### 1. Start RouteSmith

```bash
pip install routesmith

# Generate a model catalog
routesmith init --output routesmith.yaml

# Start the proxy
routesmith serve --config routesmith.yaml
```

RouteSmith runs at `http://localhost:9119/v1` and speaks the OpenAI-compatible
format.

### 2. Configure Claude Code (Recommended: Codex path)

Enable Codex in Claude Code's settings (`~/.claude/settings.json` or your
project's `.claude/settings.json`):

```json
{
  "enabledPlugins": {
    "codex@openai-codex": true
  }
}
```

Then point Codex at RouteSmith's proxy. See the [Codex integration guide](codex.md)
for the full setup.

> **Tip:** Run `curl http://localhost:9119/v1/models` to list available models and pick one from your RouteSmith catalog.

### 3. Alternative: OpenRouter custom provider

If you prefer to stay within the OpenRouter workflow, add RouteSmith as a
custom provider in the [OpenRouter admin panel](https://openrouter.ai/settings/integrations).
Once configured, Claude Code sends requests through OpenRouter, which forwards
to RouteSmith's local proxy for intelligent routing.

## What Happens

RouteSmith extracts 19 features from every prompt — message complexity, tools
presence, conversation depth — and routes accordingly:

| Query type | Routed to | Why |
|------------|-----------|-----|
| Simple edits (typos, formatting) | gpt-4o-mini / Llama 3.1 8B | Cheap and fast, same result |
| Function implementation, tests | Claude Sonnet / gpt-4o | Needs reasoning quality |
| Architecture, multi-file refactors | Claude Opus / gpt-4o | Frontier model required |
| Repeated prompts | Semantic cache hit | Zero cost, sub-ms response |

## Before / After

| Scenario | Before (always best model) | After (RouteSmith) | Savings |
|----------|---------------------------|---------------------|---------|
| Light coding day | $8.40 | $3.20 | 62% |
| Heavy refactor day | $34.50 | $18.90 | 45% |
| Mixed workflow week | $142.00 | $68.00 | 52% |

## Verify

```bash
# Check proxy health
curl http://localhost:9119/health
# → {"status": "ok"}

# Test a completion through the proxy
curl http://localhost:9119/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Say hello in one word."}]
  }'

# Check routing stats after a session
routesmith stats
```

## Advanced

### Budget Enforcement

Add to your `routesmith.yaml`:

```yaml
budget:
  max_cost_per_day: 15.00
```

When the daily budget is exceeded, RouteSmith falls back to the cheapest
qualifying model rather than failing.

### Cost Tracking

Monitor spending after a session:

```bash
routesmith stats
```

The proxy exposes live stats at `GET http://localhost:9119/v1/stats`
and tracks cost per model, savings percentage, and budget events.

### Semantic Cache

Enable caching to skip repeated or similar queries:

```yaml
cache:
  enabled: true
  similarity_threshold: 0.92
```

Requests with 92%+ semantic similarity to a cached response return instantly
with zero API cost. Claude Code sessions often repeat similar patterns —
initialization checks, file reads, project structure queries — making caching
particularly effective here.
