# Claude Code Integration

RouteSmith optimizes your Claude Code sessions — route each request through
intelligent model selection for 40-60% cost savings with zero quality loss.

Claude Code connects through OpenRouter to reach any model provider. RouteSmith
sits between them, analyzing every query and picking the best model for each
task: cheap models for simple edits, frontier models for architecture decisions.

## How It Works

```
Claude Code  →  OpenRouter  →  RouteSmith  →  Models
                                  │
                            analyzes query
                            picks best model
                            cascades if needed
                            caches repeats
```

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
format that OpenRouter already translates for Claude Code.

### 2. Configure Claude Code

Edit `~/.claude/settings.json` or your project's `.claude/settings.json`:

```json
{
  "model": "anthropic/open_router/openai/gpt-4o-mini"
}
```

Replace `anthropic/open_router/openai/gpt-4o-mini` with any model in your
RouteSmith catalog. Claude Code sends every request through OpenRouter, and
RouteSmith intercepts to route it intelligently.

> **Tip:** Use `routesmith models` to list available models. Pick the model ID
> that matches your RouteSmith catalog.

### 3. Or use Codex (OpenAI-native path)

Claude Code's Codex plugin speaks OpenAI format directly — point it at
RouteSmith:

```json
{
  "enabledPlugins": {
    "codex@openai-codex": true
  }
}
```

Then set Codex to use RouteSmith's proxy (see the [Codex integration guide](codex.md)).

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
