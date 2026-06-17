# OpenCode Integration

RouteSmith optimizes your OpenCode sessions — route every request through
intelligent model selection for 40-60% cost savings.

OpenCode supports any OpenAI-compatible endpoint. Pointing it at
RouteSmith is a one-line config change.

## Setup

```bash
pip install routesmith
routesmith init --output routesmith.yaml
routesmith serve --config routesmith.yaml
```

Proxy runs at `http://localhost:9119/v1`.

## Configure OpenCode

Set the API base URL to RouteSmith's proxy:

```json
{
  "providers": {
    "routesmith": {
      "base_url": "http://localhost:9119/v1",
      "api_key": "${OPENAI_API_KEY}"
    }
  }
}
```

Or via environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:9119/v1"
```

That's it. OpenCode now routes every request through RouteSmith.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for
simple completions, frontier models for complex refactors. You get better
quality at lower cost without changing how you use OpenCode.

## Verify

```bash
curl http://localhost:9119/health
# → {"status": "ok"}
```

## Advanced

See the [OpenClaw integration guide](openclaw.md) for budget enforcement,
cost tracking (`routesmith stats`), and semantic caching configuration.