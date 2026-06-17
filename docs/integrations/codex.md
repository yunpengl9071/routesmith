# Codex Integration

RouteSmith optimizes your Codex sessions — route every request through
intelligent model selection for 40-60% cost savings.

Codex speaks OpenAI format natively. Pointing it at RouteSmith is one
line of configuration.

## Setup

```bash
pip install routesmith
routesmith init --output routesmith.yaml
routesmith serve --config routesmith.yaml
```

Proxy runs at `http://localhost:9119/v1`.

## Configure Codex

Set the OpenAI base URL to RouteSmith's proxy:

```bash
export OPENAI_BASE_URL="http://localhost:9119/v1"
```

Or in your Codex config file (`~/.codex/config.yaml`):

```yaml
provider: openai
base_url: http://localhost:9119/v1
api_key: ${OPENAI_API_KEY}
```

That's it. Codex now routes every request through RouteSmith.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for
simple completions, frontier models for complex refactors. You get better
quality at lower cost without changing how you use Codex.

## Verify

```bash
curl http://localhost:9119/health
# → {"status": "ok"}
```

## Advanced

See the [OpenClaw integration guide](openclaw.md) for budget enforcement,
cost tracking (`routesmith stats`), and semantic caching configuration.
