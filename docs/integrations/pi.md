# pi Integration

RouteSmith optimizes your pi sessions — route every request through intelligent
model selection for better quality at lower cost.

pi speaks OpenAI format through its provider system. Pointing it at
RouteSmith is one line of configuration.

## Setup

```bash
pip install routesmith
routesmith init --output routesmith.yaml
routesmith serve --config routesmith.yaml
```

Proxy runs at `http://localhost:9119/v1`.

## Configure pi

pi uses an OpenClaw-compatible provider config. Generate it:

```bash
routesmith openclaw-config --output routesmith-provider.json
```

Add the generated provider to your pi configuration. RouteSmith appears as
`routesmith/auto` in your model list.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for
simple completions, frontier models for complex reasoning. You get better
quality at lower cost without changing your workflow.

## Verify

```bash
curl http://localhost:9119/health
# → {"status": "ok"}
```

## Advanced

See the [OpenClaw integration guide](openclaw.md) for budget enforcement,
cost tracking (`routesmith stats`), and semantic caching configuration.