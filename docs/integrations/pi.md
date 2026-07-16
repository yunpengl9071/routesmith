# pi Integration

RouteSmith optimizes your pi sessions — route every request through intelligent
model selection.

## Setup

```bash
pip install "routesmith-llm[proxy]"
routesmith quickstart
```

pi uses an OpenClaw-compatible provider config. Generate it:

```bash
routesmith connect pi
```

This is the same generator as `routesmith openclaw-config` — it creates a provider
entry that adds `routesmith/auto` to your model list. Add the generated config to
your pi configuration, and make sure the proxy is running (`routesmith serve` or
`routesmith run <another tool>` first) before starting pi.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for
simple completions, frontier models for complex reasoning. You get better
quality at lower cost without changing your workflow.

## Verify

```bash
routesmith connect pi --verify
```

Does a live round-trip through the proxy and confirms the response was actually
routed, not just that the server answered.

## Advanced

See the [Claude Code integration guide](claude-code.md) for budget enforcement,
cost tracking (`routesmith stats`), catalog refresh (`routesmith models refresh`),
and semantic caching configuration.
