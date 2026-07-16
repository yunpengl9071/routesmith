# Hermes Integration

RouteSmith optimizes your Hermes sessions — route every request through intelligent
model selection.

Hermes accepts any OpenAI-compatible provider configuration. Pointing it at
RouteSmith is a one-line config change.

## Setup

```bash
pip install "routesmith-llm[proxy]"
routesmith quickstart
routesmith serve
```

Proxy runs at `http://localhost:9119/v1`.

## Configure Hermes

```bash
routesmith connect hermes
```

Prints the `OPENAI_BASE_URL` export to add to your Hermes provider configuration.
Any Hermes provider entry that accepts an OpenAI-compatible `base_url` works —
point it at `http://localhost:9119/v1`.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for simple
completions, frontier models for complex reasoning. You get better quality at lower
cost without changing your workflow.

## Verify

```bash
routesmith connect hermes --verify
```

Does a live round-trip through the proxy and confirms the response was actually
routed, not just that the server answered.

## Advanced

See the [Claude Code integration guide](claude-code.md) for budget enforcement,
cost tracking (`routesmith stats`), catalog refresh (`routesmith models refresh`),
and semantic caching configuration.
