# Codex Integration

RouteSmith optimizes your Codex sessions — route every request through
intelligent model selection.

## Setup

```bash
pip install "routesmith-llm[proxy]"
routesmith quickstart
routesmith run codex
```

`routesmith run codex` starts the proxy (if needed) and launches Codex with
`OPENAI_BASE_URL` pointed at it for this session only.

## Manual configuration (alternative to `routesmith run`)

```bash
routesmith connect codex
```

Prints the `OPENAI_BASE_URL` export and the `~/.codex/config.yaml` provider block.
Codex defaults to OpenAI's Responses API — the generated config sets
`wire_api: chat` explicitly, since RouteSmith serves the chat-completions wire
format.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for
simple completions, frontier models for complex refactors. You get better
quality at lower cost without changing how you use Codex.

## Verify

```bash
routesmith connect codex --verify
```

Does a live round-trip through the proxy and confirms the response was actually
routed, not just that the server answered.

## Advanced

See the [Claude Code integration guide](claude-code.md) for budget enforcement,
cost tracking (`routesmith stats`), catalog refresh (`routesmith models refresh`),
and semantic caching configuration.
