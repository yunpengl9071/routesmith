# OpenCode Integration

RouteSmith optimizes your OpenCode sessions — route every request through
intelligent model selection.

## Setup

```bash
pip install "routesmith-llm[proxy]"
routesmith quickstart
routesmith run opencode
```

`routesmith run opencode` starts the proxy (if needed) and launches OpenCode with
`OPENAI_BASE_URL` pointed at it for this session only.

## Manual configuration (alternative to `routesmith run`)

```bash
routesmith connect opencode
```

Prints the exact `providers.routesmith` block for `opencode.json`, or the
equivalent `OPENAI_BASE_URL` export.

## What Happens

RouteSmith analyzes each prompt and picks the best model — cheap ones for
simple completions, frontier models for complex refactors. You get better
quality at lower cost without changing how you use OpenCode.

## Verify

```bash
routesmith connect opencode --verify
```

Does a live round-trip through the proxy and confirms the response was actually
routed, not just that the server answered. `curl http://localhost:9119/health` only
proves the process is up — it will report OK even when every request is silently
passing through unrouted.

## Advanced

See the [Claude Code integration guide](claude-code.md) for budget enforcement,
cost tracking (`routesmith stats`), catalog refresh (`routesmith models refresh`),
and semantic caching configuration.
