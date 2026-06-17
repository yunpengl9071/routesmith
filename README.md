# RouteSmith

**The smart router for AI coding tools.** 40-60% cost savings. Zero quality loss.

```bash
pip install "routesmith[proxy]"
routesmith init
routesmith serve
# → Proxy at http://localhost:9119/v1
```

RouteSmith sits between your AI coding tool and the LLM. It routes every request
to the best model for that specific task — cheap models for simple edits, frontier
models for complex refactors. You never think about model IDs again.

## Who it's for

**💰 You pay for API access. Cut your bill 40-60%.**
Your Claude Code or Codex session burns through tokens. RouteSmith sends typos and
formatting to gpt-4o-mini, saves Claude Opus for architecture decisions.

**🆓 You use free models. Get better answers.**
Free models are good individually — but none is great at everything. RouteSmith
orchestrates them: hard problems get the strongest free model, easy ones get the
fastest, and weak answers cascade to second opinions.

```python
from routesmith import RouteSmith

# Free models — zero cost, smart routing
rs = RouteSmith.with_free_models()

# Or bring your own models
rs = RouteSmith()
rs.register_model("gpt-4o-mini", cost_per_1k_input=0.15, cost_per_1k_output=0.60, quality_score=0.85)
rs.register_model("claude-sonnet-4", cost_per_1k_input=3.0, cost_per_1k_output=15.0, quality_score=0.92)

response = rs.completion(messages=[{"role": "user", "content": "Explain recursion"}])
```

## AI coding tools

Point any AI coding tool at `http://localhost:9119/v1`:

| Tool | Config |
|------|--------|
| Claude Code | Enable Codex plugin, set `OPENAI_BASE_URL=http://localhost:9119/v1` |
| Codex | `export OPENAI_BASE_URL="http://localhost:9119/v1"` |
| OpenClaw | `routesmith openclaw-config` |
| pi | `routesmith openclaw-config` (OpenClaw-compatible provider) |
| OpenCode | Set `base_url` to `http://localhost:9119/v1` in provider config |

[Integration guides →](https://github.com/yunpengl9071/routesmith/tree/dev/docs/integrations)

## Why RouteSmith

| | Raw OpenRouter | Manual routing | RouteSmith |
|---|---|---|---|
| Picks model per query | ❌ | 😓 You do it | ✅ Automatic |
| Cascades when answer is weak | ❌ | ❌ | ✅ |
| Caches repetitive queries | ❌ | ❌ | ✅ |
| Enforces budget limits | ❌ | ❌ | ✅ |
| Tracks costs per model | ✅ | ❌ | ✅ |
| Works with 100+ models | ✅ | ❌ | ✅ |
| Zero-config start | ✅ | ❌ | ✅ |

## Features

- **Intelligent routing** — 19 features per query, adaptive learning from feedback
- **Cascade execution** — try cheap first, escalate on low confidence
- **Semantic cache** — embedding-based dedup, configurable similarity
- **Budget enforcement** — daily caps, fallback to cheaper models
- **Framework adapters** — LangChain, DSPy, CrewAI, AutoGen, Anthropic, OpenClaw
- **OpenAI-compatible proxy** — works with any tool, zero code changes
- **Observability** — Prometheus metrics, structured logging, cost tracking
- **Production-ready** — circuit breakers, retry, health checks, Docker

## Framework integrations

```python
# Anthropic SDK
from routesmith.integrations.anthropic import RouteSmithAnthropic
client = RouteSmithAnthropic.with_openrouter_models()

# LangChain
from routesmith.integrations.langchain import ChatRouteSmith
llm = ChatRouteSmith.with_openai_models()

# DSPy
from routesmith.integrations.dspy import RouteSmithLM
lm = RouteSmithLM()

# CrewAI
from routesmith.integrations.crewai import routesmith_crewai_chat_model
llm = routesmith_crewai_chat_model()

# AutoGen
from routesmith.integrations.autogen import routesmith_autogen_agents
assistant, user = routesmith_autogen_agents()
```

## Quick start

```bash
# Interactive setup: browse OpenRouter catalog, pick models
routesmith init

# Start the proxy
routesmith serve

# Check stats
routesmith stats
```

```python
# Python API
from routesmith import RouteSmith

rs = RouteSmith.with_free_models()
response = rs.completion(messages=[{"role": "user", "content": "Hello!"}])
print(response.choices[0].message.content)

# Learn from user feedback
rs.record_outcome(response._routesmith_request_id, score=0.9)
```

## Documentation

- [Tutorial](docs/tutorial.md)
- [Integration guides](docs/integrations/)
- [CLI reference](docs/cli.md)
- [Concepts](docs/concepts/)

## Installation

```bash
pip install "routesmith[proxy]"    # proxy + interactive setup (recommended)
pip install routesmith             # core Python API only
```

Requires Python 3.10+. Set `OPENROUTER_API_KEY` to use OpenRouter models.

## License

MIT