# RouteSmith

**The smart router for AI coding tools.** 40-60% cost savings. Zero quality loss. Backed by contextual bandit research.

```bash
pip install "routesmith[proxy]"
routesmith init
routesmith serve
# → Proxy at http://localhost:9119/v1
```

RouteSmith sits between your AI coding tool and the LLM. It routes every request
to the best model for that specific task — cheap models for simple edits, frontier
models for complex refactors. You never think about model IDs again.

**New in v0.5.0-beta**: NeuralUCB, REINFORCE, and WarmStart LinUCB predictors. Research paper with benchmark results included.

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
| Learns from feedback | ❌ | ❌ | ✅ |

## Features

### Intelligent Routing
- **7 predictor types**: LinUCB, LinTS, NeuralUCB, REINFORCE, WarmStart LinUCB, Adaptive (random forest), Embedding
- **27-dimensional feature space**: query type classification, difficulty estimation, model metadata
- **Online learning**: bandits improve from the first query onward — no pretraining labels needed
- **Multi-model routing**: scales to $K$ arms naturally (validated on 5-model deployments)

### Enterprise
- **Provisioned throughput support**: prioritize pre-paid capacity, overflow to on-demand
- **Compliance routing**: tag-based filtering (HIPAA, SOC2, PCI)
- **Budget enforcement**: FAIL, FALLBACK, QUEUE behaviors
- **Multi-project isolation**: per-project cost allocation and stats

### Production
- **Semantic cache**: embedding-based dedup, configurable similarity
- **Framework adapters**: LangChain, DSPy, CrewAI, AutoGen, Anthropic, OpenClaw
- **OpenAI-compatible proxy**: works with any tool, zero code changes
- **Observability**: Prometheus metrics, structured logging, cost tracking, dashboard TUI
- **Resilience**: circuit breakers, retry with backoff, health checks, Docker

## Research

RouteSmith is backed by a research paper evaluating contextual bandit routing:

- **LinTS-27d** achieves 46% cost reduction with APGR=0.593 on MMLU
- **LinUCB-27d** achieves APGR=1.126 by selective strong-arm routing
- **5-arm routing**: 45% cost savings across GPT-4o, Claude-Sonnet-4.5, Qwen-Plus, MiniMax-M1, DeepSeek-V3
- **Zero pretraining labels** — learns from ~100 queries vs. 55K+ required by supervised routers
- Sub-millisecond routing latency (<0.5ms P99)

Paper: [`paper/main.pdf`](paper/main.pdf) | Compile with: `cd paper && tectonic main.tex`

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

- [CHANGELOG.md](CHANGELOG.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Integration guides](docs/integrations/)
- [CLI reference](docs/cli.md)
- [Concepts](docs/concepts/)

## Installation

```bash
# Proxy + interactive setup (recommended)
pip install "routesmith[proxy]"

# Core Python API only
pip install routesmith

# With specific integrations
pip install "routesmith[langchain]"
pip install "routesmith[anthropic]"
pip install "routesmith[cache]"
pip install "routesmith[all]"
```

Requires Python 3.10+. Set `OPENROUTER_API_KEY` to use OpenRouter models.

## License

MIT — see [LICENSE](LICENSE)