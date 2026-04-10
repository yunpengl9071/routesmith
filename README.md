# RouteSmith

Adaptive LLM execution engine with online contextual bandit routing. Routes queries to the right model dynamically, learning which models work best for which tasks while minimizing cost.

## What it does

Instead of hardcoding which LLM to use, RouteSmith:

- **Routes each query** to the cheapest model expected to answer it correctly
- **Learns online** — updates routing decisions after each response using Thompson Sampling (LinTS) or UCB (LinUCB)
- **Tracks budget** — enforces per-request and daily cost limits
- **Caches semantically** — reuses responses to similar past queries

## Quick start

```bash
pip install "routesmith[proxy]"

# Interactive setup: browse OpenRouter catalog, select models, generate config
routesmith init

# Start OpenAI-compatible proxy
routesmith serve
```

Point any OpenAI client at `http://localhost:9119`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9119/v1", api_key="any")
response = client.chat.completions.create(
    model="auto",   # RouteSmith picks the model
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

## Algorithms

### LinTS-27d (recommended, default)

Linear Thompson Sampling. Maintains a Gaussian posterior over routing quality per model. Samples from each arm's posterior at decision time — no hyperparameter to tune, automatic exploration-exploitation balance.

### LinUCB-27d

Contextual bandit with UCB exploration. Uses an `alpha` parameter (default 1.5) to control exploration. Achieves higher routing accuracy in benchmarks (APGR=1.126 vs LinTS=0.593 on MMLU) at the cost of needing alpha tuning.

Both use a 27-dimensional feature vector: message features (length, complexity, question type) × model features (cost, quality score, context window) + interaction terms.

## Config file

```yaml
# routesmith.yaml

routing:
  strategy: direct
  predictor: lints           # lints | linucb | embedding | adaptive
  lints_v_sq: 1.0            # posterior variance scaling (LinTS)
  # linucb_alpha: 1.5        # UCB exploration parameter (LinUCB)

budget:
  max_cost_per_request: 0.10
  max_cost_per_day: 50.0
  quality_threshold: 0.75

# Option 1: specify models with full pricing
models:
  - id: openai/gpt-4o-mini
    cost_per_1k_input: 0.15
    cost_per_1k_output: 0.60
    quality_score: 0.85

  - id: deepseek/deepseek-chat
    cost_per_1k_input: 0.014
    cost_per_1k_output: 0.028
    quality_score: 0.72

# Option 2: let RouteSmith fetch pricing from OpenRouter automatically
openrouter_models:
  - openai/gpt-4o-mini
  - deepseek/deepseek-chat
  - anthropic/claude-3-haiku
```

Generate a config interactively:

```bash
routesmith init                         # guided setup
routesmith init --output custom.yaml    # custom output path
routesmith init --force                 # overwrite existing
```

## Python API

```python
from routesmith import RouteSmith, RouteSmithConfig, PredictorConfig

config = RouteSmithConfig(
    predictor_type="lints",
    predictor=PredictorConfig(lints_v_sq=1.0),
    max_cost_per_request=0.05,
)
rs = RouteSmith(config=config)

rs.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15, cost_per_1k_output=0.60, quality_score=0.85)
rs.register_model("deepseek/deepseek-chat", cost_per_1k_input=0.014, cost_per_1k_output=0.028, quality_score=0.72)

response = rs.complete(messages=[{"role": "user", "content": "Explain recursion"}])

# Provide feedback to update routing posteriors
rs.record_feedback(response, quality=0.9)

# Session stats
print(rs.stats)
# {'request_count': 1, 'total_cost_usd': 0.0023, 'cost_savings_usd': 0.0190, 'savings_percent': 89.2}
```

## Framework integrations

RouteSmith works with every major agent framework — natively (no proxy needed) or via the OpenAI-compatible proxy.

### Anthropic SDK

```python
from routesmith.integrations.anthropic import RouteSmithAnthropic

# Auto-fetches Claude pricing from OpenRouter
client = RouteSmithAnthropic.with_openrouter_models()

msg = client.messages.create(
    model="auto",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain transformers"}],
)
print(msg.content[0].text)
```

### LangChain

```python
from routesmith.integrations.langchain import ChatRouteSmith

llm = ChatRouteSmith()
llm.routesmith.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                               cost_per_1k_output=0.60, quality_score=0.85)
response = llm.invoke("What is 2+2?")
```

### DSPy

```python
import dspy
from routesmith.integrations.dspy import RouteSmithLM

# Native mode — no proxy needed
lm = RouteSmithLM()
lm.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                  cost_per_1k_output=0.60, quality_score=0.85)
dspy.configure(lm=lm)

# Or proxy mode (requires routesmith serve)
from routesmith.integrations.dspy import routesmith_lm
dspy.configure(lm=routesmith_lm())
```

### CrewAI

```python
from crewai import Agent
from routesmith.integrations.crewai import routesmith_crewai_chat_model

# Native mode — ChatRouteSmith as CrewAI LLM
llm = routesmith_crewai_chat_model()
llm.routesmith.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15,
                               cost_per_1k_output=0.60, quality_score=0.85)
agent = Agent(role="Analyst", goal="Answer questions", backstory="...", llm=llm)

# Or proxy mode
from routesmith.integrations.crewai import routesmith_crewai_llm
agent = Agent(role="Analyst", goal="...", backstory="...", llm=routesmith_crewai_llm())
```

### AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent
from routesmith.integrations.autogen import routesmith_autogen_agents

# One-line agent pair
assistant, user = routesmith_autogen_agents()
user.initiate_chat(assistant, message="Write a haiku about cost optimization.")

# Or configure manually
from routesmith.integrations.autogen import routesmith_autogen_llm_config
llm_config = routesmith_autogen_llm_config()
agent = AssistantAgent("assistant", llm_config=llm_config)
```

### OpenClaw

```bash
# Generate provider config and add to OpenClaw settings
routesmith openclaw-config --output openclaw-provider.json

# Or pipe directly
routesmith openclaw-config >> ~/.openclaw/settings.json
```

## Routing strategies

```python
from routesmith.config import RoutingStrategy

# Direct: pick single best model per query (default)
response = rs.completion(messages=[...], strategy=RoutingStrategy.DIRECT)

# Cascade: try cheap model first, escalate if quality too low
response = rs.completion(messages=[...], strategy=RoutingStrategy.CASCADE)
```

## Benchmark results

Evaluated on MMLU (600 queries, GPT-4o as strong model, DeepSeek-V3 as weak model):

| Method | Accuracy | Cost/query | APGR |
|--------|----------|-----------|------|
| Static strong (GPT-4o) | 77.7% | $0.213 | — |
| Static weak (DeepSeek-V3) | 73.2% | $0.018 | — |
| RouteLLM-SW | 73.0–74.8% | — | −0.111 to −0.222 |
| **LinTS-27d** | **75.8%** | **$0.116** | 0.593 |
| **LinUCB-27d** | **78.2%** | — | **1.126** |

APGR (Performance Gap Recovery) = (router_acc − weak_acc) / (strong_acc − weak_acc). Values above 1.0 mean the router outperforms always-using-the-strong-model.

5-arm routing across GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, DeepSeek-V3, and GPT-4o-mini achieves 71.0% accuracy at $0.117/query — 45% cheaper than GPT-4o alone.

## CLI reference

```bash
routesmith init [--output FILE] [--force]        # generate config interactively
routesmith serve [--config FILE] [--port N]      # start proxy (default port 9119)
routesmith openclaw-config [--host URL] [-o FILE] # generate OpenClaw provider config
routesmith stats [--server URL] [--json]         # show session stats
```

## Installation

```bash
# Proxy server + interactive setup (recommended)
pip install "routesmith[proxy]"

# Anthropic SDK integration
pip install "routesmith[anthropic]"

# LangChain integration
pip install "routesmith[langchain]"

# DSPy integration
pip install "routesmith[dspy]"

# CrewAI integration
pip install "routesmith[crewai]"

# AutoGen integration
pip install "routesmith[autogen]"

# Semantic caching
pip install "routesmith[cache]"

# Core only (Python API, no proxy)
pip install routesmith
```

Requires Python 3.10+. Set `OPENROUTER_API_KEY` to use OpenRouter models.

## Development

```bash
git clone https://github.com/yunpengl9071/routesmith.git
cd routesmith
uv venv .venv --python 3.13
uv pip install -e ".[dev]"
.venv/bin/pytest tests/
```

## License

MIT
