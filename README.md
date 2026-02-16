# RouteSmith

**The adaptive LLM router that learns what works for your agents.**

RouteSmith is an intelligent cost-quality optimization layer for LLM applications. It automatically routes queries to optimal models, cascades through model tiers, caches semantically similar responses, and enforces budget constraints—all while maintaining quality guarantees.

## Why RouteSmith?

Most teams default to expensive models (GPT-4, Claude Opus) for everything because choosing the right model per query is hard. RouteSmith solves this by:

- **Routing intelligently**: Automatically select the cheapest model that meets your quality threshold
- **Learning from outcomes**: Unlike static routers, RouteSmith improves over time based on what works for YOUR specific use case
- **Supporting enterprise patterns**: AWS Bedrock provisioned throughput, Azure OpenAI PTUs, compliance-based routing

## Quick Start

```bash
pip install routesmith
```

```python
from routesmith import RouteSmith

rs = RouteSmith()

# Register your available models
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)

# RouteSmith automatically picks the best model
response = rs.completion(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    min_quality=0.8  # Use cheapest model meeting this threshold
)

# See what happened
print(rs.stats)
# {'request_count': 1, 'total_cost_usd': 0.0001, ...}
```

## Features

### Intelligent Routing

```python
# Simple queries → cheap models
# Complex queries → premium models
response = rs.completion(messages=[...], min_quality=0.8)
```

### Cost Tracking with Savings Calculation

```python
print(rs.stats)
# {
#   'total_cost_usd': 12.45,
#   'estimated_without_routing': 89.20,
#   'savings_percent': 86.0
# }
```

### Multiple Routing Strategies

```python
from routesmith.config import RoutingStrategy

# Direct: Pick single best model
response = rs.completion(messages=[...], strategy=RoutingStrategy.DIRECT)

# Cascade: Try cheap model first, escalate if needed
response = rs.completion(messages=[...], strategy=RoutingStrategy.CASCADE)
```

### Budget Controls

```python
from routesmith import RouteSmithConfig

config = RouteSmithConfig().with_budget(
    max_cost_per_request=0.01,  # Cap per request
    max_cost_per_day=50.0,      # Daily budget
)
rs = RouteSmith(config=config)
```

## Framework Integrations

RouteSmith works as middleware with popular agent frameworks:

```python
# OpenClaw - via local proxy
routesmith serve --port 9119

# LangChain
from routesmith.integrations.langchain import RouteSmithLLM
llm = RouteSmithLLM()

# DSPy
from routesmith.integrations.dspy import RouteSmithLM
dspy.configure(lm=RouteSmithLM())
```

## Enterprise Features

- **AWS Bedrock**: Provisioned throughput support (maximize utilization of pre-paid capacity)
- **Azure OpenAI**: PTU (Provisioned Throughput Units) support
- **Compliance routing**: Tag models with `hipaa`, `soc2`, etc. and route accordingly
- **Multi-tenant**: Per-project cost allocation and budgets

```python
# Provisioned throughput: marginal cost = $0 (already paid)
rs.register_model(
    "bedrock/claude-3-sonnet-provisioned",
    cost_model=CostModel.PROVISIONED,
    hourly_cost=66.0,
    capacity_requests_per_min=20,
)

# Compliance-based routing
rs.register_model("bedrock/claude", tags=["hipaa", "soc2"])
response = rs.completion(messages=[...], require_tags=["hipaa"])
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Application                             │
├─────────────────────────────────────────────────────────────────┤
│                       RouteSmith                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Model    │  │ Quality  │  │ Router   │  │ Cost Tracker   │  │
│  │ Registry │  │ Predictor│  │ Engine   │  │ + Feedback     │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    LiteLLM (100+ providers)                      │
├─────────────────────────────────────────────────────────────────┤
│            OpenAI, Anthropic, Bedrock, Azure, Groq, etc.         │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone and install
git clone https://github.com/yourusername/routesmith.git
cd routesmith
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/
```

## Requirements

- Python 3.10+
- At least one LLM provider API key (OpenAI, Anthropic, Groq, etc.)

## License

MIT

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.
