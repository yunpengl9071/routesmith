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

## Adaptive Routing with Contextual Bandits

RouteSmith uses contextual bandit algorithms to learn optimal routing policies online. Unlike static routers, these predictors adapt to your specific workload by learning from feedback on every request.

### Predictor Types

| Predictor | Config value | Best for |
|-----------|-------------|----------|
| **LinUCB** | `"linucb"` | Fast, stable baseline. O(d²) updates, good theoretical guarantees |
| **NeuralUCB** | `"neural_ucb"` | Complex workloads with nonlinear feature interactions |
| **WarmStartLinUCB** | `"warmstart_linucb"` | When you have historical quality data to eliminate cold-start |
| **REINFORCE** | `"reinforce"` | Stochastic policy with entropy-regularized exploration |
| **Adaptive (RF)** | `"adaptive"` | Legacy batch-trained random forest baseline |

```python
from routesmith import RouteSmith, RouteSmithConfig
from routesmith.config import PredictorConfig

# LinUCB: good default for most workloads
config = RouteSmithConfig(
    predictor_type="linucb",
    predictor=PredictorConfig(
        linucb_alpha=1.5,        # Exploration width
        linucb_cost_lambda=0.3,  # Cost penalty weight
    ),
)
rs = RouteSmith(config=config)

# NeuralUCB: captures nonlinear patterns
config = RouteSmithConfig(
    predictor_type="neural_ucb",
    predictor=PredictorConfig(
        neural_ucb_alpha=0.5,
        neural_ucb_hidden_dim=64,
        neural_ucb_replay_size=2000,
    ),
)

# WarmStartLinUCB: pre-initialize from labeled data
config = RouteSmithConfig(
    predictor_type="warmstart_linucb",
    predictor=PredictorConfig(
        warmstart_alpha=1.5,
        warmstart_cost_lambda=0.3,
        warmstart_latency_lambda=0.1,
    ),
)
# After creating the router, warm-start from historical data:
# router.predictor.warm_start(labeled_examples, epochs=1)
```

### 27-Dimensional Context Features

The routing decision uses a rich feature vector combining query characteristics and model metadata:

- **Query type classification**: Math, reasoning, code, and creative task scores via keyword analysis
- **Difficulty estimation**: Combines length, vocabulary complexity, and structural indicators
- **Model metadata**: Cost, latency, quality prior, capabilities
- **Interaction features**: Estimated response length, difficulty × quality prior

### Empirical Results

On RouteLLM public evaluation data (10 seeds, 14K MMLU questions):

- **53% less cumulative regret** vs random routing
- **Cold-start in ~100 queries** vs RF's 100+ label batch requirement
- **Sub-millisecond routing decisions** for all predictor types
- **WarmStartLinUCB eliminates cold-start** entirely with 500+ labeled examples

### Comparison with Other Routers

| | RouteSmith (CB) | RouteLLM | FrugalGPT | AutoMix |
|---|---|---|---|---|
| **Routing type** | Contextual bandit | Supervised classifier | LLM cascade | POMDP + self-verify |
| **When it routes** | Before generation | Before generation | After generation | After generation |
| **Pre-training data** | None (or optional 500 labels) | 55K+ labels | Scorer training | POMDP tuning |
| **Routing latency** | <1ms | 5-800ms | Multiple LLM calls | ≥2× SLM calls |
| **Online adaptation** | Yes | No | No | No |
| **Multi-model** | N models | 2 models | Ordered chain | 2 models |

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
