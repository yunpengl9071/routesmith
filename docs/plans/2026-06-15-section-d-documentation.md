# Section D: Documentation & Onboarding — Implementation Plan

> **REQUIRED SUB-SKILL:** Use subagent-driven-development to execute.
> **Branch:** feature/v0.2.0-documentation (from dev)
> **CRITICAL:** New user must be able to install and save 50% within 5 minutes.

**Goal:** Complete documentation: mkdocs site, API reference, quickstart, migration guide, best practices.

**Architecture:** mkdocs + Material theme + mkdocstrings for API ref. Docs live in `docs/`, examples in `examples/`.

**Tech Stack:** mkdocs, mkdocs-material, mkdocstrings, pymdown-extensions

---

## Task D1: mkdocs Site Setup

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/quickstart.md`
- Create: `docs/concepts/strategies.md`
- Create: `docs/concepts/budget.md`
- Create: `docs/concepts/adaptive.md`
- Create: `docs/concepts/cache.md`
- Create: `docs/integrations/standalone.md`
- Create: `docs/integrations/langchain.md`
- Create: `docs/integrations/dspy.md`
- Create: `docs/deployment/docker.md`
- Create: `docs/api/reference.md`
- Create: `docs/migration/litellm.md`
- Create: `docs/best-practices.md`
- Create: `docs/faq.md`
- Deps: Add `mkdocs-material`, `mkdocstrings[python]` to optional-dependencies `docs`

### Step 1: Add docs dependency

In `pyproject.toml`:
```toml
[project.optional-dependencies]
docs = [
    "mkdocs-material>=9.5.0",
    "mkdocstrings[python]>=0.25.0",
    "pymdown-extensions>=10.0.0",
]
```

### Step 2: Create mkdocs.yml

```yaml
site_name: RouteSmith
site_description: Adaptive LLM Execution Engine — intelligent routing, cascading, caching, and budget management
repo_url: https://github.com/routesmith/routesmith
theme:
  name: material
  palette:
    primary: indigo
    accent: amber
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.sections
    - content.code.copy
markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - admonition
  - toc:
      permalink: true
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            heading_level: 2
nav:
  - Home: index.md
  - Quickstart: quickstart.md
  - Concepts:
      - Routing Strategies: concepts/strategies.md
      - Cost Management: concepts/budget.md
      - Adaptive Learning: concepts/adaptive.md
      - Caching: concepts/cache.md
  - Integrations:
      - Standalone: integrations/standalone.md
      - LangChain: integrations/langchain.md
      - DSPy: integrations/dspy.md
      - CrewAI: integrations/crewai.md
      - AutoGen: integrations/autogen.md
  - Configuration:
      - Model Registry: config/registry.md
      - Budget & Alerts: config/budget.md
  - Deployment:
      - Docker: deployment/docker.md
      - Proxy Server: deployment/proxy.md
      - Monitoring: deployment/monitoring.md
  - API Reference: api/reference.md
  - Migration: migration/litellm.md
  - Best Practices: best-practices.md
  - FAQ: faq.md
```

### Step 3: Write index.md

```markdown
# RouteSmith

**Adaptive LLM Execution Engine** — save 40-60% on LLM costs without sacrificing quality.

RouteSmith sits between your application and LLM providers, automatically routing each query
to the optimal model based on cost, quality, and your constraints. It learns from your usage
patterns and gets smarter over time.

## Key Features

- **Intelligent Routing**: Direct, cascade, parallel, and speculative strategies
- **Budget Control**: Per-project daily budgets with enforcement
- **Semantic Cache**: Avoid redundant API calls
- **Adaptive Learning**: Improves routing from production feedback
- **Framework Support**: LangChain, DSPy, CrewAI, AutoGen, OpenClaw
- **Production Ready**: Docker, Prometheus metrics, structured logging

## Quick Example

```python
from routesmith import RouteSmith

rs = RouteSmith()
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)

response = rs.completion(messages=[{"role": "user", "content": "Hello!"}])
print(response.choices[0].message.content)
print(rs.stats)  # See your savings
```

[Get started in 5 minutes →](quickstart.md)
```

### Step 4: Write quickstart.md

```markdown
# Quickstart: Save 50% on LLM Costs in 5 Minutes

## 1. Install

```bash
pip install routesmith
```

## 2. Create Client

```python
from routesmith import RouteSmith

rs = RouteSmith()
```

## 3. Register Models

```python
rs.register_model("gpt-4o", 
    cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
rs.register_model("gpt-4o-mini",
    cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)
```

## 4. Route Queries

```python
# Before (OpenAI)
import openai
response = openai.chat.completions.create(model="gpt-4o", messages=[...])

# After (RouteSmith — one line change!)
response = rs.completion(messages=[...])
```

## 5. See Savings

```python
stats = rs.stats
print(f"Requests: {stats['request_count']}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Saved: {stats['savings_percent']}%")
```

## Next Steps

- [Set a budget](config/budget.md) to cap daily spend
- [Enable caching](concepts/cache.md) for repeated queries
- [Integrate with LangChain](integrations/langchain.md)
```

### Step 5: Build and verify

```bash
uv pip install -e ".[docs]"
mkdocs build --strict
# Or: mkdocs serve
```

### Step 6: Commit

```bash
git add mkdocs.yml docs/ pyproject.toml
git commit -m "docs: add mkdocs site with quickstart and concept guides"
```

---

## Task D2: API Reference

**Files:**
- Modify: `docs/api/reference.md`
- Modify: Multiple `src/routesmith/*.py` (add docstrings)

### Step 1: Write API reference page

```markdown
# API Reference

## RouteSmith Client

::: routesmith.client.RouteSmith
    options:
      members:
        - completion
        - acompletion
        - register_model
        - deregister_model
        - record_outcome
        - recommend_model_for_agent
        - register_reward_fn

## Configuration

::: routesmith.config.RouteSmithConfig
    options:
      members: true

## Model Registry

::: routesmith.registry.models.ModelRegistry
::: routesmith.registry.models.ModelConfig

## Router

::: routesmith.strategy.router.Router
    options:
      members:
        - route

## Predictors

::: routesmith.predictor.learner.AdaptivePredictor
::: routesmith.predictor.linucb.LinUCBPredictor

## Feedback

::: routesmith.feedback.collector.FeedbackCollector
::: routesmith.feedback.storage.FeedbackStorage

## Exceptions

::: routesmith.exceptions
    options:
      members: true
```

### Step 2: Ensure docstrings exist on key methods

Verify `src/routesmith/client.py` has complete docstrings for:
- `completion()`
- `acompletion()`
- `register_model()`
- `record_outcome()`
- `stats` property

### Step 3: Build and verify

```bash
mkdocs build --strict
# No warnings
```

### Step 4: Commit

```bash
git add docs/api/reference.md
git commit -m "docs: add API reference with mkdocstrings"
```

---

## Task D3: Migration Guide

**Files:**
- Modify: `docs/migration/litellm.md`

### Step 1: Write migration guide

```markdown
# Migrating from LiteLLM to RouteSmith

## Why Migrate?

| Feature | LiteLLM | RouteSmith |
|---------|---------|------------|
| Provider abstraction | ✅ | ✅ (via LiteLLM) |
| Intelligent routing | ❌ | ✅ Cost-quality optimization |
| Cascade routing | ❌ | ✅ Try cheap, escalate if needed |
| Budget enforcement | ❌ | ✅ Per-project daily limits |
| Semantic caching | ❌ | ✅ Embedding-based |
| Adaptive learning | ❌ | ✅ Learns from outcomes |
| Framework adapters | ❌ | ✅ LangChain, DSPy, etc. |

## Drop-in Replacement

```python
# LiteLLM
import litellm
response = litellm.completion(model="gpt-4o", messages=[...])

# RouteSmith (same call signature, better results)
from routesmith import RouteSmith
rs = RouteSmith()
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
response = rs.completion(model="gpt-4o", messages=[...])
```

## Key Differences

1. **Model registration required**: You must register models with cost/quality before routing
2. **Auto-routing by default**: If you don't specify a model, RouteSmith picks the best one
3. **Stats available**: Call `rs.stats` for cost/savings
4. **framework adapters**: Replace `ChatOpenAI()` with `ChatRouteSmith()`

## Cost Savings

Typical savings: 40-60% on mixed workloads with zero configuration beyond listing models.
```

### Step 2: Commit

```bash
git add docs/migration/litellm.md
git commit -m "docs: add migration guide from LiteLLM"
```

---

## Task D4: Best Practices Guide

**Files:**
- Modify: `docs/best-practices.md`

### Step 1: Write best practices

```markdown
# Best Practices

## Model Registration

- Use **real pricing** from provider dashboards
- Set `quality_score` from eval benchmarks, or start conservative (0.7-0.8)
- Add `capabilities` for tool_calling/vision models

## Routing Strategy Selection

| Use Case | Strategy | Why |
|----------|----------|-----|
| General chat | `DIRECT` | Simple, fast |
| Cost-sensitive | `CASCADE` | Tries cheap first |
| High-stakes | `PARALLEL` | Multiple models, best result |
| Latency-sensitive | `SPECULATIVE` | Starts cheap, switches mid-stream |
| Bedrock provisioned | `PROVISIONED_FIRST` | Uses pre-paid capacity |

## Budget Management

- Start with `daily_budget = monthly_budget / 30`
- Use `FALLBACK` behavior for production (never fail)
- Set alerts at 50%, 75%, 90% of budget
- Monitor `routesmith_budget_remaining_usd` metric

## Feedback Loop

- Enable `feedback_sample_rate=1.0` during development
- Use `record_outcome()` for explicit feedback (thumbs up/down)
- Let implicit signals handle the rest (latency, token count, etc.)
- Check predictor diagnostics weekly: `rs.router.predictor.diagnostics()`

## Production Checklist

- [ ] Structured JSON logging enabled
- [ ] Prometheus metrics exported
- [ ] Health checks configured
- [ ] Circuit breaker thresholds tuned
- [ ] Budget exceeded behavior configured
- [ ] Docker image built and scanned
- [ ] Nightly live tests passing
```

### Step 2: Commit

```bash
git add docs/best-practices.md
git commit -m "docs: add best practices guide"
```

---

## Task D5: Example Projects

**Files:**
- Create: `examples/langchain_agent.py`
- Create: `examples/budget_enforcement.py`
- Create: `examples/semantic_cache.py`

### Step 1: Write langchain_agent.py

```python
"""LangChain agent with RouteSmith automatic cost optimization."""
from routesmith.integrations.langchain import ChatRouteSmith

# Create with built-in OpenRouter model presets
llm = ChatRouteSmith.with_openai_models()

# Use like any LangChain chat model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = create_react_agent(llm, tools=[multiply])
result = agent.invoke({"messages": [{"role": "user", "content": "What is 137 * 42?"}]})

print(result["messages"][-1].content)
print(f"\nCost: ${llm.routesmith.stats['total_cost_usd']:.4f}")
```

### Step 2: Write budget_enforcement.py

```python
"""Set a daily budget and handle budget exceeded gracefully."""
from routesmith import RouteSmith, RouteSmithConfig
from routesmith.config import RoutingStrategy

config = RouteSmithConfig().with_budget(max_cost_per_day=10.0)
rs = RouteSmith(config=config)

rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)

for i in range(50):
    response = rs.completion(
        messages=[{"role": "user", "content": f"Count: {i}"}],
        strategy=RoutingStrategy.CASCADE,
        min_quality=0.7,
    )

print(f"Spent: ${rs.stats['total_cost_usd']:.4f}")
print(f"Saved: {rs.stats['savings_percent']}%")
```

### Step 3: Commit

```bash
git add examples/langchain_agent.py examples/budget_enforcement.py
git commit -m "docs: add example projects for LangChain agent and budget enforcement"
```

---

## Task D6: FAQ

**Files:**
- Modify: `docs/faq.md`

### Step 1: Write FAQ

```markdown
# FAQ

## How much can I save?

40-60% on mixed workloads. Simple queries route to cheap models, complex ones to expensive models.

## Does it work with my provider?

Yes. RouteSmith wraps LiteLLM, which supports 100+ providers (OpenAI, Anthropic, Groq, Bedrock, Azure, etc.).

## How does routing work?

RouteSmith predicts which model will produce the best quality for each query, then selects the optimal model based on your cost/quality constraints.

## What if I don't know quality scores?

Start with conservative defaults (0.7-0.8) and let RouteSmith learn from production feedback.

## Can I force a specific model?

Yes: `rs.completion(model="gpt-4o", messages=[...])`

## Does it add latency?

Routing overhead is <5ms P99. LLM calls are 500-5000ms — routing is imperceptible.

## Where is data stored?

Feedback and stats are stored in a local SQLite database by default. No data leaves your infrastructure.

## Can I use it with LangChain?

Yes: `from routesmith.integrations.langchain import ChatRouteSmith`. See [LangChain integration](integrations/langchain.md).

## What happens when budget is exceeded?

Configurable: fail with error, fall back to cheapest model, or queue until reset.

## Is it production ready?

v0.2.0+ is production-ready with circuit breakers, structured logging, Prometheus metrics, and Docker support.
```

### Step 2: Commit

```bash
git add docs/faq.md
git commit -m "docs: add FAQ"
```

---

## UAT Validation (Real Use Case)

```bash
# 1. Build docs
uv pip install -e ".[docs]"
mkdocs build --strict
# Expected: zero warnings

# 2. Check new user flow
# Start timer, follow quickstart, verify cost savings reported in <5 min

# 3. Verify all links work
mkdocs serve &
# Browse to http://localhost:8000, click through all pages

# 4. Check examples run (mocked)
.venv/bin/python examples/basic_usage.py
```
