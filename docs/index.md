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
- **Framework Support**: LangChain, DSPy, CrewAI, AutoGen
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