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
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)
```

## 4. Route Queries

```python
# Before — direct OpenAI
response = openai.chat.completions.create(model="gpt-4o", messages=[...])

# After — RouteSmith (one line change!)
response = rs.completion(messages=[...])
```

## 5. See Your Savings

```python
stats = rs.stats
print(f"Requests: {stats['request_count']}")
print(f"Total cost: ${stats['total_cost_usd']:.4f}")
print(f"Saved: {stats['savings_percent']}%")
```

## Next Steps

- [Integrate with LangChain](integrations/langchain.md)
- [Deploy with Docker](deployment/docker.md)