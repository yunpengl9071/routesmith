# Standalone Usage

Use RouteSmith directly with any OpenAI-compatible provider via LiteLLM.

## Basic Usage

```python
from routesmith import RouteSmith

rs = RouteSmith()
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)

response = rs.completion(messages=[{"role": "user", "content": "Hello!"}])
print(response.choices[0].message.content)
```

## With Constraints

```python
# Enforce minimum quality
response = rs.completion(messages=[...], min_quality=0.9)

# Enforce maximum cost
response = rs.completion(messages=[...], max_cost=0.001)

# Force a specific model
response = rs.completion(messages=[...], model="gpt-4o")

# Use cascade strategy
from routesmith.config import RoutingStrategy
response = rs.completion(messages=[...], strategy=RoutingStrategy.CASCADE)
```

## Include Metadata

```python
response = rs.completion(messages=[...], include_metadata=True)
print(response.routesmith_metadata)
# {
#   "model_selected": "gpt-4o-mini",
#   "routing_reason": "direct routing with cost constraint",
#   "estimated_cost_usd": 0.0003,
# }
```

## Async

```python
response = await rs.acompletion(messages=[...])
```