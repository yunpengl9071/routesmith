# Cost Management

RouteSmith helps you control LLM costs with per-project budgets, alerts, and enforcement behaviors.

## Setting a Daily Budget

```python
from routesmith import RouteSmith, RouteSmithConfig

config = RouteSmithConfig().with_budget(max_cost_per_day=100.0)
rs = RouteSmith(config=config)
```

## Budget Exceeded Behavior

Configure what happens when the budget is exceeded:

```python
config = RouteSmithConfig().with_budget(
    max_cost_per_day=100.0,
    exceeded_behavior="fallback",  # "fail", "fallback", or "queue"
)
```

- **fail**: Raise `BudgetExceededError`
- **fallback**: Use the cheapest model regardless of quality (default)
- **queue**: Queue requests until the budget resets

## Cost Tracking

```python
stats = rs.stats
# {
#   "request_count": 1500,
#   "total_cost_usd": 12.45,
#   "counterfactual_cost_usd": 89.20,  # What it would have cost without routing
#   "savings_percent": 86.0,
#   "by_model": {"gpt-4o-mini": 1434, "gpt-4o": 66},
# }
```

## Viewing Savings

The `routesmith_metadata` on each response includes cost breakdown:

```python
response = rs.completion(messages=[...], include_metadata=True)
print(response.routesmith_metadata)
# {
#   "model_selected": "gpt-4o-mini",
#   "estimated_cost_usd": 0.0003,
#   "counterfactual_cost_usd": 0.015,
#   "cost_savings_usd": 0.0147,
# }
```