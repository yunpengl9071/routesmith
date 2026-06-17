# Cost Models

RouteSmith supports three pricing models for registered models: **On-Demand**, **Provisioned**, and **Self-Hosted**. Each model changes how RouteSmith calculates costs and tracks capacity.

---

## On-Demand (Default)

Pay per token. This is the standard pricing model for most API-based LLM providers. All models default to `CostModel.ON_DEMAND` unless you specify otherwise.

```python
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015)
# cost_model defaults to CostModel.ON_DEMAND — same as:
```

On-demand models always have available capacity (no RPM limit from RouteSmith — provider rate limits apply separately).

---

## Provisioned Throughput

Fixed hourly cost for guaranteed capacity. Common with [AWS Bedrock](https://aws.amazon.com/bedrock/provisioned-throughput/) and [Azure OpenAI PTUs](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/provisioned-throughput). Once you've purchased provisioned throughput, the **marginal cost per request is $0**.

```python
from routesmith.config import CostModel

rs.register_model(
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    cost_per_1k_input=0.0,        # Marginal cost is $0
    cost_per_1k_output=0.0,
    quality_score=0.92,
    cost_model=CostModel.PROVISIONED,
    capacity_requests_per_min=10,  # Max requests per minute
    provisioned_hourly_cost=66.0,  # For cost reporting only
    provisioned_units=2,           # Purchased capacity units
)
```

### Capacity Tracking

Provisioned models have a rolling 60-second window RPM limit enforced by `CapacityTracker`. When you exceed this limit, RouteSmith can overflow to on-demand models (see [PROVISIONED_FIRST Routing](concepts/strategies.md)).

```python
stats = rs.stats
print(stats["provisioned_utilization"])  # {"bedrock/...": 0.78} — 78% utilized
print(stats["provisioned_overflow"])     # 234 requests that went to on-demand fallback
```

> **Note:** Capacity tracking is cooperative, not enforced at the provider level. If you also send direct requests to the same provisioned endpoint, actual throughput may differ.

---

## Self-Hosted

Local or self-managed models with near-zero marginal cost (electricity + compute). RouteSmith treats these as the cheapest possible tier with no capacity tracking.

```python
from routesmith.config import CostModel

rs.register_model(
    "ollama/llama3.1:70b",
    cost_per_1k_input=0.0001,
    cost_per_1k_output=0.0001,
    quality_score=0.82,
    cost_model=CostModel.SELF_HOSTED,
)
```

---

## PROVISIONED_FIRST Routing

When `RoutingStrategy.PROVISIONED_FIRST` is selected, RouteSmith tries provisioned models first, then overflows to on-demand:

```
Request arrives
    │
    ▼
Check provisioned capacity
    ├─ Available → Route to provisioned ($0)
    └─ Exhausted → Route to on-demand (quality prediction)
```

See [Routing Strategies](concepts/strategies.md) for full details.

```python
from routesmith.config import RoutingStrategy

response = rs.completion(
    messages=[{"role": "user", "content": "Analyze this document"}],
    strategy=RoutingStrategy.PROVISIONED_FIRST,
)
```

### Stats by Cost Model

```python
stats = rs.stats
print(stats["by_cost_model"])
# {"provisioned": {"requests": 4766, "cost": 0.0},
#  "on_demand": {"requests": 234, "cost": 12.45}}
```

---

## Best Practices

- **Set accurate pricing** — Use real provider pricing from your dashboard. RouteSmith's routing decisions depend on it.
- **Start with ON_DEMAND** — If you're not using provisioned throughput, the default is correct.
- **Register provisioned models last** — During cold start, cheaper default routing won't accidentally exhaust scarce provisioned capacity.
- **Monitor overflow percentage** — If overflow > 20%, consider buying more provisioned capacity:

```python
overflow = rs.stats.get("provisioned_overflow", 0)
total = rs.stats["request_count"]
if total > 0 and overflow / total > 0.2:
    print(f"High overflow: {overflow}/{total} requests fell back to on-demand")
```