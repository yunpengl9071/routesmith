# Migrating from LiteLLM to RouteSmith

## Why Migrate?

| Feature | LiteLLM | RouteSmith |
|---------|---------|------------|
| Provider abstraction | ✅ | ✅ (via LiteLLM) |
| Intelligent routing | ❌ | ✅ Cost-quality optimization |
| Cascade routing | ❌ | ✅ Try cheap, escalate |
| Budget enforcement | ❌ | ✅ Per-project daily limits |
| Semantic caching | ❌ | ✅ Embedding-based |
| Adaptive learning | ❌ | ✅ Learns from outcomes |
| Framework adapters | ❌ | ✅ LangChain, DSPy, etc. |

## Drop-in Replacement

```python
# Before — LiteLLM
import litellm
response = litellm.completion(model="gpt-4o", messages=[...])

# After — RouteSmith (same call signature!)
from routesmith import RouteSmith
rs = RouteSmith()
rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.95)
response = rs.completion(messages=[...])  # Cuts costs by 40-60%
```

## Key API Differences

1. **Models must be registered** with cost and quality metadata
2. **Auto-routing by default** — omitting `model` triggers cost-quality optimization
3. **Call `rs.stats`** for cost savings, model usage, and efficiency reports
4. **Use framework adapters** for LangChain (`ChatRouteSmith`), DSPy (`DSPyLM`), etc.