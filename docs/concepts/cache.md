# Caching

RouteSmith supports semantic caching to avoid redundant API calls.

## Enabling the Cache

```python
from routesmith import RouteSmithConfig
config = RouteSmithConfig().with_cache(
    enabled=True,
    similarity_threshold=0.85,  # 0-1, higher = stricter matching
    max_size=10000,
)
rs = RouteSmith(config=config)
```

## How It Works

1. Each query is embedded using sentence-transformers
2. The embedding is compared against cached query embeddings
3. If similarity exceeds the threshold, the cached response is returned
4. Cache hits cost $0 and have <1ms overhead

## When to Use

- Repetitive queries (customer support FAQs)
- Template-based generation (same prompts with minor variations)
- Idempotent operations (summarization of known content)

## Cache Stats

```python
stats = rs.stats
# Includes: cache_hit_rate, cache_size, cache_evictions
```