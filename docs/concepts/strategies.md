# Routing Strategies

RouteSmith supports four routing strategies for different use cases.

## Direct Routing

Selects the single best model for each query based on quality prediction and cost constraints.

```python
from routesmith.config import RoutingStrategy
response = rs.completion(messages=[...], strategy=RoutingStrategy.DIRECT)
```

**Use when**: You want the best cost-quality tradeoff per query.

## Cascade Routing

Tries the cheapest model first, then escalates if the response quality is uncertain.

```python
response = rs.completion(messages=[...], strategy=RoutingStrategy.CASCADE)
```

**Use when**: Cost is a priority but you want a quality safety net. Most queries will succeed on the cheap model.

## Parallel Routing

Sends the query to multiple models simultaneously and returns the highest-quality response.

```python
response = rs.completion(messages=[...], strategy=RoutingStrategy.PARALLEL)
```

**Use when**: Response quality is critical and you're willing to pay more for the best result.

## Speculative Routing

Starts generation with a cheap model while evaluating whether to switch to a better model mid-stream.

```python
response = rs.completion(messages=[...], strategy=RoutingStrategy.SPECULATIVE)
```

**Use when**: You need low latency but may benefit from higher quality on complex queries.

## Default Strategy

The default strategy can be set during configuration:

```python
from routesmith import RouteSmithConfig
config = RouteSmithConfig(default_strategy=RoutingStrategy.CASCADE)
rs = RouteSmith(config=config)
```