"""Basic usage example for RouteSmith."""

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.config import RoutingStrategy

# Initialize RouteSmith with default config
rs = RouteSmith()

# Register available models with their costs and quality scores
rs.register_model(
    "gpt-4o",
    cost_per_1k_input=0.005,
    cost_per_1k_output=0.015,
    quality_score=0.95,
)
rs.register_model(
    "gpt-4o-mini",
    cost_per_1k_input=0.00015,
    cost_per_1k_output=0.0006,
    quality_score=0.85,
)
rs.register_model(
    "claude-3-5-sonnet-20241022",
    cost_per_1k_input=0.003,
    cost_per_1k_output=0.015,
    quality_score=0.93,
)
rs.register_model(
    "claude-3-5-haiku-20241022",
    cost_per_1k_input=0.001,
    cost_per_1k_output=0.005,
    quality_score=0.82,
)

# Simple completion - RouteSmith automatically selects optimal model
messages = [{"role": "user", "content": "What is the capital of France?"}]
response = rs.completion(messages)
print(f"Response: {response.choices[0].message.content}")

# Completion with cost constraint
response = rs.completion(
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_cost=0.001,  # USD per 1k tokens
)

# Completion with quality constraint
response = rs.completion(
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    min_quality=0.9,  # Use high-quality model
)

# Use specific routing strategy
response = rs.completion(
    messages=[{"role": "user", "content": "Summarize this article..."}],
    strategy=RoutingStrategy.CASCADE,  # Try cheap model first
)

# Check session statistics
print(f"Stats: {rs.stats}")
