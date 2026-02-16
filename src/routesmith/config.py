"""Configuration for RouteSmith."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RoutingStrategy(Enum):
    """Available routing strategies."""

    DIRECT = "direct"  # Route to single best model
    CASCADE = "cascade"  # Try cheap model first, escalate if needed
    PARALLEL = "parallel"  # Run multiple models, select best response
    SPECULATIVE = "speculative"  # Start with cheap model while evaluating


@dataclass
class CacheConfig:
    """Configuration for semantic cache layer."""

    enabled: bool = False
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600
    max_entries: int = 10000
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class BudgetConfig:
    """Configuration for budget constraints."""

    max_cost_per_request: float | None = None  # Max cost in USD per request
    max_cost_per_minute: float | None = None  # Max cost in USD per minute
    max_cost_per_hour: float | None = None  # Max cost in USD per hour
    max_cost_per_day: float | None = None  # Max cost in USD per day
    quality_threshold: float = 0.8  # Minimum acceptable quality score (0-1)


@dataclass
class RouteSmithConfig:
    """Main configuration for RouteSmith."""

    # Routing behavior
    default_strategy: RoutingStrategy = RoutingStrategy.DIRECT
    fallback_model: str | None = None  # Model to use if routing fails

    # Quality prediction
    predictor_type: str = "embedding"  # embedding, classifier, random_forest
    predictor_model: str | None = None  # Custom predictor model path

    # Cache settings
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Budget constraints
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    # Feedback loop
    feedback_enabled: bool = True
    feedback_sample_rate: float = 0.1  # Fraction of requests to evaluate

    # Performance
    routing_timeout_ms: int = 50  # Max time for routing decision
    enable_telemetry: bool = True

    # Provider settings passed to LiteLLM
    litellm_params: dict[str, Any] = field(default_factory=dict)

    def with_cache(self, **kwargs: Any) -> RouteSmithConfig:
        """Return a new config with updated cache settings."""
        new_cache = CacheConfig(
            enabled=kwargs.get("enabled", self.cache.enabled),
            similarity_threshold=kwargs.get(
                "similarity_threshold", self.cache.similarity_threshold
            ),
            ttl_seconds=kwargs.get("ttl_seconds", self.cache.ttl_seconds),
            max_entries=kwargs.get("max_entries", self.cache.max_entries),
            embedding_model=kwargs.get("embedding_model", self.cache.embedding_model),
        )
        return RouteSmithConfig(
            default_strategy=self.default_strategy,
            fallback_model=self.fallback_model,
            predictor_type=self.predictor_type,
            predictor_model=self.predictor_model,
            cache=new_cache,
            budget=self.budget,
            feedback_enabled=self.feedback_enabled,
            feedback_sample_rate=self.feedback_sample_rate,
            routing_timeout_ms=self.routing_timeout_ms,
            enable_telemetry=self.enable_telemetry,
            litellm_params=self.litellm_params,
        )

    def with_budget(self, **kwargs: Any) -> RouteSmithConfig:
        """Return a new config with updated budget settings."""
        new_budget = BudgetConfig(
            max_cost_per_request=kwargs.get(
                "max_cost_per_request", self.budget.max_cost_per_request
            ),
            max_cost_per_minute=kwargs.get(
                "max_cost_per_minute", self.budget.max_cost_per_minute
            ),
            max_cost_per_hour=kwargs.get("max_cost_per_hour", self.budget.max_cost_per_hour),
            max_cost_per_day=kwargs.get("max_cost_per_day", self.budget.max_cost_per_day),
            quality_threshold=kwargs.get("quality_threshold", self.budget.quality_threshold),
        )
        return RouteSmithConfig(
            default_strategy=self.default_strategy,
            fallback_model=self.fallback_model,
            predictor_type=self.predictor_type,
            predictor_model=self.predictor_model,
            cache=self.cache,
            budget=new_budget,
            feedback_enabled=self.feedback_enabled,
            feedback_sample_rate=self.feedback_sample_rate,
            routing_timeout_ms=self.routing_timeout_ms,
            enable_telemetry=self.enable_telemetry,
            litellm_params=self.litellm_params,
        )
