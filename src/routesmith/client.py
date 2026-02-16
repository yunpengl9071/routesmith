"""Main RouteSmith client - drop-in replacement for LLM API calls."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, AsyncIterator, Iterator

import litellm
from litellm import ModelResponse

from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.registry.models import ModelRegistry
from routesmith.strategy.router import Router
from routesmith.feedback.collector import FeedbackCollector


@dataclass
class RoutingMetadata:
    """Metadata about a routing decision for transparency."""

    model_selected: str
    routing_strategy: str
    routing_reason: str
    routing_latency_ms: float
    estimated_cost_usd: float
    counterfactual_cost_usd: float  # What it would have cost with most expensive model
    cost_savings_usd: float
    models_considered: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for response attachment."""
        return asdict(self)


class RouteSmith:
    """
    Adaptive LLM execution engine.

    Drop-in replacement for litellm.completion() that automatically routes
    queries to optimal models based on cost, quality, and latency constraints.

    Example:
        >>> from routesmith import RouteSmith
        >>> rs = RouteSmith()
        >>> rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015)
        >>> rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006)
        >>> response = rs.completion(messages=[{"role": "user", "content": "Hello!"}])
    """

    def __init__(
        self,
        config: RouteSmithConfig | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        """
        Initialize RouteSmith.

        Args:
            config: Configuration for routing behavior, caching, and budget.
            registry: Pre-configured model registry. If None, creates empty registry.
        """
        self.config = config or RouteSmithConfig()
        self.registry = registry or ModelRegistry()
        self.router = Router(self.config, self.registry)
        self.feedback = FeedbackCollector(self.config)
        self._request_count = 0
        self._total_cost = 0.0
        self._counterfactual_cost = 0.0  # Cost if always used most expensive model
        self._last_routing_metadata: RoutingMetadata | None = None

    def register_model(
        self,
        model_id: str,
        cost_per_1k_input: float,
        cost_per_1k_output: float,
        quality_score: float = 0.8,
        latency_p50_ms: float = 500.0,
        latency_p99_ms: float = 2000.0,
        context_window: int = 128000,
        **kwargs: Any,
    ) -> None:
        """
        Register a model for routing.

        Args:
            model_id: LiteLLM model identifier (e.g., "gpt-4o", "claude-3-opus")
            cost_per_1k_input: Cost in USD per 1000 input tokens
            cost_per_1k_output: Cost in USD per 1000 output tokens
            quality_score: Expected quality score 0-1 (default 0.8)
            latency_p50_ms: Median latency in milliseconds
            latency_p99_ms: 99th percentile latency in milliseconds
            context_window: Maximum context window size
            **kwargs: Additional model metadata
        """
        self.registry.register(
            model_id=model_id,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
            quality_score=quality_score,
            latency_p50_ms=latency_p50_ms,
            latency_p99_ms=latency_p99_ms,
            context_window=context_window,
            **kwargs,
        )

    def completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        max_cost: float | None = None,
        min_quality: float | None = None,
        include_metadata: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Execute a completion request with intelligent routing.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Specific model to use (bypasses routing if provided).
            strategy: Override default routing strategy.
            max_cost: Maximum cost constraint for this request (USD).
            min_quality: Minimum quality threshold for this request (0-1).
            include_metadata: If True, attach routesmith_metadata to response.
            **kwargs: Additional arguments passed to litellm.completion().

        Returns:
            ModelResponse from the selected model. If include_metadata is True,
            response will have a routesmith_metadata attribute with routing details.
        """
        routing_start = time.perf_counter()
        self._request_count += 1

        # Determine routing strategy
        effective_strategy = strategy or self.config.default_strategy
        routing_reason = ""
        models_considered = [m.model_id for m in self.registry.list_models()]

        # If specific model requested, skip routing
        if model:
            selected_model = model
            routing_reason = "explicit model specified"
        else:
            # Route to optimal model
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
                max_cost=max_cost,
                min_quality=min_quality or self.config.budget.quality_threshold,
            )
            routing_reason = self._get_routing_reason(
                effective_strategy, selected_model, max_cost, min_quality
            )

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000

        # Execute completion via LiteLLM
        response = litellm.completion(
            model=selected_model,
            messages=messages,
            **{**self.config.litellm_params, **kwargs},
        )

        # Track costs and calculate counterfactual
        actual_cost = 0.0
        counterfactual_cost = 0.0

        if hasattr(response, "usage") and response.usage:
            model_config = self.registry.get(selected_model)
            if model_config:
                actual_cost = (
                    (response.usage.prompt_tokens / 1000) * model_config.cost_per_1k_input
                    + (response.usage.completion_tokens / 1000) * model_config.cost_per_1k_output
                )
                self._total_cost += actual_cost

            # Calculate counterfactual cost (what would most expensive model cost?)
            most_expensive = self.registry.get_best_quality()
            if most_expensive and most_expensive.model_id != selected_model:
                counterfactual_cost = (
                    (response.usage.prompt_tokens / 1000) * most_expensive.cost_per_1k_input
                    + (response.usage.completion_tokens / 1000) * most_expensive.cost_per_1k_output
                )
            else:
                counterfactual_cost = actual_cost
            self._counterfactual_cost += counterfactual_cost

        # Build routing metadata
        metadata = RoutingMetadata(
            model_selected=selected_model,
            routing_strategy=effective_strategy.value,
            routing_reason=routing_reason,
            routing_latency_ms=round(routing_latency_ms, 3),
            estimated_cost_usd=round(actual_cost, 6),
            counterfactual_cost_usd=round(counterfactual_cost, 6),
            cost_savings_usd=round(counterfactual_cost - actual_cost, 6),
            models_considered=models_considered,
        )
        self._last_routing_metadata = metadata

        # Attach metadata to response if requested
        if include_metadata:
            response.routesmith_metadata = metadata.to_dict()  # type: ignore[attr-defined]

        # Collect feedback sample
        total_latency_ms = (time.perf_counter() - routing_start) * 1000
        self.feedback.record(
            messages=messages,
            model=selected_model,
            response=response,
            latency_ms=total_latency_ms,
        )

        return response

    def _get_routing_reason(
        self,
        strategy: RoutingStrategy,
        selected_model: str,
        max_cost: float | None,
        min_quality: float | None,
    ) -> str:
        """Generate human-readable routing reason."""
        model_config = self.registry.get(selected_model)
        quality_str = f"quality={model_config.quality_score:.2f}" if model_config else ""
        cost_str = f"cost=${model_config.cost_per_1k_total:.4f}/1k" if model_config else ""

        if strategy == RoutingStrategy.DIRECT:
            if max_cost is not None:
                return f"cheapest model meeting quality threshold under ${max_cost}/1k ({quality_str}, {cost_str})"
            elif min_quality is not None:
                return f"cheapest model with quality >= {min_quality} ({quality_str}, {cost_str})"
            else:
                return f"best quality-cost tradeoff ({quality_str}, {cost_str})"
        elif strategy == RoutingStrategy.CASCADE:
            return f"cascade start with cheapest qualifying model ({quality_str}, {cost_str})"
        elif strategy == RoutingStrategy.PARALLEL:
            return f"parallel execution primary model ({quality_str}, {cost_str})"
        elif strategy == RoutingStrategy.SPECULATIVE:
            return f"speculative start with cheap model ({quality_str}, {cost_str})"
        return f"selected by {strategy.value} strategy"

    async def acompletion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        max_cost: float | None = None,
        min_quality: float | None = None,
        include_metadata: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Async version of completion().

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Specific model to use (bypasses routing if provided).
            strategy: Override default routing strategy.
            max_cost: Maximum cost constraint for this request (USD).
            min_quality: Minimum quality threshold for this request (0-1).
            include_metadata: If True, attach routesmith_metadata to response.
            **kwargs: Additional arguments passed to litellm.acompletion().

        Returns:
            ModelResponse from the selected model. If include_metadata is True,
            response will have a routesmith_metadata attribute with routing details.
        """
        routing_start = time.perf_counter()
        self._request_count += 1

        # Determine routing strategy
        effective_strategy = strategy or self.config.default_strategy
        routing_reason = ""
        models_considered = [m.model_id for m in self.registry.list_models()]

        # If specific model requested, skip routing
        if model:
            selected_model = model
            routing_reason = "explicit model specified"
        else:
            # Route to optimal model
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
                max_cost=max_cost,
                min_quality=min_quality or self.config.budget.quality_threshold,
            )
            routing_reason = self._get_routing_reason(
                effective_strategy, selected_model, max_cost, min_quality
            )

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000

        # Execute completion via LiteLLM
        response = await litellm.acompletion(
            model=selected_model,
            messages=messages,
            **{**self.config.litellm_params, **kwargs},
        )

        # Track costs and calculate counterfactual
        actual_cost = 0.0
        counterfactual_cost = 0.0

        if hasattr(response, "usage") and response.usage:
            model_config = self.registry.get(selected_model)
            if model_config:
                actual_cost = (
                    (response.usage.prompt_tokens / 1000) * model_config.cost_per_1k_input
                    + (response.usage.completion_tokens / 1000) * model_config.cost_per_1k_output
                )
                self._total_cost += actual_cost

            # Calculate counterfactual cost
            most_expensive = self.registry.get_best_quality()
            if most_expensive and most_expensive.model_id != selected_model:
                counterfactual_cost = (
                    (response.usage.prompt_tokens / 1000) * most_expensive.cost_per_1k_input
                    + (response.usage.completion_tokens / 1000) * most_expensive.cost_per_1k_output
                )
            else:
                counterfactual_cost = actual_cost
            self._counterfactual_cost += counterfactual_cost

        # Build routing metadata
        metadata = RoutingMetadata(
            model_selected=selected_model,
            routing_strategy=effective_strategy.value,
            routing_reason=routing_reason,
            routing_latency_ms=round(routing_latency_ms, 3),
            estimated_cost_usd=round(actual_cost, 6),
            counterfactual_cost_usd=round(counterfactual_cost, 6),
            cost_savings_usd=round(counterfactual_cost - actual_cost, 6),
            models_considered=models_considered,
        )
        self._last_routing_metadata = metadata

        # Attach metadata to response if requested
        if include_metadata:
            response.routesmith_metadata = metadata.to_dict()  # type: ignore[attr-defined]

        # Collect feedback sample
        total_latency_ms = (time.perf_counter() - routing_start) * 1000
        self.feedback.record(
            messages=messages,
            model=selected_model,
            response=response,
            latency_ms=total_latency_ms,
        )

        return response

    def completion_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Streaming completion with intelligent routing.

        Args:
            messages: List of message dicts.
            model: Specific model to use.
            strategy: Override default routing strategy.
            **kwargs: Additional arguments passed to litellm.completion().

        Yields:
            Streaming chunks from the selected model.
        """
        if model:
            selected_model = model
        else:
            effective_strategy = strategy or self.config.default_strategy
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
            )

        yield from litellm.completion(
            model=selected_model,
            messages=messages,
            stream=True,
            **{**self.config.litellm_params, **kwargs},
        )

    async def acompletion_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming completion with intelligent routing.

        Args:
            messages: List of message dicts.
            model: Specific model to use.
            strategy: Override default routing strategy.
            **kwargs: Additional arguments passed to litellm.acompletion().

        Yields:
            Streaming chunks from the selected model.
        """
        if model:
            selected_model = model
        else:
            effective_strategy = strategy or self.config.default_strategy
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
            )

        async for chunk in await litellm.acompletion(
            model=selected_model,
            messages=messages,
            stream=True,
            **{**self.config.litellm_params, **kwargs},
        ):
            yield chunk

    @property
    def stats(self) -> dict[str, Any]:
        """
        Get current session statistics.

        Returns:
            Dictionary with:
            - request_count: Number of requests made
            - total_cost_usd: Actual cost of all requests
            - estimated_without_routing: What it would have cost using most expensive model
            - cost_savings_usd: Total savings from intelligent routing
            - savings_percent: Percentage saved vs using most expensive model
            - registered_models: Number of models available
            - feedback_samples: Number of feedback records collected
            - last_routing: Metadata from last routing decision (if any)
        """
        savings = self._counterfactual_cost - self._total_cost
        savings_percent = (
            (savings / self._counterfactual_cost * 100)
            if self._counterfactual_cost > 0
            else 0.0
        )

        result = {
            "request_count": self._request_count,
            "total_cost_usd": round(self._total_cost, 6),
            "estimated_without_routing": round(self._counterfactual_cost, 6),
            "cost_savings_usd": round(savings, 6),
            "savings_percent": round(savings_percent, 1),
            "registered_models": len(self.registry),
            "feedback_samples": len(self.feedback),
        }

        if self._last_routing_metadata:
            result["last_routing"] = self._last_routing_metadata.to_dict()

        return result

    @property
    def last_routing_metadata(self) -> RoutingMetadata | None:
        """Get metadata from the last routing decision."""
        return self._last_routing_metadata

    def reset_stats(self) -> None:
        """Reset session statistics."""
        self._request_count = 0
        self._total_cost = 0.0
        self._counterfactual_cost = 0.0
        self._last_routing_metadata = None
