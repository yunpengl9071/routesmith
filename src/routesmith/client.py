"""Main RouteSmith client - drop-in replacement for LLM API calls."""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Iterator

import litellm
from litellm import ModelResponse

from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.registry.models import ModelRegistry
from routesmith.strategy.router import Router
from routesmith.feedback.collector import FeedbackCollector


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
            **kwargs: Additional arguments passed to litellm.completion().

        Returns:
            ModelResponse from the selected model.
        """
        start_time = time.perf_counter()
        self._request_count += 1

        # If specific model requested, skip routing
        if model:
            selected_model = model
        else:
            # Route to optimal model
            effective_strategy = strategy or self.config.default_strategy
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
                max_cost=max_cost,
                min_quality=min_quality or self.config.budget.quality_threshold,
            )

        # Execute completion via LiteLLM
        response = litellm.completion(
            model=selected_model,
            messages=messages,
            **{**self.config.litellm_params, **kwargs},
        )

        # Track costs
        if hasattr(response, "usage") and response.usage:
            model_config = self.registry.get(selected_model)
            if model_config:
                cost = (
                    (response.usage.prompt_tokens / 1000) * model_config.cost_per_1k_input
                    + (response.usage.completion_tokens / 1000) * model_config.cost_per_1k_output
                )
                self._total_cost += cost

        # Collect feedback sample
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.feedback.record(
            messages=messages,
            model=selected_model,
            response=response,
            latency_ms=latency_ms,
        )

        return response

    async def acompletion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        max_cost: float | None = None,
        min_quality: float | None = None,
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
            **kwargs: Additional arguments passed to litellm.acompletion().

        Returns:
            ModelResponse from the selected model.
        """
        start_time = time.perf_counter()
        self._request_count += 1

        # If specific model requested, skip routing
        if model:
            selected_model = model
        else:
            # Route to optimal model
            effective_strategy = strategy or self.config.default_strategy
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
                max_cost=max_cost,
                min_quality=min_quality or self.config.budget.quality_threshold,
            )

        # Execute completion via LiteLLM
        response = await litellm.acompletion(
            model=selected_model,
            messages=messages,
            **{**self.config.litellm_params, **kwargs},
        )

        # Track costs
        if hasattr(response, "usage") and response.usage:
            model_config = self.registry.get(selected_model)
            if model_config:
                cost = (
                    (response.usage.prompt_tokens / 1000) * model_config.cost_per_1k_input
                    + (response.usage.completion_tokens / 1000) * model_config.cost_per_1k_output
                )
                self._total_cost += cost

        # Collect feedback sample
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.feedback.record(
            messages=messages,
            model=selected_model,
            response=response,
            latency_ms=latency_ms,
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
        """Get current session statistics."""
        return {
            "request_count": self._request_count,
            "total_cost_usd": round(self._total_cost, 6),
            "registered_models": len(self.registry),
            "feedback_samples": len(self.feedback),
        }

    def reset_stats(self) -> None:
        """Reset session statistics."""
        self._request_count = 0
        self._total_cost = 0.0
