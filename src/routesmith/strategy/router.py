"""Core routing engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.predictor.embedding import EmbeddingPredictor

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry


class Router:
    """
    Core routing engine for selecting optimal models.

    Implements multiple routing strategies:
    - DIRECT: Route to single best model based on quality prediction
    - CASCADE: Try cheap model first, escalate if confidence is low
    - PARALLEL: Run multiple models, select best response
    - SPECULATIVE: Start with cheap model while evaluating
    """

    def __init__(
        self,
        config: RouteSmithConfig,
        registry: "ModelRegistry",
    ) -> None:
        """
        Initialize router.

        Args:
            config: Routing configuration.
            registry: Model registry for available models.
        """
        self.config = config
        self.registry = registry

        # Initialize quality predictor based on config
        model_priors = {
            m.model_id: m.quality_score for m in registry.list_models()
        }
        self.predictor = EmbeddingPredictor(model_quality_priors=model_priors)

    def route(
        self,
        messages: list[dict[str, str]],
        strategy: RoutingStrategy = RoutingStrategy.DIRECT,
        max_cost: float | None = None,
        min_quality: float = 0.0,
    ) -> str:
        """
        Select the optimal model for a query.

        Args:
            messages: Input messages.
            strategy: Routing strategy to use.
            max_cost: Maximum cost constraint (USD per 1k tokens).
            min_quality: Minimum quality threshold (0-1).

        Returns:
            Selected model ID.

        Raises:
            ValueError: If no suitable model found.
        """
        if len(self.registry) == 0:
            raise ValueError("No models registered. Call register_model() first.")

        if strategy == RoutingStrategy.DIRECT:
            return self._route_direct(messages, max_cost, min_quality)
        elif strategy == RoutingStrategy.CASCADE:
            return self._route_cascade(messages, max_cost, min_quality)
        elif strategy == RoutingStrategy.PARALLEL:
            return self._route_parallel(messages, max_cost, min_quality)
        elif strategy == RoutingStrategy.SPECULATIVE:
            return self._route_speculative(messages, max_cost, min_quality)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

    def _route_direct(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
    ) -> str:
        """
        Direct routing: select single best model.

        Optimizes for quality while respecting cost constraints.
        """
        # Filter by cost constraint
        if max_cost is not None:
            candidates = self.registry.filter_by_cost(max_cost)
        else:
            candidates = self.registry.list_models()

        if not candidates:
            # Fallback to cheapest model if no candidates meet constraints
            cheapest = self.registry.get_cheapest()
            if cheapest:
                return cheapest.model_id
            raise ValueError("No models available")

        # Get quality predictions
        model_ids = [m.model_id for m in candidates]
        prediction = self.predictor.predict_best(messages, model_ids, min_quality)

        if prediction:
            return prediction.model_id

        # Fallback: return highest quality model that meets cost constraint
        best = self.registry.get_best_quality(max_cost)
        if best:
            return best.model_id

        # Last resort: return cheapest model
        cheapest = self.registry.get_cheapest()
        if cheapest:
            return cheapest.model_id

        raise ValueError("No models available")

    def _route_cascade(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
    ) -> str:
        """
        Cascade routing: start with cheap model, escalate if needed.

        Uses Lagrangian optimization for threshold selection
        (Dekoninck et al., ICML 2025).

        Note: Full cascade execution happens in client.py.
        This method returns the initial model to try.
        """
        # For cascade, start with the cheapest model meeting minimum quality
        cheapest = self.registry.get_cheapest(min_quality)
        if cheapest:
            return cheapest.model_id

        # Fallback to cheapest overall
        cheapest = self.registry.get_cheapest()
        if cheapest:
            return cheapest.model_id

        raise ValueError("No models available for cascade")

    def _route_parallel(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
    ) -> str:
        """
        Parallel routing: run multiple models, select best.

        Note: Full parallel execution happens in client.py.
        This returns the primary model to use.
        """
        # For parallel strategy, return highest quality model
        best = self.registry.get_best_quality(max_cost)
        if best:
            return best.model_id

        raise ValueError("No models available for parallel execution")

    def _route_speculative(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
    ) -> str:
        """
        Speculative routing: start cheap while evaluating escalation.

        Similar to cascade but begins generation immediately.
        """
        # Start with cheapest model
        return self._route_cascade(messages, max_cost, min_quality)

    def get_cascade_models(
        self,
        min_quality: float = 0.0,
        max_tiers: int = 3,
    ) -> list[str]:
        """
        Get ordered list of models for cascade execution.

        Returns models sorted by cost (cheapest first) that can
        be tried in sequence during cascade routing.

        Args:
            min_quality: Minimum quality threshold.
            max_tiers: Maximum number of cascade tiers.

        Returns:
            List of model IDs ordered by cost (cheapest first).
        """
        candidates = self.registry.filter_by_quality(min_quality)
        sorted_models = sorted(candidates, key=lambda m: m.cost_per_1k_total)
        return [m.model_id for m in sorted_models[:max_tiers]]
