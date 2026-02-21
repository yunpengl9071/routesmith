"""Core routing engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.predictor.base import BasePredictor
from routesmith.predictor.embedding import EmbeddingPredictor

if TYPE_CHECKING:
    from routesmith.feedback.storage import FeedbackStorage
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
        storage: "FeedbackStorage | None" = None,
    ) -> None:
        """
        Initialize router.

        Args:
            config: Routing configuration.
            registry: Model registry for available models.
            storage: Optional feedback storage for adaptive predictor training.
        """
        self.config = config
        self.registry = registry

        # Initialize quality predictor based on config
        self.predictor: BasePredictor = self._create_predictor(
            config, registry, storage
        )

    @staticmethod
    def _create_predictor(
        config: RouteSmithConfig,
        registry: "ModelRegistry",
        storage: Any,
    ) -> BasePredictor:
        """Create the appropriate predictor based on config."""
        if config.predictor_type == "adaptive":
            from routesmith.predictor.learner import AdaptivePredictor

            return AdaptivePredictor(
                registry=registry,
                storage=storage,
                min_samples=config.predictor.min_samples_for_training,
                retrain_interval=config.predictor.retrain_interval,
                n_estimators=config.predictor.n_estimators,
                blend_alpha=config.predictor.blend_alpha,
            )

        # Fallback to embedding predictor
        model_priors = {
            m.model_id: m.quality_score for m in registry.list_models()
        }
        return EmbeddingPredictor(model_quality_priors=model_priors)

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
        Direct routing: select cheapest model meeting quality threshold.

        Uses predictor.predict() for quality estimation instead of static
        registry quality_score.
        """
        # Filter by cost constraint first
        if max_cost is not None:
            candidates = self.registry.filter_by_cost(max_cost)
        else:
            candidates = self.registry.list_models()

        if not candidates:
            # Fallback to cheapest model if no candidates meet cost constraints
            cheapest = self.registry.get_cheapest()
            if cheapest:
                return cheapest.model_id
            raise ValueError("No models available")

        # Get predicted quality for all candidates
        candidate_ids = [m.model_id for m in candidates]
        predictions = self.predictor.predict(messages, candidate_ids)

        # Build a cost lookup
        cost_map = {m.model_id: m.cost_per_1k_total for m in candidates}

        # Filter by predicted quality threshold
        qualifying = [
            p for p in predictions if p.predicted_quality >= min_quality
        ]

        if qualifying:
            # Return cheapest model meeting quality threshold
            cheapest_qualifying = min(
                qualifying, key=lambda p: cost_map.get(p.model_id, float("inf"))
            )
            return cheapest_qualifying.model_id

        # No model meets quality threshold - fall back to highest predicted quality
        if predictions:
            best = max(predictions, key=lambda p: p.predicted_quality)
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

        Uses predicted quality to pick the cheapest model meeting min_quality.

        Note: Full cascade execution happens in client.py.
        This method returns the initial model to try.
        """
        candidates = self.registry.list_models()
        if not candidates:
            raise ValueError("No models available for cascade")

        candidate_ids = [m.model_id for m in candidates]
        predictions = self.predictor.predict(messages, candidate_ids)

        cost_map = {m.model_id: m.cost_per_1k_total for m in candidates}

        # Find cheapest model meeting min_quality
        qualifying = [
            p for p in predictions if p.predicted_quality >= min_quality
        ]

        if qualifying:
            cheapest = min(
                qualifying, key=lambda p: cost_map.get(p.model_id, float("inf"))
            )
            return cheapest.model_id

        # Fallback to cheapest overall
        cheapest_model = self.registry.get_cheapest()
        if cheapest_model:
            return cheapest_model.model_id

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
