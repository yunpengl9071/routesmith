"""Core routing engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.predictor.base import BasePredictor
from routesmith.predictor.embedding import EmbeddingPredictor

if TYPE_CHECKING:
    from routesmith.config import RouteContext
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
        registry: ModelRegistry,
        storage: FeedbackStorage | None = None,
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
        registry: ModelRegistry,
        storage: Any,
    ) -> BasePredictor:
        """Create the appropriate predictor based on config.

        Supported predictor_type values:
          - "lints"          : LinTS-27d (Thompson Sampling, recommended)
          - "linucb"         : LinUCB-27d (UCB exploration)
          - "neural_ucb"     : NeuralUCB (shallow NN + UCB)
          - "reinforce"      : Policy-gradient REINFORCE
          - "warmstart_linucb": LinUCB with benchmark-prior warm start
          - "adaptive"       : Random-forest adaptive predictor (offline)
          - "embedding"      : Embedding-based predictor (default fallback)
        """
        if config.predictor_type == "lints":
            from routesmith.predictor.lints import LinTSPredictor

            return LinTSPredictor(
                registry=registry,
                v_sq=config.predictor.lints_v_sq,
                cost_lambda=config.predictor.linucb_cost_lambda,
                seed=config.predictor.seed,
            )

        if config.predictor_type == "linucb":
            from routesmith.predictor.linucb import LinUCBPredictor

            return LinUCBPredictor(
                registry=registry,
                alpha=config.predictor.linucb_alpha,
                cost_lambda=config.predictor.linucb_cost_lambda,
                warmup_rounds=config.predictor.linucb_warmup_rounds,
            )

        if config.predictor_type == "neural_ucb":
            from routesmith.predictor.neural_ucb import NeuralUCBPredictor

            return NeuralUCBPredictor(
                registry=registry,
                alpha=config.predictor.neural_ucb_alpha,
                cost_lambda=config.predictor.neural_ucb_cost_lambda,
                latency_lambda=config.predictor.neural_ucb_latency_lambda,
                learning_rate=config.predictor.neural_ucb_lr,
                hidden_dim=config.predictor.neural_ucb_hidden_dim,
                warmup_rounds=config.predictor.neural_ucb_warmup_rounds,
                replay_size=config.predictor.neural_ucb_replay_size,
            )

        if config.predictor_type == "reinforce":
            from routesmith.predictor.reinforce import ReinforcePredictor

            return ReinforcePredictor(
                registry=registry,
                learning_rate=config.predictor.reinforce_lr,
                baseline_lr=config.predictor.reinforce_baseline_lr,
                cost_lambda=config.predictor.reinforce_cost_lambda,
                temperature=config.predictor.reinforce_temperature,
                entropy_bonus=config.predictor.reinforce_entropy_bonus,
            )

        if config.predictor_type == "warmstart_linucb":
            from routesmith.predictor.warmstart_linucb import (
                WarmStartLinUCBPredictor,
            )

            return WarmStartLinUCBPredictor(
                registry=registry,
                alpha=config.predictor.warmstart_alpha,
                cost_lambda=config.predictor.warmstart_cost_lambda,
                latency_lambda=config.predictor.warmstart_latency_lambda,
                warmup_rounds=config.predictor.warmstart_warmup_rounds,
            )

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
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
    ) -> str:
        """
        Select the optimal model for a query.

        Args:
            messages: Input messages.
            strategy: Routing strategy to use.
            max_cost: Maximum cost constraint (USD per 1k tokens).
            min_quality: Minimum quality threshold (0-1).
            required_capabilities: Capabilities the selected model must support.
            required_compliance: Compliance tags the selected model must have.
            context: Optional agent/conversation context for business rules.

        Returns:
            Selected model ID.

        Raises:
            ValueError: If no suitable model found.
        """
        if len(self.registry) == 0:
            raise ValueError("No models registered. Call register_model() first.")

        if strategy == RoutingStrategy.DIRECT:
            return self._route_direct(messages, max_cost, min_quality, required_capabilities, required_compliance, context)
        elif strategy == RoutingStrategy.CASCADE:
            return self._route_cascade(messages, max_cost, min_quality, required_capabilities, required_compliance, context)
        elif strategy == RoutingStrategy.PARALLEL:
            return self._route_parallel(messages, max_cost, min_quality, required_capabilities, required_compliance, context)
        elif strategy == RoutingStrategy.SPECULATIVE:
            return self._route_speculative(messages, max_cost, min_quality, required_capabilities, required_compliance, context)
        elif strategy == RoutingStrategy.PROVISIONED_FIRST:
            return self._route_provisioned_first(messages, max_cost, min_quality, required_capabilities, required_compliance, context)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

    def _apply_business_rules(
        self,
        candidates: list[Any],
        context: RouteContext | None,
    ) -> list[Any]:
        """Apply configured business rules to the candidate model list.

        Rules are applied in order. Each rule receives the current candidate list
        and the RouteContext, and returns a filtered list. If any rule empties the
        list, a ValueError is raised immediately.
        """
        for rule in self.config.business_rules:
            candidates = rule(candidates, context)
            if not candidates:
                raise ValueError(
                    "All models were filtered out by business rules. "
                    "Check your RouteSmithConfig.business_rules."
                )
        return candidates

    def _filter_by_compliance(
        self,
        candidates: list[Any],
        required_compliance: set[str] | None,
    ) -> list[Any]:
        """Filter candidates by required compliance tags."""
        if not required_compliance:
            return candidates
        filtered = [c for c in candidates if required_compliance.issubset(c.compliance_tags)]
        if not filtered:
            from routesmith.exceptions import NoCompliantModelError
            available_tags: set[str] = set()
            for m in self.registry.list_models():
                available_tags.update(m.compliance_tags)
            raise NoCompliantModelError(
                required_tags=required_compliance,
                available_tags=available_tags,
            )
        return filtered

    def _filter_by_capabilities(
        self,
        candidates: list[Any],
        required_capabilities: set[str] | None,
    ) -> list[Any]:
        """Filter candidates by required capabilities, raising if none match."""
        if not required_capabilities:
            return candidates
        filtered = [c for c in candidates if required_capabilities.issubset(c.capabilities)]
        if not filtered:
            available = {cap for m in self.registry.list_models() for cap in m.capabilities}
            missing = required_capabilities - available
            raise ValueError(
                f"No models support required capabilities: {missing or required_capabilities}"
            )
        return filtered

    def _route_direct(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
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

        # Apply business rules before capability/compliance filtering
        candidates = self._apply_business_rules(candidates, context)

        # Filter by required capabilities
        candidates = self._filter_by_capabilities(candidates, required_capabilities)

        # Filter by required compliance
        candidates = self._filter_by_compliance(candidates, required_compliance)

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
            # Determine selection criterion based on tradeoff
            tradeoff = context.metadata.get("tradeoff", 7) if context and hasattr(context, 'metadata') else 7
            if tradeoff <= 2:
                # Low tradeoff: pick highest quality model
                best = max(qualifying, key=lambda p: p.predicted_quality)
            elif tradeoff >= 8:
                # High tradeoff: pick cheapest model
                best = min(qualifying, key=lambda p: cost_map.get(p.model_id, float("inf")))
            else:
                # Mid tradeoff: cheapest above quality threshold (default)
                best = min(qualifying, key=lambda p: cost_map.get(p.model_id, float("inf")))
            return best.model_id

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
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
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

        # Apply business rules before capability/compliance filtering
        candidates = self._apply_business_rules(candidates, context)

        # Filter by required capabilities
        candidates = self._filter_by_capabilities(candidates, required_capabilities)

        # Filter by required compliance
        candidates = self._filter_by_compliance(candidates, required_compliance)

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
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
    ) -> str:
        """
        Parallel routing: return highest-quality model as primary.

        The actual parallel execution (running multiple models concurrently,
        comparing responses) happens in client.py._completion_parallel().
        """
        candidates = self._get_candidates(
            max_cost, required_capabilities, required_compliance, context
        )
        if candidates:
            best = max(candidates, key=lambda m: m.quality_score)
            return best.model_id
        raise ValueError("No models available for parallel execution")

    def _route_speculative(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
    ) -> str:
        """
        Speculative routing: return cheapest qualifying model as primary.

        The actual speculative flow (start cheap, escalate on low confidence)
        happens in client.py._completion_speculative().
        """
        candidates = self._get_candidates(
            max_cost, required_capabilities, required_compliance, context
        )
        candidate_ids = [m.model_id for m in candidates]
        predictions = self.predictor.predict(messages, candidate_ids) if candidate_ids else []
        cost_map = {m.model_id: m.cost_per_1k_total for m in candidates}

        qualifying = [p for p in predictions if p.predicted_quality >= min_quality]
        if qualifying:
            return min(qualifying, key=lambda p: cost_map.get(p.model_id, float("inf"))).model_id
        cheapest = self.registry.get_cheapest()
        if cheapest:
            return cheapest.model_id
        raise ValueError("No models available for speculative routing")

    def _get_candidates(
        self,
        max_cost: float | None,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext | None,
    ) -> list[Any]:
        """Filter and return candidate models for execution strategies."""
        if max_cost is not None:
            candidates = self.registry.filter_by_cost(max_cost)
        else:
            candidates = self.registry.list_models()
        candidates = self._apply_business_rules(candidates, context)
        candidates = self._filter_by_capabilities(candidates, required_capabilities)
        candidates = self._filter_by_compliance(candidates, required_compliance)
        return candidates

    def get_parallel_candidates(
        self,
        messages: list[dict[str, str]],
        n_candidates: int = 2,
        max_cost: float | None = None,
        min_quality: float = 0.0,
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
    ) -> list[str]:
        """Return top N diverse model candidates to run in parallel.

        Selects models that span cost-quality space:
          1. Cheapest model (cost-efficient)
          2. Highest quality model (best quality)
        More than 2 is wasteful — most value comes from comparing cheap vs best.
        Falls back to registered quality_score when predictor confidence is low.
        """
        candidates = self._get_candidates(
            max_cost, required_capabilities, required_compliance, context
        )
        if not candidates:
            cheapest = self.registry.get_cheapest()
            return [cheapest.model_id] if cheapest else []

        candidate_ids = [m.model_id for m in candidates]
        predictions = self.predictor.predict(messages, candidate_ids)
        qual_map = {m.model_id: m.quality_score for m in candidates}

        result: list[str] = []

        pred_qual = {p.model_id: p.predicted_quality for p in predictions}
        lowest_conf = min((p.confidence for p in predictions), default=1.0)

        # On cold start (low confidence), use registered quality_score instead
        if lowest_conf < 0.05:
            effective_quality = qual_map
        else:
            effective_quality = pred_qual

        # Cheapest model
        cheapest = min(candidates, key=lambda m: m.cost_per_1k_total)
        if cheapest.model_id not in result:
            result.append(cheapest.model_id)

        # Highest quality (by effective quality)
        sorted_by_qual = sorted(predictions, key=lambda p: effective_quality.get(p.model_id, 0.0), reverse=True)
        for p in sorted_by_qual:
            if p.model_id not in result:
                result.append(p.model_id)
                if len(result) >= n_candidates:
                    break

        return result[:n_candidates]

    def get_speculative_plan(
        self,
        messages: list[dict[str, str]],
        min_quality: float = 0.0,
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
    ) -> tuple[str, str | None]:
        """Return (cheap_model, expensive_model_or_None) for speculative execution.

        The cheap model is the cheapest overall (cost-efficient).
        The expensive model is the highest-quality model (escalation target).
        Falls back to registered quality_score when predictor confidence is low.
        """
        candidates = self._get_candidates(
            None, required_capabilities, required_compliance, context
        )
        if not candidates:
            cheapest = self.registry.get_cheapest()
            return (cheapest.model_id, None) if cheapest else ("", None)

        # On cold start, use registered quality_score instead of predictor
        candidate_ids = [m.model_id for m in candidates]
        predictions = self.predictor.predict(messages, candidate_ids)
        qual_map = {m.model_id: m.quality_score for m in candidates}

        lowest_conf = min((p.confidence for p in predictions), default=1.0)
        if lowest_conf < 0.05:
            effective_quality = qual_map
        else:
            effective_quality = {p.model_id: p.predicted_quality for p in predictions}

        # Cheapest model (by cost)
        cheapest = min(candidates, key=lambda m: m.cost_per_1k_total)
        cheap_id = cheapest.model_id

        # Highest quality model (by effective quality, different from cheap)
        sorted_models = sorted(candidates, key=lambda m: effective_quality.get(m.model_id, 0.0), reverse=True)
        expensive_id = None
        for m in sorted_models:
            if m.model_id != cheap_id:
                expensive_id = m.model_id
                break

        return (cheap_id, expensive_id)

    def get_cascade_models(
        self,
        min_quality: float = 0.0,
        max_tiers: int = 3,
        required_capabilities: set[str] | None = None,
    ) -> list[str]:
        """
        Get ordered list of models for cascade execution.

        Returns models sorted by cost (cheapest first) that can
        be tried in sequence during cascade routing.

        Args:
            min_quality: Minimum quality threshold.
            max_tiers: Maximum number of cascade tiers.
            required_capabilities: Capabilities each model must support.

        Returns:
            List of model IDs ordered by cost (cheapest first).
        """
        candidates = self.registry.filter_by_quality(min_quality)
        if required_capabilities:
            candidates = [c for c in candidates if required_capabilities.issubset(c.capabilities)]
        sorted_models = sorted(candidates, key=lambda m: m.cost_per_1k_total)
        return [m.model_id for m in sorted_models[:max_tiers]]

    def _route_provisioned_first(
        self,
        messages: list[dict[str, str]],
        max_cost: float | None,
        min_quality: float,
        required_capabilities: set[str] | None = None,
        required_compliance: set[str] | None = None,
        context: RouteContext | None = None,
    ) -> str:
        """
        PROVISIONED_FIRST strategy: route to provisioned capacity first.

        Checks provisioned models in cost order. If capacity available,
        uses provisioned model (marginal cost = $0). If all provisioned
        capacity exhausted, falls through to on-demand routing using
        quality prediction.

        Raises CapacityExhaustedError if no provisioned capacity AND
        no on-demand models are available.
        """
        from routesmith.config import CostModel
        from routesmith.exceptions import CapacityExhaustedError

        # Get all models, apply filters
        candidates = self.registry.list_models()
        candidates = self._apply_business_rules(candidates, context)
        candidates = self._filter_by_capabilities(candidates, required_capabilities)
        candidates = self._filter_by_compliance(candidates, required_compliance)

        # Separate into provisioned and on-demand
        provisioned = [m for m in candidates if m.cost_model == CostModel.PROVISIONED]
        on_demand = [m for m in candidates if m.cost_model != CostModel.PROVISIONED]

        # Try provisioned models (sorted by cost for deterministic behavior)
        provisioned.sort(key=lambda m: m.provisioned_hourly_cost)
        for model in provisioned:
            tracker = self.registry.get_capacity_tracker(model.model_id)
            if tracker is not None:
                if tracker.available():
                    tracker.record_request()
                    return model.model_id
                else:
                    tracker.mark_overflow()

        # All provisioned capacity exhausted — fall through to on-demand
        if not on_demand:
            provisioned_ids = [m.model_id for m in provisioned]
            raise CapacityExhaustedError(
                model_id=provisioned_ids[0] if provisioned_ids else "unknown",
            )

        # Use direct routing on on-demand models
        if max_cost is not None:
            on_demand = [m for m in on_demand if m.cost_per_1k_total <= max_cost]

        candidate_ids = [m.model_id for m in on_demand]
        predictions = self.predictor.predict(messages, candidate_ids)
        cost_map = {m.model_id: m.cost_per_1k_total for m in on_demand}

        qualifying = [p for p in predictions if p.predicted_quality >= min_quality]
        if qualifying:
            return min(qualifying, key=lambda p: cost_map.get(p.model_id, float("inf"))).model_id

        if predictions:
            return max(predictions, key=lambda p: p.predicted_quality).model_id

        raise ValueError("No models available for provisioned-first routing")
