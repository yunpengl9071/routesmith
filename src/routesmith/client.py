"""Main RouteSmith client - drop-in replacement for LLM API calls."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import litellm
from litellm import ModelResponse

from routesmith.cache.semantic import SemanticCache
from routesmith.config import (
    BudgetBehavior,
    RouteContext,
    RouteSmithConfig,
    RoutingStrategy,
)
from routesmith.exceptions import BudgetExceededError
from routesmith.feedback.collector import FeedbackCollector
from routesmith.registry.models import ModelRegistry
from routesmith.strategy.circuit_breaker import CircuitBreaker
from routesmith.strategy.router import Router
from routesmith.utils.logging import RouteSmithLogger, setup_logger
from routesmith.utils.retry import RetryExhaustedError, retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class RoutingMetadata:
    """Metadata about a routing decision for transparency."""

    request_id: str
    model_selected: str
    routing_strategy: str
    routing_reason: str
    routing_latency_ms: float
    estimated_cost_usd: float
    counterfactual_cost_usd: float  # What it would have cost with most expensive model
    cost_savings_usd: float
    models_considered: list[str]
    cache_hit: bool = False

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
        project: str | None = None,
    ) -> None:
        """
        Initialize RouteSmith.

        Args:
            config: Configuration for routing behavior, caching, and budget.
            registry: Pre-configured model registry. If None, creates empty registry.
            project: Project name for per-project cost isolation and stats.
        """
        self.config = config or RouteSmithConfig()
        self.registry = registry or ModelRegistry()
        self.project = project
        self.feedback = FeedbackCollector(self.config, registry=self.registry)
        self.router = Router(
            self.config, self.registry, storage=self.feedback._storage
        )
        self._request_count = 0
        self._total_cost = 0.0
        self._counterfactual_cost = 0.0  # Cost if always used most expensive model
        self._last_routing_metadata: RoutingMetadata | None = None
        self._budget_events: dict[str, int] = {
            "failures": 0,
            "fallbacks": 0,
            "queued": 0,
        }
        self._cost_model_counts: dict[str, dict[str, float]] = {}

        # Resilience: circuit breakers per model, structured logging
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._log: RouteSmithLogger = RouteSmithLogger(
            setup_logger("routesmith", json_format=True)
        )

        # Semantic cache (lazy-instantiated when enabled)
        self._cache: SemanticCache | None = None
        if self.config.cache.enabled:
            self._cache = SemanticCache(
                similarity_threshold=config.cache.similarity_threshold,
                ttl_seconds=config.cache.ttl_seconds,
                max_entries=config.cache.max_entries,
                embedding_model=config.cache.embedding_model,
            )

        # Resolve reward_fn from config (fail fast on bad expressions).
        self._reward_fn: Callable[..., float] | None = None
        if self.config.reward_fn is not None:
            self._reward_fn = self.config.reward_fn
        elif self.config.reward_expr is not None:
            from routesmith.feedback.reward import compile_reward_fn
            self._reward_fn = compile_reward_fn(self.config.reward_expr)

        # Load persisted predictor state if storage is configured.
        if self.config.feedback_storage_path:
            self._load_predictor_state()

    def _load_predictor_state(self) -> None:
        """Load persisted predictor weights from storage on startup."""
        if self.feedback._storage is None:
            return
        blob = self.feedback._storage.load_predictor_state(self.config.predictor_type)
        if blob is not None:
            predictor = self.router.predictor
            if hasattr(predictor, "load_state"):
                try:
                    predictor.load_state(blob)
                except Exception:
                    pass  # corrupt or incompatible state; cold start

    @classmethod
    def with_auto(
        cls,
        tradeoff: int = 7,
        providers: list[str] | None = None,
        include_all: bool = False,
        cache: bool = False,
        seed_quality: dict[str, float] | None = None,
        openrouter_api_key: str | None = None,
    ) -> RouteSmith:
        """Create a RouteSmith with auto-discovered models.

        Zero-config entry point. Automatically registers models from
        OpenRouter (if API key available) or a curated fallback list.
        The bandit refines cold-start quality scores from actual usage.

        Args:
            tradeoff: Default cost-quality tradeoff 0-10 (0=quality, 10=cost).
            providers: Filter to specific providers.
            include_all: Register all OpenRouter models, not just curated.
            cache: Enable semantic caching.
            seed_quality: Override initial quality scores per model.
            openrouter_api_key: OpenRouter API key for live pricing/catalog.
        """
        import os

        from routesmith.config import CacheConfig
        from routesmith.registry.discovery import discover_models

        config = RouteSmithConfig(
            cache=CacheConfig(enabled=cache),
        )

        # Store tradeoff for context injection
        config._auto_tradeoff = tradeoff  # type: ignore[attr-defined]

        # Resolve API key from env if not provided
        api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

        # Discover models
        models = discover_models(
            api_key=api_key,
            providers=providers,
            include_all=include_all,
        )

        rs = cls(config=config)

        for m in models:
            # Override quality seed if user provided
            quality = seed_quality.get(m["model_id"], m["quality_score"]) if seed_quality else m["quality_score"]

            rs.register_model(
                m["model_id"],
                cost_per_1k_input=m["cost_per_1k_input"],
                cost_per_1k_output=m["cost_per_1k_output"],
                quality_score=quality,
                context_window=m["context_window"],
                **{"supports_vision": m.get("supports_vision", False)},
            )

        return rs

    @classmethod
    def with_free_models(cls) -> RouteSmith:
        """Create a RouteSmith instance pre-configured with the best free models.

        Designed for the "get paid-model quality from free models" use case.
        Registers free-tier models from OpenRouter with zero cost and
        quality scores estimated from public benchmarks.
        """
        rs = cls()

        # Best free models available on OpenRouter (as of June 2026).
        # Quality scores are rough estimates from public benchmarks.
        free_models: list[tuple[str, float]] = [
            ("google/gemini-2.5-flash", 0.82),
            ("meta-llama/llama-3.3-70b-instruct:free", 0.78),
            ("qwen/qwen3-coder:free", 0.76),
            ("google/gemma-4-26b-a4b-it:free", 0.72),
            ("google/gemma-4-31b-it:free", 0.74),
            ("mistralai/ministral-3b-2512", 0.55),
            ("mistralai/ministral-8b-2512", 0.62),
            ("nvidia/nemotron-3-nano-30b-a3b:free", 0.68),
            ("nvidia/nemotron-3-super-120b-a12b:free", 0.80),
            ("openai/gpt-oss-20b:free", 0.70),
        ]

        for model_id, quality in free_models:
            rs.register_model(
                model_id,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                quality_score=quality,
            )

        return rs

    @staticmethod
    def _has_image_content(message: dict[str, Any]) -> bool:
        """Check if a message contains image content."""
        content = message.get("content")
        if isinstance(content, list):
            return any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in content
            )
        return False

    @staticmethod
    def _detect_required_capabilities(
        messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> set[str]:
        """Auto-detect required capabilities from messages and kwargs."""
        required: set[str] = set()
        if "tools" in kwargs or "functions" in kwargs:
            required.add("tool_calling")
        if any(RouteSmith._has_image_content(m) for m in messages):
            required.add("vision")
        return required

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
        predictor = self.router.predictor
        if hasattr(predictor, "add_arm"):
            # Pass quality_score to predictors that support it
            import inspect
            sig = inspect.signature(predictor.add_arm)
            if "quality_score" in sig.parameters:
                predictor.add_arm(model_id, quality_score=quality_score)
            else:
                predictor.add_arm(model_id)

    def deregister_model(self, model_id: str) -> None:
        """Remove a model from routing.

        Raises ValueError if it's the last registered model.
        Historical feedback records are preserved; predictor arm is retired.
        """
        if self.registry.get(model_id) is None:
            return
        if len(self.registry.list_models()) <= 1:
            raise ValueError(
                f"Cannot deregister '{model_id}': it is the last registered model."
            )
        self.registry.deregister(model_id)
        predictor = self.router.predictor
        if hasattr(predictor, "remove_arm"):
            predictor.remove_arm(model_id)
        self._persist_predictor_state()

    def _persist_predictor_state(self) -> None:
        """Serialize current predictor state to storage (non-fatal on failure)."""
        if not self.config.feedback_storage_path or not self.feedback._storage:
            return
        predictor = self.router.predictor
        if hasattr(predictor, "serialize_state"):
            try:
                blob = predictor.serialize_state()
                self.feedback._storage.save_predictor_state(
                    self.config.predictor_type, blob
                )
            except Exception:
                pass

    def completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        max_cost: float | None = None,
        min_quality: float | None = None,
        tradeoff: int | None = None,
        include_metadata: bool = False,
        context: RouteContext | None = None,
        required_compliance: set[str] | None = None,
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
            required_compliance: Required compliance tags (e.g., {"hipaa", "soc2"}).
            **kwargs: Additional arguments passed to litellm.completion().

        Returns:
            ModelResponse from the selected model. If include_metadata is True,
            response will have a routesmith_metadata attribute with routing details.
        """
        routing_start = time.perf_counter()
        self._request_count += 1
        request_id = uuid.uuid4().hex[:16]

        # Infer agent role from messages when context is provided without one.
        if context is not None and context.agent_role is None:
            if not hasattr(self, "_agent_inferencer"):
                from routesmith.predictor.agent_inferencer import AgentInferencer
                self._agent_inferencer = AgentInferencer()
            role, confidence = self._agent_inferencer.infer(messages)
            if role is not None:
                context = RouteContext(
                    agent_id=context.agent_id,
                    agent_role=role,
                    conversation_id=context.conversation_id,
                    turn_index=context.turn_index,
                    metadata={
                        **context.metadata,
                        "role_inferred": True,
                        "role_confidence": confidence,
                    },
                )

        # Determine routing strategy
        effective_strategy = strategy or self.config.default_strategy
        routing_reason = ""
        models_considered = [m.model_id for m in self.registry.list_models()]

        # Resolve tradeoff: explicit parameter > auto_tradeoff > default 7
        if tradeoff is not None:
            effective_tradeoff = tradeoff
        elif hasattr(self.config, '_auto_tradeoff'):
            effective_tradeoff = self.config._auto_tradeoff
        else:
            effective_tradeoff = 7

        # Inject tradeoff into routing context for bandit predictors
        if context is None:
            context = RouteContext()
        context.metadata["tradeoff"] = effective_tradeoff

        # Auto-detect required capabilities
        required_capabilities = self._detect_required_capabilities(messages, kwargs)

        # Budget enforcement
        budget = self.config.budget
        over_budget = (budget.max_cost_per_day is not None and self._total_cost >= budget.max_cost_per_day)

        if over_budget:
            if self.config.budget_behavior == BudgetBehavior.FAIL:
                self._budget_events["failures"] += 1
                raise BudgetExceededError(
                    "Budget exceeded.",
                    current_spend=self._total_cost,
                    limit=budget.max_cost_per_day or 0.0,
                )
            elif self.config.budget_behavior == BudgetBehavior.QUEUE:
                self._budget_events["queued"] += 1
                raise BudgetExceededError(
                    "Budget exceeded. Use acompletion() with QUEUE behavior for async queueing.",
                    current_spend=self._total_cost,
                    limit=budget.max_cost_per_day or 0.0,
                )
            # FALLBACK: handled below — select cheapest model

        # If specific model requested, skip routing
        if model:
            selected_model = model
            routing_reason = "explicit model specified"
        elif over_budget and self.config.budget_behavior == BudgetBehavior.FALLBACK:
            # FALLBACK: use cheapest model regardless of quality
            self._budget_events["fallbacks"] += 1
            cheapest = self.registry.get_cheapest()
            if cheapest:
                selected_model = cheapest.model_id
                routing_reason = f"budget exhausted, fallback to {cheapest.model_id}"
            else:
                raise BudgetExceededError(
                    "Budget exceeded and no fallback model available.",
                    current_spend=self._total_cost,
                    limit=budget.max_cost_per_day or 0.0,
                )
        else:
            # Route to optimal model
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
                max_cost=max_cost,
                min_quality=min_quality or self.config.budget.quality_threshold,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
            )
            routing_reason = self._get_routing_reason(
                effective_strategy, selected_model, max_cost, min_quality
            )

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000

        # Cache check: after routing (to know model_id), before LLM call
        cache_hit = False
        cached_response: ModelResponse | None = None
        if self._cache is not None:
            cached_entry = self._cache.get(messages, model_id=selected_model)
            if cached_entry is not None:
                cache_hit = True
                cached_response = cached_entry.response

        if cache_hit and cached_response is not None:
            response = cached_response
            # Attach request_id so feedback tracking can find this record
            response._routesmith_request_id = request_id  # type: ignore[attr-defined]
            self._log.info(
                "cache_hit", model_id=selected_model,
                request_id=request_id,
            )
        else:
            # Execute completion via LiteLLM with circuit breaker and retry
            breaker = self._circuit_breakers.get(selected_model)
            if breaker is None:
                breaker = CircuitBreaker(selected_model)
                self._circuit_breakers[selected_model] = breaker

            if not breaker.allow_request():
                from routesmith.exceptions import CircuitOpenError
                self._log.warning(
                    "circuit_open", model_id=selected_model,
                    request_id=request_id,
                )
                raise CircuitOpenError(
                    selected_model, retry_after=breaker.retry_after_seconds()
                )

            try:
                response = retry_with_backoff(
                    lambda: litellm.completion(
                        model=selected_model,
                        messages=messages,
                        **{**self.config.litellm_params, **kwargs},
                    ),
                    max_retries=2,
                    base_delay=1.0,
                )
                breaker.record_success()
                self._log.info(
                    "llm_call_success", model_id=selected_model,
                    request_id=request_id,
                )

                # Store in cache after successful LLM call
                if self._cache is not None:
                    try:
                        self._cache.put(
                            messages, response, model_id=selected_model
                        )
                    except Exception:
                        pass  # cache store failure is non-fatal
            except RetryExhaustedError as e:
                breaker.record_failure()
                self._log.error(
                    "llm_call_exhausted", model_id=selected_model,
                    request_id=request_id,
                )
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=str(e)
                )
                raise
            except Exception as e:
                breaker.record_failure()
                self._log.error(
                    "llm_call_failed", model_id=selected_model,
                    request_id=request_id,
                )
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=str(e)
                )
                raise

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

                # Track per-cost-model usage
                cm = model_config.cost_model.value
                if cm not in self._cost_model_counts:
                    self._cost_model_counts[cm] = {"request_count": 0.0, "total_cost": 0.0}
                self._cost_model_counts[cm]["request_count"] += 1
                self._cost_model_counts[cm]["total_cost"] += actual_cost

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
            request_id=request_id,
            model_selected=selected_model,
            routing_strategy=effective_strategy.value,
            routing_reason=routing_reason,
            routing_latency_ms=round(routing_latency_ms, 3),
            estimated_cost_usd=round(actual_cost, 6),
            counterfactual_cost_usd=round(counterfactual_cost, 6),
            cost_savings_usd=round(counterfactual_cost - actual_cost, 6),
            models_considered=models_considered,
            cache_hit=cache_hit,
        )
        self._last_routing_metadata = metadata

        # Attach metadata to response if requested
        if include_metadata:
            response.routesmith_metadata = metadata.to_dict()  # type: ignore[attr-defined]

        # Attach request_id to response for outcome tracking
        response._routesmith_request_id = request_id  # type: ignore[attr-defined]

        # Collect feedback sample
        total_latency_ms = (time.perf_counter() - routing_start) * 1000
        self.feedback.record(
            request_id=request_id,
            messages=messages,
            model=selected_model,
            response=response,
            latency_ms=total_latency_ms,
            agent_id=context.agent_id if context else None,
            agent_role=context.agent_role if context else None,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )

        # Periodic predictor state persistence every 50 updates
        updates = getattr(self.router.predictor, "_total_updates", None)
        if updates is None:
            updates = getattr(self.router.predictor, "_update_count", 0)
        if updates > 0 and updates % 50 == 0:
            self._persist_predictor_state()

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
        tradeoff: int | None = None,
        include_metadata: bool = False,
        context: RouteContext | None = None,
        required_compliance: set[str] | None = None,
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
            context: Optional routing context (agent_id, agent_role, etc.).
            required_compliance: Required compliance tags (e.g., {"hipaa", "soc2"}).
            **kwargs: Additional arguments passed to litellm.acompletion().

        Returns:
            ModelResponse from the selected model. If include_metadata is True,
            response will have a routesmith_metadata attribute with routing details.
        """
        routing_start = time.perf_counter()
        self._request_count += 1
        request_id = uuid.uuid4().hex[:16]

        # Infer agent role from messages when context is provided without one.
        if context is not None and context.agent_role is None:
            if not hasattr(self, "_agent_inferencer"):
                from routesmith.predictor.agent_inferencer import AgentInferencer
                self._agent_inferencer = AgentInferencer()
            role, confidence = self._agent_inferencer.infer(messages)
            if role is not None:
                context = RouteContext(
                    agent_id=context.agent_id,
                    agent_role=role,
                    conversation_id=context.conversation_id,
                    turn_index=context.turn_index,
                    metadata={
                        **context.metadata,
                        "role_inferred": True,
                        "role_confidence": confidence,
                    },
                )

        # Auto-detect required capabilities
        required_capabilities = self._detect_required_capabilities(messages, kwargs)

        # Budget enforcement
        budget = self.config.budget
        over_budget = (budget.max_cost_per_day is not None and self._total_cost >= budget.max_cost_per_day)

        if over_budget:
            if self.config.budget_behavior == BudgetBehavior.FAIL:
                self._budget_events["failures"] += 1
                raise BudgetExceededError(
                    "Budget exceeded.",
                    current_spend=self._total_cost,
                    limit=budget.max_cost_per_day or 0.0,
                )
            elif self.config.budget_behavior == BudgetBehavior.QUEUE:
                self._budget_events["queued"] += 1
                raise BudgetExceededError(
                    "Budget exceeded. Use acompletion() with QUEUE behavior for async queueing.",
                    current_spend=self._total_cost,
                    limit=budget.max_cost_per_day or 0.0,
                )
            # FALLBACK: handled below

        # Determine routing strategy
        effective_strategy = strategy or self.config.default_strategy
        routing_reason = ""
        models_considered = [m.model_id for m in self.registry.list_models()]

        # Resolve tradeoff: explicit parameter > auto_tradeoff > default 7
        if tradeoff is not None:
            effective_tradeoff = tradeoff
        elif hasattr(self.config, '_auto_tradeoff'):
            effective_tradeoff = self.config._auto_tradeoff
        else:
            effective_tradeoff = 7

        # Inject tradeoff into routing context for bandit predictors
        if context is None:
            context = RouteContext()
        context.metadata["tradeoff"] = effective_tradeoff

        # If specific model requested, skip routing
        if model:
            selected_model = model
            routing_reason = "explicit model specified"
        elif over_budget and self.config.budget_behavior == BudgetBehavior.FALLBACK:
            self._budget_events["fallbacks"] += 1
            cheapest = self.registry.get_cheapest()
            if cheapest:
                selected_model = cheapest.model_id
                routing_reason = f"budget exhausted, fallback to {cheapest.model_id}"
            else:
                raise BudgetExceededError(
                    "Budget exceeded and no fallback model available.",
                    current_spend=self._total_cost,
                    limit=budget.max_cost_per_day or 0.0,
                )
        else:
            # Route to optimal model
            selected_model = self.router.route(
                messages=messages,
                strategy=effective_strategy,
                max_cost=max_cost,
                min_quality=min_quality or self.config.budget.quality_threshold,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
            )
            routing_reason = self._get_routing_reason(
                effective_strategy, selected_model, max_cost, min_quality
            )

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000

        # Cache check: after routing (to know model_id), before LLM call
        cache_hit = False
        cached_response: ModelResponse | None = None
        if self._cache is not None:
            cached_entry = self._cache.get(messages, model_id=selected_model)
            if cached_entry is not None:
                cache_hit = True
                cached_response = cached_entry.response

        if cache_hit and cached_response is not None:
            response = cached_response
            response._routesmith_request_id = request_id  # type: ignore[attr-defined]
            self._log.info(
                "cache_hit", model_id=selected_model,
                request_id=request_id,
            )
        else:
            # Execute completion via LiteLLM
            try:
                response = await litellm.acompletion(
                    model=selected_model,
                    messages=messages,
                    **{**self.config.litellm_params, **kwargs},
                )
            except Exception as e:
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=str(e)
                )
                raise

            # Store in cache after successful async LLM call
            if self._cache is not None:
                try:
                    self._cache.put(
                        messages, response, model_id=selected_model
                    )
                except Exception:
                    pass  # cache store failure is non-fatal

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

                # Track per-cost-model usage
                cm = model_config.cost_model.value
                if cm not in self._cost_model_counts:
                    self._cost_model_counts[cm] = {"request_count": 0.0, "total_cost": 0.0}
                self._cost_model_counts[cm]["request_count"] += 1
                self._cost_model_counts[cm]["total_cost"] += actual_cost

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
            request_id=request_id,
            model_selected=selected_model,
            routing_strategy=effective_strategy.value,
            routing_reason=routing_reason,
            routing_latency_ms=round(routing_latency_ms, 3),
            estimated_cost_usd=round(actual_cost, 6),
            counterfactual_cost_usd=round(counterfactual_cost, 6),
            cost_savings_usd=round(counterfactual_cost - actual_cost, 6),
            models_considered=models_considered,
            cache_hit=cache_hit,
        )
        self._last_routing_metadata = metadata

        # Attach metadata to response if requested
        if include_metadata:
            response.routesmith_metadata = metadata.to_dict()  # type: ignore[attr-defined]

        # Attach request_id to response for outcome tracking
        response._routesmith_request_id = request_id  # type: ignore[attr-defined]

        # Collect feedback sample
        total_latency_ms = (time.perf_counter() - routing_start) * 1000
        self.feedback.record(
            request_id=request_id,
            messages=messages,
            model=selected_model,
            response=response,
            latency_ms=total_latency_ms,
            agent_id=context.agent_id if context else None,
            agent_role=context.agent_role if context else None,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )

        # Periodic predictor state persistence every 50 updates
        updates = getattr(self.router.predictor, "_total_updates", None)
        if updates is None:
            updates = getattr(self.router.predictor, "_update_count", 0)
        if updates > 0 and updates % 50 == 0:
            self._persist_predictor_state()

        return response

    def completion_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        strategy: RoutingStrategy | None = None,
        required_compliance: set[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Streaming completion with intelligent routing.

        Args:
            messages: List of message dicts.
            model: Specific model to use.
            strategy: Override default routing strategy.
            required_compliance: Required compliance tags.
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
                required_compliance=required_compliance,
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
        required_compliance: set[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming completion with intelligent routing.

        Args:
            messages: List of message dicts.
            model: Specific model to use.
            strategy: Override default routing strategy.
            required_compliance: Required compliance tags.
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
                required_compliance=required_compliance,
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
            "project": self.project,
            "budget_events": dict(self._budget_events),
            "by_cost_model": self._by_cost_model_stats(),
            "provisioned_utilization": self._provisioned_utilization_stats(),
        }

        if self._last_routing_metadata:
            result["last_routing"] = self._last_routing_metadata.to_dict()

        return result

    @property
    def last_routing_metadata(self) -> RoutingMetadata | None:
        """Get metadata from the last routing decision."""
        return self._last_routing_metadata

    def _by_cost_model_stats(self) -> dict[str, dict[str, float]]:
        """Aggregate per-cost-model request counts and costs."""
        return dict(self._cost_model_counts)

    def _provisioned_utilization_stats(self) -> dict[str, float]:
        """Get utilization per provisioned model."""
        result: dict[str, float] = {}
        for model in self.registry.list_models():
            if model.cost_model.value == "provisioned":
                tracker = self.registry.get_capacity_tracker(model.model_id)
                if tracker:
                    result[model.model_id] = tracker.current_utilization
        return result

    def record_outcome(
        self,
        request_id: str,
        success: bool | None = None,
        score: float | None = None,
        feedback: str | None = None,
    ) -> bool:
        """
        Record explicit feedback for a previous request.

        Use this to provide quality signals that improve future routing.

        Args:
            request_id: Request ID from response._routesmith_request_id
                or RoutingMetadata.request_id.
            success: Whether the response was successful.
            score: Explicit quality score (0-1).
            feedback: Free-text user feedback.

        Returns:
            True if the request was found, False otherwise.
        """
        found = self.feedback.record_outcome(
            request_id=request_id,
            success=success,
            score=score,
            feedback=feedback,
        )

        # Feed quality score to predictor for online learning
        quality = score
        if quality is None and success is not None:
            quality = 1.0 if success else 0.0

        if quality is not None:
            record = self.feedback.get_record_by_id(request_id)
            if record is not None:
                reward_override = None
                # Resolve per-role reward function, falling back to global reward_fn.
                effective_reward_fn = self.feedback.resolve_reward_fn(
                    agent_role=record.agent_role
                ) or self._reward_fn
                if effective_reward_fn is not None:
                    from routesmith.feedback.reward import build_reward_context
                    ctx = build_reward_context(
                        model_id=record.model_id,
                        quality=quality,
                        response=record.response,
                        latency_ms=record.latency_ms,
                        registry=self.registry,
                    )
                    try:
                        reward_override = float(effective_reward_fn(ctx))
                    except Exception as e:
                        logger.warning(
                            "reward_fn raised an error, skipping reward override: %s", e
                        )
                self.router.predictor.update(
                    messages=record.messages,
                    model_id=record.model_id,
                    actual_quality=quality,
                    reward_override=reward_override,
                )

        return found

    def recommend_model_for_agent(
        self,
        agent_role: str | None,
        min_samples: int = 50,
    ) -> dict[str, Any] | None:
        """Return the historically best model for an agent role.

        Returns None when agent_role is None or fewer than min_samples
        quality records exist for the role.

        Returns a dict with:
            model: str — recommended model_id
            confidence: float — 0-1, based on sample count
            sample_count: int — records for the recommended model
            avg_quality: float
            avg_cost_usd: float
            new_models_to_explore: list[str] — registered models with < min_samples data
        """
        if agent_role is None:
            return None

        if self.feedback._storage is None:
            return None

        records = self.feedback._storage.get_records_by_agent_role(agent_role)
        if len(records) < min_samples:
            return None

        model_quality: dict[str, list[float]] = defaultdict(list)
        for r in records:
            if r["quality_score"] is not None:
                model_quality[r["model_id"]].append(float(r["quality_score"]))

        registered = {m.model_id: m for m in self.registry.list_models()}
        # Filter to only registered models, then check per-model sample threshold.
        model_quality = defaultdict(
            list,
            {k: v for k, v in model_quality.items() if k in registered},
        )
        if not any(len(q) >= min_samples for q in model_quality.values()):
            return None

        best_model = None
        best_efficiency = -1.0
        model_stats: dict[str, dict[str, Any]] = {}

        for model_id, qualities in model_quality.items():
            model = registered[model_id]
            avg_quality = sum(qualities) / len(qualities)
            avg_cost = (model.cost_per_1k_input + model.cost_per_1k_output) / 2
            efficiency = avg_quality / max(avg_cost * 1000, 1e-6)
            model_stats[model_id] = {
                "avg_quality": avg_quality,
                "avg_cost_usd": avg_cost,
                "sample_count": len(qualities),
            }
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_model = model_id

        if best_model is None:
            return None

        new_models_to_explore = [
            m.model_id for m in self.registry.list_models()
            if len(model_quality.get(m.model_id, [])) < min_samples
            and m.model_id != best_model
        ]
        total_samples = sum(len(q) for q in model_quality.values())
        confidence = min(1.0, total_samples / (min_samples * 3))
        stats = model_stats[best_model]

        return {
            "model": best_model,
            "confidence": round(confidence, 3),
            "sample_count": stats["sample_count"],
            "avg_quality": round(stats["avg_quality"], 3),
            "avg_cost_usd": round(stats["avg_cost_usd"], 6),
            "new_models_to_explore": new_models_to_explore,
        }

    def register_reward_fn(self, agent_role: str, fn: Callable[..., float]) -> None:
        """Register a per-role reward function at runtime.

        Takes priority over the global reward_fn/reward_expr for this role.
        """
        self.config.reward_fns[agent_role] = fn

    def reset_stats(self) -> None:
        """Reset session statistics."""
        self._request_count = 0
        self._total_cost = 0.0
        self._counterfactual_cost = 0.0
        self._cost_model_counts = {}
        self._last_routing_metadata = None
