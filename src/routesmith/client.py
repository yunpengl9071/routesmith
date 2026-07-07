"""Main RouteSmith client - drop-in replacement for LLM API calls."""

from __future__ import annotations

import logging
import random
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import litellm
from litellm import ModelResponse

from routesmith.budget import BudgetTracker
from routesmith.cache.semantic import SemanticCache
from routesmith.config import (
    BudgetBehavior,
    RouteContext,
    RouteSmithConfig,
    RoutingStrategy,
)
from routesmith.exceptions import BudgetExceededError
from routesmith.explanation import format_explanation
from routesmith.feedback.audit import AuditStorage, NullAuditStorage
from routesmith.feedback.collector import FeedbackCollector
from routesmith.feedback.signals import implicit_quality
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
    fallback_from: str | None = None

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
        self.feedback = FeedbackCollector(self.config, registry=self.registry, project_id=self.project)
        self._audit_storage: AuditStorage | NullAuditStorage
        try:
            self._audit_storage = AuditStorage(
                self.config.cache.store_path if hasattr(self.config.cache, "store_path") and self.config.cache.store_path else "routesmith_feedback.db"
            )
        except Exception:
            self._audit_storage = NullAuditStorage()
        self.router = Router(
            self.config, self.registry, storage=self.feedback._storage
        )
        self._request_count = 0
        self._total_cost = 0.0
        self._budget = BudgetTracker(self.config.budget)
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

        # Conversation-scoped model stickiness
        self._conversation_models: dict[str, str] = {}

        # Quality poll sampler
        from routesmith.feedback.polls import PollSampler
        self._poll_sampler = PollSampler(
            base_rate=self.config.poll_sample_rate
        )
        self._polls: dict[str, Any] = {}  # poll_id -> Poll

        # Trust-but-verify shadow execution tracker
        from routesmith.verification import VerificationTracker
        self._verification = VerificationTracker()

        # LLM-as-judge evaluator (sampled, off by default)
        self._judge: Any = None
        if self.config.judge.enabled:
            from routesmith.feedback.judge import LLMJudge
            self._judge = LLMJudge(
                model=self.config.judge.judge_model,
                timeout_s=self.config.judge.timeout_s,
            )
            self._judge_rng = random.Random(self.config.predictor.seed)

        # Semantic cache (lazy-instantiated when enabled)
        self._cache: SemanticCache | None = None
        if self.config.cache.enabled:
            ns = self.config.cache.project_name or self.project or self.config.cache.namespace
            self._cache = SemanticCache(
                similarity_threshold=config.cache.similarity_threshold,
                ttl_seconds=config.cache.ttl_seconds,
                max_entries=config.cache.max_entries,
                embedding_model=config.cache.embedding_model,
                namespace=ns,
            )

        import importlib.util
        self._cache_semantic = importlib.util.find_spec("sentence_transformers") is not None
        if self.config.cache.enabled and not self._cache_semantic:
            logger.warning("sentence-transformers not installed; cache runs exact-match only "
                           "(pip install routesmith[cache] for semantic matching)")
        self._cache_hits = 0

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
        quality_score: float | None = None,
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
            quality_score: Expected quality score 0-1 (default: prior lookup or 0.8)
            latency_p50_ms: Median latency in milliseconds
            latency_p99_ms: 99th percentile latency in milliseconds
            context_window: Maximum context window size
            **kwargs: Additional model metadata
        """
        if quality_score is None:
            from routesmith.registry.priors import load_default_priors, lookup_prior
            quality_score = lookup_prior(model_id, load_default_priors()) or 0.8
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

        # Cache check (before routing, for exact/semantic match)
        if self._cache is not None and not kwargs.get("tools") and not kwargs.get("stream"):
            entry = self._cache.get(messages, model_id=model, semantic=self._cache_semantic)
            if entry is not None:
                self._cache_hits += 1
                import copy
                cached = copy.deepcopy(entry.response)
                cached._routesmith_request_id = request_id
                # Add basic metadata to cached response
                cached.routesmith_explanation = f"Cache hit (request {request_id})"
                if include_metadata:
                    cached.routesmith_metadata = {
                        "request_id": request_id,
                        "cache_hit": True,
                        "model_selected": entry.model_id,
                    }
                return cached

        # Rolling-window budget check (pre-flight), after cache check to allow free cache hits
        self._budget.check()

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
                waited = self._budget.wait_until_available()
                if waited > 0:
                    self._log.info("queue_resolved", waited_seconds=round(waited, 2))
            # FALLBACK: handled below — select cheapest model

        # Derive max_cost from max_cost_per_request before routing
        if max_cost is None and self.config.budget.max_cost_per_request is not None:
            est_tokens = sum(len(m.get("content", "")) for m in messages) / 4.0 + float(kwargs.get("max_tokens") or 1024)
            max_cost = self.config.budget.max_cost_per_request / est_tokens * 1000.0

        # If specific model requested, skip routing
        if model:
            selected_model = model
            routing_reason = "explicit model specified"
        elif context.conversation_id and context.conversation_id in self._conversation_models:
            # Conversation stickiness: reuse the model from the first turn
            selected_model = self._conversation_models[context.conversation_id]
            routing_reason = "conversation stickiness (reusing model from turn 1)"
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

        # Cascade execution — intercept before single-model LLM call
        if effective_strategy == RoutingStrategy.CASCADE:
            return self._completion_cascade(
                messages=messages,
                min_quality=min_quality or self.config.budget.quality_threshold,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
                request_id=request_id,
                include_metadata=include_metadata,
                **kwargs,
            )

        # Parallel execution — run multiple models, compare, pick best
        if effective_strategy == RoutingStrategy.PARALLEL:
            return self._completion_parallel(
                messages=messages,
                min_quality=min_quality or self.config.budget.quality_threshold,
                max_cost=max_cost,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
                request_id=request_id,
                include_metadata=include_metadata,
                **kwargs,
            )

        # Speculative execution — start cheap, escalate on low confidence
        if effective_strategy == RoutingStrategy.SPECULATIVE:
            return self._completion_speculative(
                messages=messages,
                min_quality=min_quality or self.config.budget.quality_threshold,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
                request_id=request_id,
                include_metadata=include_metadata,
                **kwargs,
            )

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000
        # Record conversation model for session stickiness
        if context and context.conversation_id and context.conversation_id not in self._conversation_models:
            self._conversation_models[context.conversation_id] = selected_model

        # Cache check: after routing (to know model_id), before LLM call
        cache_hit = False
        cached_response: ModelResponse | None = None
        if self._cache is not None and not kwargs.get("tools") and not kwargs.get("stream"):
            cached_entry = self._cache.get(messages, model_id=selected_model)
            if cached_entry is not None:
                cache_hit = True
                cached_response = cached_entry.response

        fallback_from: str | None = None

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
                        import copy
                        self._cache.put(
                            messages, copy.deepcopy(response), model_id=selected_model, semantic=self._cache_semantic
                        )
                    except Exception:
                        pass  # cache store failure is non-fatal
            except RetryExhaustedError as e:
                primary_error = e.__cause__ or e
                fb = self.config.fallback_model
                if fb and fb != selected_model and self.registry.get(fb) is not None:
                    logger.warning(
                        "Primary model %s failed (%s); retrying with fallback %s",
                        selected_model, primary_error, fb,
                    )
                    fallback_from = selected_model
                    selected_model = fb
                    breaker = self._circuit_breakers.get(selected_model)
                    if breaker is None:
                        breaker = CircuitBreaker(selected_model)
                        self._circuit_breakers[selected_model] = breaker
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

                        # Store in cache after successful fallback LLM call
                        if self._cache is not None:
                            try:
                                self._cache.put(
                                    messages, response, model_id=selected_model
                                )
                            except Exception:
                                pass
                    except Exception:
                        breaker.record_failure()
                        self._log.error(
                            "llm_call_failed", model_id=selected_model,
                            request_id=request_id,
                        )
                        self.feedback.record_outcome(
                            request_id=request_id, success=False, feedback=str(primary_error),
                        )
                        raise primary_error from None
                else:
                    breaker.record_failure()
                    self._log.error(
                        "llm_call_exhausted", model_id=selected_model,
                        request_id=request_id,
                    )
                    self.feedback.record_outcome(
                        request_id=request_id, success=False, feedback=str(e),
                    )
                    raise
            except Exception as e:
                primary_error = e
                fb = self.config.fallback_model
                if fb and fb != selected_model and self.registry.get(fb) is not None:
                    logger.warning(
                        "Primary model %s failed (%s); retrying with fallback %s",
                        selected_model, primary_error, fb,
                    )
                    fallback_from = selected_model
                    selected_model = fb
                    breaker = self._circuit_breakers.get(selected_model)
                    if breaker is None:
                        breaker = CircuitBreaker(selected_model)
                        self._circuit_breakers[selected_model] = breaker
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

                        # Store in cache after successful fallback LLM call
                        if self._cache is not None:
                            try:
                                self._cache.put(
                                    messages, response, model_id=selected_model
                                )
                            except Exception:
                                pass
                    except Exception:
                        breaker.record_failure()
                        self._log.error(
                            "llm_call_failed", model_id=selected_model,
                            request_id=request_id,
                        )
                        self.feedback.record_outcome(
                            request_id=request_id, success=False, feedback=str(primary_error),
                        )
                        raise primary_error from None
                else:
                    breaker.record_failure()
                    self._log.error(
                        "llm_call_failed", model_id=selected_model,
                        request_id=request_id,
                    )
                    self.feedback.record_outcome(
                        request_id=request_id, success=False, feedback=str(e),
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
                self._budget.record(actual_cost)

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
            fallback_from=fallback_from,
        )
        self._last_routing_metadata = metadata
        self._record_audit(metadata, context)

        # Attach metadata to response if requested
        if include_metadata:
            response.routesmith_metadata = metadata.to_dict()  # type: ignore[attr-defined]

        # Attach human-readable explanation to every response
        response.routesmith_explanation = format_explanation(  # type: ignore[attr-defined]
            metadata,
            qualifying=0,
            rejected=0,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )

        # Quality poll injection (adaptive sampling)
        convergence = context.metadata.get("convergence", 0.0) if context else 0.0
        if self._poll_sampler.should_sample(
            agent_id=context.agent_id if context else None,
            convergence=convergence,
        ):
            from routesmith.feedback.polls import generate_poll
            poll = generate_poll(
                request_id=request_id,
                model_id=selected_model,
                cost_usd=actual_cost,
            )
            poll_dict = poll.to_dict()
            response.routesmith_poll = poll_dict  # type: ignore[attr-defined]
            self._polls[request_id] = poll
            if self.config.on_poll is not None:
                self.config.on_poll(poll_dict)

        # Trust-but-verify shadow execution
        if self.config.verify_rate > 0.0 and random.random() < self.config.verify_rate:
            expensive = self.registry.get_best_quality()
            if expensive and expensive.model_id != selected_model:
                try:
                    from routesmith.verification import shadow_execute
                    cheap_cost = actual_cost

                    # Run shadow call to most expensive model
                    shadow_resp = litellm.completion(
                        model=expensive.model_id,
                        messages=messages,
                        **{**self.config.litellm_params, **{k: v for k, v in kwargs.items() if k != 'model'}},
                    )
                    shadow_tokens = shadow_resp.usage.prompt_tokens + shadow_resp.usage.completion_tokens if hasattr(shadow_resp, 'usage') and shadow_resp.usage else 0
                    expensive_cost = (shadow_tokens / 1000) * expensive.cost_per_1k_total
                    result = shadow_execute(
                        cheap_response=response,
                        cheap_model=selected_model,
                        expensive_response=shadow_resp,
                        expensive_model=expensive.model_id,
                        cheap_cost=cheap_cost,
                        expensive_cost=expensive_cost,
                    )
                    response.routesmith_verification = result  # type: ignore[attr-defined]
                    self._verification.record(
                        agent_role=context.agent_role if context else None,
                        cheap_model=selected_model,
                        expensive_model=expensive.model_id,
                        equivalent=result['equivalent'],
                        summary=result['summary'],
                        savings=result['savings'],
                    )
                except Exception:
                    pass  # shadow execution failure is non-fatal

        # Attach request_id to response for outcome tracking
        response._routesmith_request_id = request_id  # type: ignore[attr-defined]

        # Collect feedback sample
        total_latency_ms = (time.perf_counter() - routing_start) * 1000
        record = self.feedback.record(
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

        # Feed negative implicit signals to the predictor
        if record is not None and self.config.implicit_feedback_enabled:
            iq = implicit_quality(record.signals)
            if iq is not None:
                self.router.predictor.update(
                    messages, selected_model, actual_quality=iq
                )

        # Periodic predictor state persistence every 50 updates
        updates = getattr(self.router.predictor, "_total_updates", None)
        if updates is None:
            updates = getattr(self.router.predictor, "_update_count", 0)
        if updates > 0 and updates % 50 == 0:
            self._persist_predictor_state()

        # LLM-as-judge evaluation (sampled, synchronous)
        if self._judge is not None and self._judge_rng.random() < self.config.judge.sample_rate:
            text = response.choices[0].message.content or ""
            jscore = self._judge.score(messages, text)
            if jscore is not None:
                self.record_outcome(request_id, score=jscore)

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

        # Rolling-window budget check (pre-flight)
        self._budget.check()

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
                waited = self._budget.await_until_available()
                if waited > 0:
                    self._log.info("queue_resolved", waited_seconds=round(waited, 2))
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

        # Derive max_cost from max_cost_per_request before routing
        if max_cost is None and self.config.budget.max_cost_per_request is not None:
            est_tokens = sum(len(m.get("content", "")) for m in messages) / 4.0 + float(kwargs.get("max_tokens") or 1024)
            max_cost = self.config.budget.max_cost_per_request / est_tokens * 1000.0

        # If specific model requested, skip routing
        if model:
            selected_model = model
            routing_reason = "explicit model specified"
        elif context.conversation_id and context.conversation_id in self._conversation_models:
            # Conversation stickiness: reuse the model from the first turn
            selected_model = self._conversation_models[context.conversation_id]
            routing_reason = "conversation stickiness (reusing model from turn 1)"
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

        # Cascade execution — intercept before single-model LLM call (async)
        if effective_strategy == RoutingStrategy.CASCADE:
            return await self._acompletion_cascade(
                messages=messages,
                min_quality=min_quality or self.config.budget.quality_threshold,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
                request_id=request_id,
                include_metadata=include_metadata,
                **kwargs,
            )

        # Parallel execution (async)
        if effective_strategy == RoutingStrategy.PARALLEL:
            return await self._acompletion_parallel(
                messages=messages,
                min_quality=min_quality or self.config.budget.quality_threshold,
                max_cost=max_cost,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
                request_id=request_id,
                include_metadata=include_metadata,
                **kwargs,
            )

        # Speculative execution (async)
        if effective_strategy == RoutingStrategy.SPECULATIVE:
            return await self._acompletion_speculative(
                messages=messages,
                min_quality=min_quality or self.config.budget.quality_threshold,
                required_capabilities=required_capabilities or None,
                required_compliance=required_compliance,
                context=context,
                request_id=request_id,
                include_metadata=include_metadata,
                **kwargs,
            )

        routing_latency_ms = (time.perf_counter() - routing_start) * 1000
        # Record conversation model for session stickiness
        if context and context.conversation_id and context.conversation_id not in self._conversation_models:
            self._conversation_models[context.conversation_id] = selected_model

        # Cache check: after routing (to know model_id), before LLM call
        cache_hit = False
        cached_response: ModelResponse | None = None
        if self._cache is not None:
            cached_entry = self._cache.get(messages, model_id=selected_model)
            if cached_entry is not None:
                cache_hit = True
                cached_response = cached_entry.response

        fallback_from: str | None = None

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
                primary_error = e
                fb = self.config.fallback_model
                if fb and fb != selected_model and self.registry.get(fb) is not None:
                    logger.warning(
                        "Primary model %s failed (%s); retrying with fallback %s",
                        selected_model, primary_error, fb,
                    )
                    fallback_from = selected_model
                    selected_model = fb
                    try:
                        response = await litellm.acompletion(
                            model=selected_model,
                            messages=messages,
                            **{**self.config.litellm_params, **kwargs},
                        )
                    except Exception:
                        self.feedback.record_outcome(
                            request_id=request_id, success=False, feedback=str(primary_error),
                        )
                        raise primary_error from None
                else:
                    self.feedback.record_outcome(
                        request_id=request_id, success=False, feedback=str(e),
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
                self._budget.record(actual_cost)

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
            fallback_from=fallback_from,
        )
        self._last_routing_metadata = metadata
        self._record_audit(metadata, context)

        # Attach metadata to response if requested
        if include_metadata:
            response.routesmith_metadata = metadata.to_dict()  # type: ignore[attr-defined]

        # Attach human-readable explanation to every response
        response.routesmith_explanation = format_explanation(  # type: ignore[attr-defined]
            metadata,
            qualifying=0,
            rejected=0,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )

        # Quality poll injection (adaptive sampling)
        convergence = context.metadata.get("convergence", 0.0) if context else 0.0
        if self._poll_sampler.should_sample(
            agent_id=context.agent_id if context else None,
            convergence=convergence,
        ):
            from routesmith.feedback.polls import generate_poll
            poll = generate_poll(
                request_id=request_id,
                model_id=selected_model,
                cost_usd=actual_cost,
            )
            poll_dict = poll.to_dict()
            response.routesmith_poll = poll_dict  # type: ignore[attr-defined]
            self._polls[request_id] = poll
            if self.config.on_poll is not None:
                self.config.on_poll(poll_dict)

        # Trust-but-verify shadow execution
        if self.config.verify_rate > 0.0 and random.random() < self.config.verify_rate:
            expensive = self.registry.get_best_quality()
            if expensive and expensive.model_id != selected_model:
                try:
                    from routesmith.verification import shadow_execute
                    cheap_cost = actual_cost

                    # Run shadow call to most expensive model
                    shadow_resp = litellm.completion(
                        model=expensive.model_id,
                        messages=messages,
                        **{**self.config.litellm_params, **{k: v for k, v in kwargs.items() if k != 'model'}},
                    )
                    shadow_tokens = shadow_resp.usage.prompt_tokens + shadow_resp.usage.completion_tokens if hasattr(shadow_resp, 'usage') and shadow_resp.usage else 0
                    expensive_cost = (shadow_tokens / 1000) * expensive.cost_per_1k_total
                    result = shadow_execute(
                        cheap_response=response,
                        cheap_model=selected_model,
                        expensive_response=shadow_resp,
                        expensive_model=expensive.model_id,
                        cheap_cost=cheap_cost,
                        expensive_cost=expensive_cost,
                    )
                    response.routesmith_verification = result  # type: ignore[attr-defined]
                    self._verification.record(
                        agent_role=context.agent_role if context else None,
                        cheap_model=selected_model,
                        expensive_model=expensive.model_id,
                        equivalent=result['equivalent'],
                        summary=result['summary'],
                        savings=result['savings'],
                    )
                except Exception:
                    pass  # shadow execution failure is non-fatal

        # Attach request_id to response for outcome tracking
        response._routesmith_request_id = request_id  # type: ignore[attr-defined]

        # Collect feedback sample
        total_latency_ms = (time.perf_counter() - routing_start) * 1000
        record = self.feedback.record(
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

        # Feed negative implicit signals to the predictor
        if record is not None and self.config.implicit_feedback_enabled:
            iq = implicit_quality(record.signals)
            if iq is not None:
                self.router.predictor.update(
                    messages, selected_model, actual_quality=iq
                )

        # Periodic predictor state persistence every 50 updates
        updates = getattr(self.router.predictor, "_total_updates", None)
        if updates is None:
            updates = getattr(self.router.predictor, "_update_count", 0)
        if updates > 0 and updates % 50 == 0:
            self._persist_predictor_state()

        # LLM-as-judge evaluation (sampled, synchronous)
        if self._judge is not None and self._judge_rng.random() < self.config.judge.sample_rate:
            text = response.choices[0].message.content or ""
            jscore = self._judge.score(messages, text)
            if jscore is not None:
                self.record_outcome(request_id, score=jscore)

        return response

    def _execute_model(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        request_id: str,
        **kwargs: Any,
    ) -> tuple[ModelResponse | None, int, int, float, str | None]:
        """Execute a single model with retry, circuit breaker, and fallback.

        Never raises — returns error_message on failure.
        Returns (response, prompt_tokens, completion_tokens, latency_ms, error_message).
        """
        start = time.perf_counter()

        breaker = self._circuit_breakers.get(model_id)
        if breaker is None:
            breaker = CircuitBreaker(model_id)
            self._circuit_breakers[model_id] = breaker
        if not breaker.allow_request():
            return (None, 0, 0, 0, f"Circuit open for {model_id}")

        models_to_try = [model_id]
        fb = self.config.fallback_model
        if fb and fb != model_id and self.registry.get(fb) is not None:
            models_to_try.append(fb)

        last_error: str | None = None
        for i, m_id in enumerate(models_to_try):
            try:
                response = retry_with_backoff(
                    lambda: litellm.completion(
                        model=m_id,
                        messages=messages,
                        **{**self.config.litellm_params, **kwargs},
                    ),
                    max_retries=2,
                    base_delay=1.0,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                brk = self._circuit_breakers.get(m_id)
                if brk:
                    brk.record_success()

                if self._cache is not None:
                    try:
                        import copy
                        self._cache.put(
                            messages, copy.deepcopy(response), model_id=m_id, semantic=self._cache_semantic
                        )
                    except Exception:
                        pass

                pt = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
                ct = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
                return (response, pt, ct, latency_ms, None)
            except Exception as e:
                err = getattr(e, '__cause__', e) if isinstance(e, RetryExhaustedError) else e
                last_error = str(err)
                brk = self._circuit_breakers.get(m_id)
                if brk:
                    brk.record_failure()
                if i == 0 and len(models_to_try) > 1:
                    logger.warning(
                        "Primary model %s failed (%s); trying fallback %s",
                        m_id, last_error, models_to_try[1],
                    )

        return (None, 0, 0, 0, last_error)

    async def _aexecute_model(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        request_id: str,
        **kwargs: Any,
    ) -> tuple[ModelResponse | None, int, int, float, str | None]:
        """Async version of _execute_model."""
        start = time.perf_counter()

        models_to_try = [model_id]
        fb = self.config.fallback_model
        if fb and fb != model_id and self.registry.get(fb) is not None:
            models_to_try.append(fb)

        last_error: str | None = None
        for i, m_id in enumerate(models_to_try):
            try:
                response = await litellm.acompletion(
                    model=m_id,
                    messages=messages,
                    **{**self.config.litellm_params, **kwargs},
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if self._cache is not None:
                    try:
                        self._cache.put(messages, response, model_id=m_id)
                    except Exception:
                        pass

                pt = getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0
                ct = getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0
                return (response, pt, ct, latency_ms, None)
            except Exception as e:
                last_error = str(e)
                if i == 0 and len(models_to_try) > 1:
                    logger.warning(
                        "Primary model %s failed (%s); trying fallback %s",
                        m_id, last_error, models_to_try[1],
                    )

        return (None, 0, 0, 0, last_error)

    def _completion_cascade(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute cascade routing with per-tier verification.

        Tries models from cheapest to most expensive, verifying each response
        before accepting. Escalates on hard-negative signals or judge rejection.
        Never fails a request that produced output — returns last response if
        all tiers are rejected.
        """
        tiers = self.router.get_cascade_models(
            min_quality=min_quality,
            max_tiers=self.config.cascade_max_tiers,
            required_capabilities=required_capabilities,
        )

        if not tiers:
            best = self.registry.get_best_quality()
            if best is None:
                raise ValueError("No models available for cascade fallback")
            response, pt, ct, lat, err = self._execute_model(
                best.model_id, messages, request_id, **kwargs
            )
            if err:
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=err,
                )
                raise RuntimeError(f"Cascade fallback model {best.model_id} failed: {err}")
            return response

        if hasattr(self.feedback, '_signal_extractor') and self.feedback._signal_extractor is not None:
            extractor = self.feedback._signal_extractor
        else:
            from routesmith.feedback.signals import SignalExtractor
            extractor = SignalExtractor()

        last_response = None
        last_model: str | None = None
        last_prompt_tokens = 0
        last_completion_tokens = 0

        cascaded_tiers: list[dict[str, Any]] = []
        escalated_count = 0
        total_cost = 0.0

        for tier_model in tiers:
            response, pt, ct, lat, err = self._execute_model(
                tier_model, messages, request_id, **kwargs
            )

            tier_info: dict[str, Any] = {
                "model": tier_model,
                "succeeded": err is None,
            }

            if err:
                tier_info["error"] = err
                cascaded_tiers.append(tier_info)
                cfg = self.registry.get(tier_model)
                if cfg:
                    total_cost += (pt / 1000) * cfg.cost_per_1k_input + (ct / 1000) * cfg.cost_per_1k_output
                continue

            signals = extractor.extract(response, tier_model, lat)
            hard_negatives = {"error_detected", "refusal_detected", "empty_response"}
            reject = any(
                s.signal_value < 0.5
                for s in signals
                if s.signal_name in hard_negatives
            )

            if reject:
                tier_info["rejected"] = True
                tier_info["reason"] = "hard_negative_signal"
                cascaded_tiers.append(tier_info)
                escalated_count += 1
                cfg = self.registry.get(tier_model)
                if cfg:
                    total_cost += (pt / 1000) * cfg.cost_per_1k_input + (ct / 1000) * cfg.cost_per_1k_output
                last_response = response
                last_model = tier_model
                last_prompt_tokens = pt
                last_completion_tokens = ct

                # Record feedback so the bandit learns from the rejection
                reject_request_id = uuid.uuid4().hex[:16]
                reject_record = self.feedback.record(
                    request_id=reject_request_id,
                    messages=messages,
                    model=tier_model,
                    response=response,
                    latency_ms=lat,
                    agent_id=context.agent_id if context else None,
                    agent_role=context.agent_role if context else None,
                    conversation_id=context.conversation_id if context else None,
                    turn_index=context.turn_index if context else None,
                )
                if reject_record is not None:
                    self.record_outcome(request_id=reject_request_id, score=0.1)

                continue

            if self._judge is not None:
                text = response.choices[0].message.content or ""
                jscore = self._judge.score(messages, text)
                if jscore is not None and jscore < self.config.cascade_accept_threshold:
                    tier_info["rejected"] = True
                    tier_info["reason"] = "judge_reject"
                    tier_info["judge_score"] = jscore
                    cascaded_tiers.append(tier_info)
                    escalated_count += 1
                    cfg = self.registry.get(tier_model)
                    if cfg:
                        total_cost += (pt / 1000) * cfg.cost_per_1k_input + (ct / 1000) * cfg.cost_per_1k_output
                    last_response = response
                    last_model = tier_model
                    last_prompt_tokens = pt
                    last_completion_tokens = ct

                    # Record feedback so the bandit learns from the rejection
                    judge_request_id = uuid.uuid4().hex[:16]
                    judge_record = self.feedback.record(
                        request_id=judge_request_id,
                        messages=messages,
                        model=tier_model,
                        response=response,
                        latency_ms=lat,
                        agent_id=context.agent_id if context else None,
                        agent_role=context.agent_role if context else None,
                        conversation_id=context.conversation_id if context else None,
                        turn_index=context.turn_index if context else None,
                    )
                    if judge_record is not None:
                        self.record_outcome(request_id=judge_request_id, score=0.1)

                    continue

            tier_info["accepted"] = True
            cascaded_tiers.append(tier_info)

            cfg = self.registry.get(tier_model)
            if cfg:
                accepted_cost = (
                    (pt / 1000) * cfg.cost_per_1k_input
                    + (ct / 1000) * cfg.cost_per_1k_output
                )
                total_cost += accepted_cost
                self._total_cost += total_cost
                self._budget.record(accepted_cost)

            cascade_metadata: dict[str, Any] = {
                "strategy": "cascade",
                "routing_reason": f"cascade accepted {tier_model} after "
                                  f"{len(cascaded_tiers)} tier(s)",
                "tiers_tried": cascaded_tiers,
                "escalations": escalated_count,
            }
            if kwargs.get("include_metadata", False):
                response.routesmith_metadata = cascade_metadata
            response.routesmith_explanation = (
                f"Cascade: tried {len(cascaded_tiers)} tier(s), "
                f"{escalated_count} escalation(s), "
                f"accepted {tier_model}"
            )
            response._routesmith_request_id = request_id

            self.feedback.record(
                request_id=request_id,
                messages=messages,
                model=tier_model,
                response=response,
                latency_ms=lat,
                agent_id=context.agent_id if context else None,
                agent_role=context.agent_role if context else None,
                conversation_id=context.conversation_id if context else None,
                turn_index=context.turn_index if context else None,
            )

            return response

        if last_response is None:
            raise ValueError(
                "Cascade exhausted: no model produced a response"
            )

        cfg = self.registry.get(last_model)
        if cfg:
            total_cost += (
                (last_prompt_tokens / 1000) * cfg.cost_per_1k_input
                + (last_completion_tokens / 1000) * cfg.cost_per_1k_output
            )
            self._total_cost += total_cost
            self._budget.record(
                (last_prompt_tokens / 1000) * cfg.cost_per_1k_input
                + (last_completion_tokens / 1000) * cfg.cost_per_1k_output
            )

        last_response.routesmith_metadata = {
            "strategy": "cascade",
            "routing_reason": f"cascade exhausted after {len(cascaded_tiers)} tiers",
            "tiers_tried": cascaded_tiers,
            "escalations": escalated_count,
            "cascade_exhausted": True,
        }
        last_response.routesmith_explanation = (
            f"Cascade exhausted ({len(cascaded_tiers)} tiers tried)"
        )
        last_response._routesmith_request_id = request_id
        return last_response

    async def _acompletion_cascade(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Async version of _completion_cascade."""
        tiers = self.router.get_cascade_models(
            min_quality=min_quality,
            max_tiers=self.config.cascade_max_tiers,
            required_capabilities=required_capabilities,
        )

        if not tiers:
            best = self.registry.get_best_quality()
            if best is None:
                raise ValueError("No models available for cascade fallback")
            response, pt, ct, lat, err = await self._aexecute_model(
                best.model_id, messages, request_id, **kwargs
            )
            if err:
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=err,
                )
                raise RuntimeError(f"Cascade fallback model {best.model_id} failed: {err}")
            return response

        if hasattr(self.feedback, '_signal_extractor') and self.feedback._signal_extractor is not None:
            extractor = self.feedback._signal_extractor
        else:
            from routesmith.feedback.signals import SignalExtractor
            extractor = SignalExtractor()

        last_response = None
        last_model: str | None = None
        last_prompt_tokens = 0
        last_completion_tokens = 0

        cascaded_tiers: list[dict[str, Any]] = []
        escalated_count = 0
        total_cost = 0.0

        for tier_model in tiers:
            response, pt, ct, lat, err = await self._aexecute_model(
                tier_model, messages, request_id, **kwargs
            )

            tier_info: dict[str, Any] = {
                "model": tier_model,
                "succeeded": err is None,
            }

            if err:
                tier_info["error"] = err
                cascaded_tiers.append(tier_info)
                cfg = self.registry.get(tier_model)
                if cfg:
                    total_cost += (pt / 1000) * cfg.cost_per_1k_input + (ct / 1000) * cfg.cost_per_1k_output
                continue

            signals = extractor.extract(response, tier_model, lat)
            hard_negatives = {"error_detected", "refusal_detected", "empty_response"}
            reject = any(
                s.signal_value < 0.5
                for s in signals
                if s.signal_name in hard_negatives
            )

            if reject:
                tier_info["rejected"] = True
                tier_info["reason"] = "hard_negative_signal"
                cascaded_tiers.append(tier_info)
                escalated_count += 1
                cfg = self.registry.get(tier_model)
                if cfg:
                    total_cost += (pt / 1000) * cfg.cost_per_1k_input + (ct / 1000) * cfg.cost_per_1k_output
                last_response = response
                last_model = tier_model
                last_prompt_tokens = pt
                last_completion_tokens = ct

                # Record feedback so the bandit learns from the rejection
                reject_request_id = uuid.uuid4().hex[:16]
                reject_record = self.feedback.record(
                    request_id=reject_request_id,
                    messages=messages,
                    model=tier_model,
                    response=response,
                    latency_ms=lat,
                    agent_id=context.agent_id if context else None,
                    agent_role=context.agent_role if context else None,
                    conversation_id=context.conversation_id if context else None,
                    turn_index=context.turn_index if context else None,
                )
                if reject_record is not None:
                    self.record_outcome(request_id=reject_request_id, score=0.1)

                continue

            if self._judge is not None:
                text = response.choices[0].message.content or ""
                jscore = self._judge.score(messages, text)
                if jscore is not None and jscore < self.config.cascade_accept_threshold:
                    tier_info["rejected"] = True
                    tier_info["reason"] = "judge_reject"
                    tier_info["judge_score"] = jscore
                    cascaded_tiers.append(tier_info)
                    escalated_count += 1
                    cfg = self.registry.get(tier_model)
                    if cfg:
                        total_cost += (pt / 1000) * cfg.cost_per_1k_input + (ct / 1000) * cfg.cost_per_1k_output
                    last_response = response
                    last_model = tier_model
                    last_prompt_tokens = pt
                    last_completion_tokens = ct

                    # Record feedback so the bandit learns from the rejection
                    judge_request_id = uuid.uuid4().hex[:16]
                    judge_record = self.feedback.record(
                        request_id=judge_request_id,
                        messages=messages,
                        model=tier_model,
                        response=response,
                        latency_ms=lat,
                        agent_id=context.agent_id if context else None,
                        agent_role=context.agent_role if context else None,
                        conversation_id=context.conversation_id if context else None,
                        turn_index=context.turn_index if context else None,
                    )
                    if judge_record is not None:
                        self.record_outcome(request_id=judge_request_id, score=0.1)

                    continue

            tier_info["accepted"] = True
            cascaded_tiers.append(tier_info)

            cfg = self.registry.get(tier_model)
            if cfg:
                accepted_cost = (
                    (pt / 1000) * cfg.cost_per_1k_input
                    + (ct / 1000) * cfg.cost_per_1k_output
                )
                total_cost += accepted_cost
                self._total_cost += total_cost
                self._budget.record(accepted_cost)

            cascade_metadata: dict[str, Any] = {
                "strategy": "cascade",
                "routing_reason": f"cascade accepted {tier_model} after "
                                  f"{len(cascaded_tiers)} tier(s)",
                "tiers_tried": cascaded_tiers,
                "escalations": escalated_count,
            }
            if kwargs.get("include_metadata", False):
                response.routesmith_metadata = cascade_metadata
            response.routesmith_explanation = (
                f"Cascade: tried {len(cascaded_tiers)} tier(s), "
                f"{escalated_count} escalation(s), "
                f"accepted {tier_model}"
            )
            response._routesmith_request_id = request_id

            self.feedback.record(
                request_id=request_id,
                messages=messages,
                model=tier_model,
                response=response,
                latency_ms=lat,
                agent_id=context.agent_id if context else None,
                agent_role=context.agent_role if context else None,
                conversation_id=context.conversation_id if context else None,
                turn_index=context.turn_index if context else None,
            )

            return response

        if last_response is None:
            raise ValueError(
                "Cascade exhausted: no model produced a response"
            )

        cfg = self.registry.get(last_model)
        if cfg:
            total_cost += (
                (last_prompt_tokens / 1000) * cfg.cost_per_1k_input
                + (last_completion_tokens / 1000) * cfg.cost_per_1k_output
            )
            self._total_cost += total_cost
            self._budget.record(
                (last_prompt_tokens / 1000) * cfg.cost_per_1k_input
                + (last_completion_tokens / 1000) * cfg.cost_per_1k_output
            )

        last_response.routesmith_metadata = {
            "strategy": "cascade",
            "routing_reason": f"cascade exhausted after {len(cascaded_tiers)} tiers",
            "tiers_tried": cascaded_tiers,
            "escalations": escalated_count,
            "cascade_exhausted": True,
        }
        last_response.routesmith_explanation = (
            f"Cascade exhausted ({len(cascaded_tiers)} tiers tried)"
        )
        last_response._routesmith_request_id = request_id
        return last_response

    def _completion_parallel(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        max_cost: float | None,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute parallel strategy: run 2 models concurrently, pick best response."""
        import concurrent.futures

        candidates = self.router.get_parallel_candidates(
            messages=messages,
            n_candidates=2,
            max_cost=max_cost,
            min_quality=min_quality,
            required_capabilities=required_capabilities,
            required_compliance=required_compliance,
            context=context,
        )
        if not candidates:
            return self._execute_single(
                messages, min_quality, required_capabilities,
                required_compliance, context, request_id, **kwargs
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            futures = {
                pool.submit(
                    self._execute_model, m_id, messages, request_id, **kwargs
                ): m_id for m_id in candidates
            }
            results: dict[str, tuple] = {}
            for future in concurrent.futures.as_completed(futures):
                m_id = futures[future]
                try:
                    results[m_id] = future.result()
                except Exception as e:
                    results[m_id] = (None, 0, 0, 0, str(e))

        # Pick best among successful responses
        cost_map = {}
        for m_id in candidates:
            cfg = self.registry.get(m_id)
            if cfg:
                cost_map[m_id] = cfg.cost_per_1k_total

        successful = {m_id: r for m_id, r in results.items() if r[0] is not None}
        if not successful:
            raise RuntimeError(f"All parallel models failed: {list(results.values())}")

        if len(successful) == 1:
            m_id, (resp, pt, ct, lat, err) = next(iter(successful.items()))
        else:
            from routesmith.verification import compare_responses
            m_ids = list(successful.keys())
            resp_a, pt_a, ct_a, lat_a, _ = successful[m_ids[0]]
            resp_b, pt_b, ct_b, lat_b, _ = successful[m_ids[1]]
            comp = compare_responses(resp_a, resp_b)
            # If responses are equivalent, pick cheaper
            if comp["equivalent"]:
                cost_a = cost_map.get(m_ids[0], float("inf"))
                cost_b = cost_map.get(m_ids[1], float("inf"))
                if cost_a <= cost_b:
                    m_id, resp, pt, ct, lat = m_ids[0], resp_a, pt_a, ct_a, lat_a
                else:
                    m_id, resp, pt, ct, lat = m_ids[1], resp_b, pt_b, ct_b, lat_b
            else:
                # Not equivalent: pick higher-quality model
                qual_map = {}
                candidate_ids = [m for m in m_ids if m in cost_map]
                preds = self.router.predictor.predict(messages, candidate_ids) if candidate_ids else []
                for p in preds:
                    qual_map[p.model_id] = p.predicted_quality
                qual_a = qual_map.get(m_ids[0], 0.0)
                qual_b = qual_map.get(m_ids[1], 0.0)
                if qual_a >= qual_b:
                    m_id, resp, pt, ct, lat = m_ids[0], resp_a, pt_a, ct_a, lat_a
                else:
                    m_id, resp, pt, ct, lat = m_ids[1], resp_b, pt_b, ct_b, lat_b

        # Track cost for the selected response
        self._track_cost(m_id, pt, ct, context, request_id)
        resp._routesmith_request_id = request_id
        metadata = RoutingMetadata(
            request_id=request_id,
            model_selected=m_id,
            routing_strategy="parallel",
            routing_reason=f"parallel: compared {len(successful)} model(s), selected {m_id}",
            routing_latency_ms=0.0,
            estimated_cost_usd=0.0,
            counterfactual_cost_usd=0.0,
            cost_savings_usd=0.0,
            models_considered=candidates,
        )
        self._last_routing_metadata = metadata
        self._record_audit(metadata, context)
        if kwargs.get("include_metadata", False):
            resp.routesmith_metadata = metadata.to_dict()
        resp.routesmith_explanation = (
            f"Parallel: ran {len(candidates)} model(s), "
            f"selected {m_id}"
        )
        self.feedback.record(
            request_id=request_id, messages=messages, model=m_id,
            response=resp, latency_ms=lat,
            agent_id=context.agent_id if context else None,
            agent_role=context.agent_role if context else None,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )
        return resp

    async def _acompletion_parallel(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        max_cost: float | None,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Async parallel execution: run 2 models concurrently via asyncio."""
        import asyncio

        candidates = self.router.get_parallel_candidates(
            messages=messages, n_candidates=2, max_cost=max_cost,
            min_quality=min_quality,
            required_capabilities=required_capabilities,
            required_compliance=required_compliance, context=context,
        )
        if not candidates:
            return await self._aexecute_single(
                messages, min_quality, required_capabilities,
                required_compliance, context, request_id, **kwargs
            )

        async def run(m_id: str) -> tuple[str, Any]:
            result = await self._aexecute_model(m_id, messages, request_id, **kwargs)
            return (m_id, result)

        results_list = await asyncio.gather(*[run(m_id) for m_id in candidates], return_exceptions=True)
        results: dict[str, tuple] = {}
        for item in results_list:
            if isinstance(item, Exception):
                continue
            m_id, result = item
            results[m_id] = result

        cost_map = {}
        for m_id in candidates:
            cfg = self.registry.get(m_id)
            if cfg:
                cost_map[m_id] = cfg.cost_per_1k_total

        successful = {m_id: r for m_id, r in results.items() if r[0] is not None}
        if not successful:
            raise RuntimeError(f"All parallel models failed: {list(results.values())}")

        if len(successful) == 1:
            m_id, (resp, pt, ct, lat, _) = next(iter(successful.items()))
        else:
            from routesmith.verification import compare_responses
            m_ids = list(successful.keys())
            resp_a, pt_a, ct_a, lat_a, _ = successful[m_ids[0]]
            resp_b, pt_b, ct_b, lat_b, _ = successful[m_ids[1]]
            comp = compare_responses(resp_a, resp_b)
            if comp["equivalent"]:
                cost_a = cost_map.get(m_ids[0], float("inf"))
                cost_b = cost_map.get(m_ids[1], float("inf"))
                if cost_a <= cost_b:
                    m_id, resp, pt, ct, lat = m_ids[0], resp_a, pt_a, ct_a, lat_a
                else:
                    m_id, resp, pt, ct, lat = m_ids[1], resp_b, pt_b, ct_b, lat_b
            else:
                qual_map = {}
                candidate_ids = [m for m in m_ids if m in cost_map]
                preds = self.router.predictor.predict(messages, candidate_ids) if candidate_ids else []
                for p in preds:
                    qual_map[p.model_id] = p.predicted_quality
                qual_a = qual_map.get(m_ids[0], 0.0)
                qual_b = qual_map.get(m_ids[1], 0.0)
                if qual_a >= qual_b:
                    m_id, resp, pt, ct, lat = m_ids[0], resp_a, pt_a, ct_a, lat_a
                else:
                    m_id, resp, pt, ct, lat = m_ids[1], resp_b, pt_b, ct_b, lat_b

        self._track_cost(m_id, pt, ct, context, request_id)
        resp._routesmith_request_id = request_id
        metadata = RoutingMetadata(
            request_id=request_id, model_selected=m_id,
            routing_strategy="parallel",
            routing_reason=f"parallel: compared {len(successful)} model(s), selected {m_id}",
            routing_latency_ms=0.0, estimated_cost_usd=0.0,
            counterfactual_cost_usd=0.0, cost_savings_usd=0.0,
            models_considered=candidates,
        )
        self._last_routing_metadata = metadata
        self._record_audit(metadata, context)
        if kwargs.get("include_metadata", False):
            resp.routesmith_metadata = metadata.to_dict()
        resp.routesmith_explanation = (
            f"Parallel: ran {len(candidates)} model(s), selected {m_id}"
        )
        self.feedback.record(
            request_id=request_id, messages=messages, model=m_id,
            response=resp, latency_ms=lat,
            agent_id=context.agent_id if context else None,
            agent_role=context.agent_role if context else None,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )
        return resp

    def _completion_speculative(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Execute speculative strategy: start cheap, escalate on hard negatives."""
        cheap_id, expensive_id = self.router.get_speculative_plan(
            messages=messages, min_quality=min_quality,
            required_capabilities=required_capabilities,
            required_compliance=required_compliance, context=context,
        )
        if not cheap_id:
            raise ValueError("No models available for speculative routing")

        if hasattr(self.feedback, '_signal_extractor') and self.feedback._signal_extractor is not None:
            extractor = self.feedback._signal_extractor
        else:
            from routesmith.feedback.signals import SignalExtractor
            extractor = SignalExtractor()

        response, pt, ct, lat, err = self._execute_model(
            cheap_id, messages, request_id, **kwargs
        )

        if err:
            if expensive_id and expensive_id != cheap_id:
                response, pt, ct, lat, err = self._execute_model(
                    expensive_id, messages, request_id, **kwargs
                )
                if err:
                    self.feedback.record_outcome(
                        request_id=request_id, success=False, feedback=err,
                    )
                    raise RuntimeError(f"Speculative: cheap ({cheap_id}) and expensive ({expensive_id}) both failed")
                escalated = True
            else:
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=err,
                )
                raise RuntimeError(f"Speculative: cheap model {cheap_id} failed and no escalation target")
        else:
            signals = extractor.extract(response, cheap_id, lat)
            hard_negatives = {"error_detected", "refusal_detected", "empty_response"}
            reject = any(
                s.signal_value < 0.5
                for s in signals
                if s.signal_name in hard_negatives
            )
            if reject and expensive_id and expensive_id != cheap_id:
                response, pt, ct, lat, err = self._execute_model(
                    expensive_id, messages, request_id, **kwargs
                )
                if err:
                    response, pt, ct, lat, _ = self._execute_model(
                        cheap_id, messages, request_id, **kwargs
                    )
                escalated = True
            else:
                escalated = False

        self._track_cost(cheap_id if not escalated else (expensive_id or cheap_id), pt, ct, context, request_id)
        response._routesmith_request_id = request_id
        routing_str = f"speculative escalated {cheap_id} → {expensive_id}" if escalated else f"speculative accepted {cheap_id}"
        routing_reason = routing_str
        metadata = RoutingMetadata(
            request_id=request_id,
            model_selected=cheap_id if not escalated else (expensive_id or cheap_id),
            routing_strategy="speculative",
            routing_reason=routing_reason,
            routing_latency_ms=0.0,
            estimated_cost_usd=0.0,
            counterfactual_cost_usd=0.0,
            cost_savings_usd=0.0,
            models_considered=[cheap_id, expensive_id] if expensive_id else [cheap_id],
        )
        self._last_routing_metadata = metadata
        self._record_audit(metadata, context)
        if kwargs.get("include_metadata", False):
            response.routesmith_metadata = metadata.to_dict()
        response.routesmith_explanation = routing_str
        self.feedback.record(
            request_id=request_id, messages=messages,
            model=cheap_id if not escalated else (expensive_id or cheap_id),
            response=response, latency_ms=lat,
            agent_id=context.agent_id if context else None,
            agent_role=context.agent_role if context else None,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )
        return response

    async def _acompletion_speculative(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Async speculative execution."""
        cheap_id, expensive_id = self.router.get_speculative_plan(
            messages=messages, min_quality=min_quality,
            required_capabilities=required_capabilities,
            required_compliance=required_compliance, context=context,
        )
        if not cheap_id:
            raise ValueError("No models available for speculative routing")

        if hasattr(self.feedback, '_signal_extractor') and self.feedback._signal_extractor is not None:
            extractor = self.feedback._signal_extractor
        else:
            from routesmith.feedback.signals import SignalExtractor
            extractor = SignalExtractor()

        response, pt, ct, lat, err = await self._aexecute_model(
            cheap_id, messages, request_id, **kwargs
        )

        if err:
            if expensive_id and expensive_id != cheap_id:
                response, pt, ct, lat, err = await self._aexecute_model(
                    expensive_id, messages, request_id, **kwargs
                )
                if err:
                    self.feedback.record_outcome(
                        request_id=request_id, success=False, feedback=err,
                    )
                    raise RuntimeError("Speculative: cheap and expensive both failed")
                escalated = True
            else:
                self.feedback.record_outcome(
                    request_id=request_id, success=False, feedback=err,
                )
                raise RuntimeError(f"Speculative: cheap model {cheap_id} failed")
        else:
            signals = extractor.extract(response, cheap_id, lat)
            hard_negatives = {"error_detected", "refusal_detected", "empty_response"}
            reject = any(
                s.signal_value < 0.5
                for s in signals
                if s.signal_name in hard_negatives
            )
            if reject and expensive_id and expensive_id != cheap_id:
                response, pt, ct, lat, err = await self._aexecute_model(
                    expensive_id, messages, request_id, **kwargs
                )
                if err:
                    response, pt, ct, lat, _ = await self._aexecute_model(
                        cheap_id, messages, request_id, **kwargs
                    )
                escalated = True
            else:
                escalated = False

        self._track_cost(
            cheap_id if not escalated else (expensive_id or cheap_id),
            pt, ct, context, request_id,
        )
        response._routesmith_request_id = request_id
        routing_str = f"speculative escalated {cheap_id} → {expensive_id}" if escalated else f"speculative accepted {cheap_id}"
        metadata = RoutingMetadata(
            request_id=request_id,
            model_selected=cheap_id if not escalated else (expensive_id or cheap_id),
            routing_strategy="speculative",
            routing_reason=routing_str,
            routing_latency_ms=0.0,
            estimated_cost_usd=0.0,
            counterfactual_cost_usd=0.0,
            cost_savings_usd=0.0,
            models_considered=[cheap_id, expensive_id] if expensive_id else [cheap_id],
        )
        self._last_routing_metadata = metadata
        self._record_audit(metadata, context)
        if kwargs.get("include_metadata", False):
            response.routesmith_metadata = metadata.to_dict()
        response.routesmith_explanation = routing_str
        self.feedback.record(
            request_id=request_id, messages=messages,
            model=cheap_id if not escalated else (expensive_id or cheap_id),
            response=response, latency_ms=lat,
            agent_id=context.agent_id if context else None,
            agent_role=context.agent_role if context else None,
            conversation_id=context.conversation_id if context else None,
            turn_index=context.turn_index if context else None,
        )
        return response

    def _execute_single(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Fallback: route via direct strategy and execute a single model."""
        selected = self.router.route(messages, strategy=RoutingStrategy.DIRECT,
                                     min_quality=min_quality,
                                     required_capabilities=required_capabilities,
                                     required_compliance=required_compliance,
                                     context=context)
        response, pt, ct, lat, err = self._execute_model(
            selected, messages, request_id, **kwargs
        )
        if err:
            raise RuntimeError(f"Single model {selected} failed: {err}")
        self._track_cost(selected, pt, ct, context, request_id)
        return response

    async def _aexecute_single(
        self,
        messages: list[dict[str, str]],
        min_quality: float,
        required_capabilities: set[str] | None,
        required_compliance: set[str] | None,
        context: RouteContext,
        request_id: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Async fallback: route via direct strategy and execute a single model."""
        selected = self.router.route(messages, strategy=RoutingStrategy.DIRECT,
                                     min_quality=min_quality,
                                     required_capabilities=required_capabilities,
                                     required_compliance=required_compliance,
                                     context=context)
        response, pt, ct, lat, err = await self._aexecute_model(
            selected, messages, request_id, **kwargs
        )
        if err:
            raise RuntimeError(f"Single model {selected} failed: {err}")
        self._track_cost(selected, pt, ct, context, request_id)
        return response

    def _track_cost(
        self, model_id: str, prompt_tokens: int, completion_tokens: int,
        context: RouteContext, request_id: str,
    ) -> None:
        """Update running cost and per-cost-model counts."""
        cfg = self.registry.get(model_id)
        if not cfg:
            return
        cost = (
            (prompt_tokens / 1000) * cfg.cost_per_1k_input
            + (completion_tokens / 1000) * cfg.cost_per_1k_output
        )
        self._total_cost += cost
        self._budget.record(cost)
        cm = cfg.cost_model.value
        if cm not in self._cost_model_counts:
            self._cost_model_counts[cm] = {"request_count": 0.0, "total_cost": 0.0}
        self._cost_model_counts[cm]["request_count"] += 1
        self._cost_model_counts[cm]["total_cost"] += cost

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
            "cache_hits": self._cache_hits,
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
            "verification": self._verification.stats(),
        }

        if self._last_routing_metadata:
            result["last_routing"] = self._last_routing_metadata.to_dict()

        return result

    def _record_audit(self, metadata: RoutingMetadata, context: RouteContext | None = None) -> None:
        """Record an audit log entry for a routing decision."""
        ctx = asdict(context) if context else {}
        try:
            self._audit_storage.record(
                request_id=metadata.request_id,
                project_id=self.project,
                model_selected=metadata.model_selected,
                routing_strategy=metadata.routing_strategy,
                routing_reason=metadata.routing_reason,
                routing_latency_ms=metadata.routing_latency_ms,
                estimated_cost_usd=metadata.estimated_cost_usd,
                counterfactual_cost_usd=metadata.counterfactual_cost_usd,
                cost_savings_usd=metadata.cost_savings_usd,
                models_considered=metadata.models_considered,
                cache_hit=metadata.cache_hit,
                fallback_from=metadata.fallback_from,
                context_metadata=ctx.get("metadata"),
                agent_role=ctx.get("agent_role"),
                conversation_id=ctx.get("conversation_id"),
            )
        except Exception:
            pass

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

    def answer_poll(self, poll_id: str, option: int) -> bool:
        """Answer a quality poll with the selected option.

        Maps the option to a quality signal and feeds it to the
        bandit predictor for per-agent quality fine-tuning.

        Args:
            poll_id: The poll ID (matches the request_id).
            option: Selected option number (1-5).

        Returns:
            True if the poll was found and processed, False otherwise.
        """
        from routesmith.feedback.polls import PollSignalMapper

        poll = self._polls.get(poll_id)
        if poll is None:
            return False

        signal = PollSignalMapper.map(option)
        if signal is None:
            return False

        # Feed back to predictor via record_outcome
        try:
            self.record_outcome(
                request_id=poll_id,
                score=signal["quality"],
                feedback=signal["reason"],
            )
        finally:
            # Clean up poll storage
            self._polls.pop(poll_id, None)

        return True

    def recommendations(self) -> dict[str, Any]:
        """Return proactive intelligence: recommendations, warnings, forecast.

        Aggregates per-agent model recommendations, anomaly warnings,
        new models to explore, and budget pacing/forecast.
        """
        recommendations: dict[str, Any] = {
            "warnings": [],
            "new_models_to_try": [],
            "forecast": {
                "monthly_cost_current": round(self._total_cost, 2),
                "request_count": self._request_count,
                "savings_total": round(self._counterfactual_cost - self._total_cost, 2),
            },
        }

        # Per-agent recommendations from feedback storage
        if self.feedback._storage is not None:
            roles = self.feedback._storage.get_known_roles()
            for role in roles:
                result = self.recommend_model_for_agent(role, min_samples=10)
                if result:
                    recommendations[role] = {
                        "current_best": result["model"],
                        "avg_quality": result["avg_quality"],
                        "avg_cost_usd": result["avg_cost_usd"],
                        "confidence": result["confidence"],
                        "sample_count": result["sample_count"],
                    }
                    # Suggest new models to try
                    for m in result.get("new_models_to_explore", []):
                        if m not in recommendations["new_models_to_try"]:
                            recommendations["new_models_to_try"].append(m)

        return recommendations

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
                        _reward_val = float(effective_reward_fn(ctx))  # noqa: F841 — computed for logging; no longer passed as kwarg
                    except Exception as e:
                        logger.warning(
                            "reward_fn raised an error, skipping reward override: %s", e
                        )
                self.router.predictor.update(
                    messages=record.messages,
                    model_id=record.model_id,
                    actual_quality=quality,
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
