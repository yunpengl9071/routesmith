"""Comprehensive tests for the routing engine."""

import pytest
from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.registry.models import ModelRegistry
from routesmith.strategy.router import Router


class TestRouterBasics:
    """Basic router functionality tests."""

    @pytest.fixture
    def registry(self):
        """Create a registry with test models."""
        reg = ModelRegistry()
        # Expensive high-quality model
        reg.register(
            "gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            quality_score=0.95,
        )
        # Cheap medium-quality model
        reg.register(
            "gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
        )
        # Mid-range model
        reg.register(
            "claude-3-haiku",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            quality_score=0.80,
        )
        return reg

    @pytest.fixture
    def config(self):
        return RouteSmithConfig()

    @pytest.fixture
    def router(self, config, registry):
        return Router(config, registry)

    def test_router_initialization(self, router, registry):
        """Test router initializes correctly."""
        assert router.registry is registry
        assert router.predictor is not None

    def test_route_raises_without_models(self, config):
        """Test routing raises error with empty registry."""
        empty_registry = ModelRegistry()
        router = Router(config, empty_registry)

        with pytest.raises(ValueError, match="No models registered"):
            router.route([{"role": "user", "content": "Hello"}])

    def test_route_unknown_strategy(self, router):
        """Test routing raises error for unknown strategy."""
        # Create a mock strategy that's not in the enum
        class FakeStrategy:
            pass

        with pytest.raises(ValueError, match="Unknown routing strategy"):
            router.route(
                [{"role": "user", "content": "Hello"}],
                strategy=FakeStrategy(),  # type: ignore
            )


class TestDirectRouting:
    """Tests for direct (single-model) routing strategy."""

    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.register("expensive", cost_per_1k_input=0.01, cost_per_1k_output=0.03, quality_score=0.95)
        reg.register("cheap", cost_per_1k_input=0.0001, cost_per_1k_output=0.0003, quality_score=0.70)
        reg.register("medium", cost_per_1k_input=0.001, cost_per_1k_output=0.003, quality_score=0.85)
        return reg

    @pytest.fixture
    def router(self, registry):
        return Router(RouteSmithConfig(), registry)

    def test_direct_routing_selects_model(self, router):
        """Test direct routing returns a valid model."""
        model = router.route(
            [{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.DIRECT,
        )
        assert model in ["expensive", "cheap", "medium"]

    def test_direct_routing_respects_quality_threshold(self, router):
        """Test direct routing respects minimum quality."""
        model = router.route(
            [{"role": "user", "content": "Complex task"}],
            strategy=RoutingStrategy.DIRECT,
            min_quality=0.90,
        )
        # Only expensive model meets 0.90 threshold
        assert model == "expensive"

    def test_direct_routing_respects_cost_constraint(self, router):
        """Test direct routing respects max cost."""
        model = router.route(
            [{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.DIRECT,
            max_cost=0.001,  # Only cheap and medium qualify
        )
        # Should not select expensive model
        assert model in ["cheap", "medium"]

    def test_direct_routing_with_both_constraints(self, router):
        """Test direct routing with both quality and cost constraints."""
        model = router.route(
            [{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.DIRECT,
            max_cost=0.01,  # Excludes expensive
            min_quality=0.80,  # Excludes cheap
        )
        # Only medium fits both
        assert model == "medium"

    def test_direct_routing_fallback_to_cheapest(self, router):
        """Test direct routing falls back to cheapest when no model meets constraints."""
        # With impossible constraints, should fallback
        model = router.route(
            [{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.DIRECT,
            max_cost=0.00001,  # No model is this cheap
        )
        # Should fallback to cheapest
        assert model == "cheap"


class TestCascadeRouting:
    """Tests for cascade routing strategy."""

    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.register("tier1", cost_per_1k_input=0.0001, cost_per_1k_output=0.0003, quality_score=0.70)
        reg.register("tier2", cost_per_1k_input=0.001, cost_per_1k_output=0.003, quality_score=0.85)
        reg.register("tier3", cost_per_1k_input=0.01, cost_per_1k_output=0.03, quality_score=0.95)
        return reg

    @pytest.fixture
    def router(self, registry):
        return Router(RouteSmithConfig(), registry)

    def test_cascade_starts_with_cheapest(self, router):
        """Test cascade routing starts with cheapest model."""
        model = router.route(
            [{"role": "user", "content": "Simple question"}],
            strategy=RoutingStrategy.CASCADE,
        )
        # Cascade should start with cheapest
        assert model == "tier1"

    def test_cascade_respects_quality_floor(self, router):
        """Test cascade respects minimum quality threshold."""
        model = router.route(
            [{"role": "user", "content": "Quality needed"}],
            strategy=RoutingStrategy.CASCADE,
            min_quality=0.80,
        )
        # Should start with cheapest that meets quality
        assert model == "tier2"

    def test_get_cascade_models_returns_ordered_list(self, router):
        """Test get_cascade_models returns models ordered by cost."""
        models = router.get_cascade_models()
        assert models == ["tier1", "tier2", "tier3"]

    def test_get_cascade_models_respects_quality(self, router):
        """Test get_cascade_models filters by quality."""
        models = router.get_cascade_models(min_quality=0.80)
        assert models == ["tier2", "tier3"]

    def test_get_cascade_models_respects_max_tiers(self, router):
        """Test get_cascade_models limits tiers."""
        models = router.get_cascade_models(max_tiers=2)
        assert len(models) == 2
        assert models == ["tier1", "tier2"]


class TestParallelRouting:
    """Tests for parallel routing strategy."""

    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.register("model_a", cost_per_1k_input=0.001, cost_per_1k_output=0.003, quality_score=0.90)
        reg.register("model_b", cost_per_1k_input=0.002, cost_per_1k_output=0.004, quality_score=0.92)
        return reg

    @pytest.fixture
    def router(self, registry):
        return Router(RouteSmithConfig(), registry)

    def test_parallel_returns_highest_quality(self, router):
        """Test parallel routing returns highest quality model as primary."""
        model = router.route(
            [{"role": "user", "content": "High stakes query"}],
            strategy=RoutingStrategy.PARALLEL,
        )
        # Should return highest quality model
        assert model == "model_b"

    def test_parallel_respects_cost_constraint(self, router):
        """Test parallel routing respects cost constraint."""
        model = router.route(
            [{"role": "user", "content": "Query"}],
            strategy=RoutingStrategy.PARALLEL,
            max_cost=0.002,  # Only model_a qualifies
        )
        assert model == "model_a"


class TestSpeculativeRouting:
    """Tests for speculative routing strategy."""

    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.register("fast_cheap", cost_per_1k_input=0.0001, cost_per_1k_output=0.0003, quality_score=0.75)
        reg.register("slow_quality", cost_per_1k_input=0.01, cost_per_1k_output=0.03, quality_score=0.95)
        return reg

    @pytest.fixture
    def router(self, registry):
        return Router(RouteSmithConfig(), registry)

    def test_speculative_starts_with_cheapest(self, router):
        """Test speculative routing starts with cheap model."""
        model = router.route(
            [{"role": "user", "content": "Speculative query"}],
            strategy=RoutingStrategy.SPECULATIVE,
        )
        # Should start with cheap model like cascade
        assert model == "fast_cheap"


class TestRouterWithDifferentConfigs:
    """Test router behavior with different configurations."""

    def test_router_with_empty_quality_priors(self):
        """Test router works even without quality priors."""
        registry = ModelRegistry()
        registry.register("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.8)

        config = RouteSmithConfig()
        router = Router(config, registry)

        model = router.route([{"role": "user", "content": "Test"}])
        assert model == "model"

    def test_router_with_single_model(self):
        """Test router with only one model."""
        registry = ModelRegistry()
        registry.register("only_model", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.9)

        router = Router(RouteSmithConfig(), registry)

        # Should always return the only model regardless of strategy
        for strategy in [RoutingStrategy.DIRECT, RoutingStrategy.CASCADE,
                         RoutingStrategy.PARALLEL, RoutingStrategy.SPECULATIVE]:
            model = router.route([{"role": "user", "content": "Test"}], strategy=strategy)
            assert model == "only_model"

    def test_router_with_equal_quality_models(self):
        """Test router picks cheaper model when quality is equal."""
        registry = ModelRegistry()
        registry.register("expensive", cost_per_1k_input=0.01, cost_per_1k_output=0.02, quality_score=0.85)
        registry.register("cheap", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.85)

        router = Router(RouteSmithConfig(), registry)

        # With equal quality, should prefer cheaper
        model = router.route(
            [{"role": "user", "content": "Test"}],
            strategy=RoutingStrategy.CASCADE,
        )
        assert model == "cheap"


class TestCapabilityRouting:
    """Tests for capability-aware routing."""

    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.register(
            "full-model", cost_per_1k_input=0.01, cost_per_1k_output=0.03,
            quality_score=0.95,
            supports_function_calling=True, supports_vision=True,
        )
        reg.register(
            "tools-only", cost_per_1k_input=0.001, cost_per_1k_output=0.003,
            quality_score=0.85,
            supports_function_calling=True, supports_vision=False,
        )
        reg.register(
            "basic-model", cost_per_1k_input=0.0001, cost_per_1k_output=0.0003,
            quality_score=0.70,
            supports_function_calling=False, supports_vision=False,
            supports_json_mode=False,
        )
        return reg

    @pytest.fixture
    def router(self, registry):
        return Router(RouteSmithConfig(), registry)

    def test_direct_route_filters_by_capability(self, router):
        """Models without tool_calling are excluded when tool_calling is required."""
        model = router.route(
            [{"role": "user", "content": "Use a tool"}],
            strategy=RoutingStrategy.DIRECT,
            required_capabilities={"tool_calling"},
        )
        assert model in ["full-model", "tools-only"]
        assert model != "basic-model"

    def test_route_requires_vision_excludes_non_vision(self, router):
        """Only vision-capable models are returned when vision is required."""
        model = router.route(
            [{"role": "user", "content": "Describe this image"}],
            strategy=RoutingStrategy.DIRECT,
            required_capabilities={"vision"},
        )
        assert model == "full-model"

    def test_route_multiple_capabilities(self, router):
        """Requiring both tool_calling and vision leaves only full-model."""
        model = router.route(
            [{"role": "user", "content": "Analyze image with tools"}],
            strategy=RoutingStrategy.DIRECT,
            required_capabilities={"tool_calling", "vision"},
        )
        assert model == "full-model"

    def test_no_capable_model_raises_error(self, router):
        """Raises ValueError when no model supports all required capabilities."""
        with pytest.raises(ValueError, match="No models support required capabilities"):
            router.route(
                [{"role": "user", "content": "Test"}],
                required_capabilities={"nonexistent_capability"},
            )

    def test_cascade_respects_capabilities(self, router):
        """Cascade routing filters by capabilities."""
        model = router.route(
            [{"role": "user", "content": "Use tools"}],
            strategy=RoutingStrategy.CASCADE,
            required_capabilities={"tool_calling"},
        )
        # Should pick cheapest with tool_calling = tools-only
        assert model == "tools-only"

    def test_get_cascade_models_respects_capabilities(self, router):
        """get_cascade_models filters by capabilities."""
        models = router.get_cascade_models(required_capabilities={"tool_calling"})
        assert "basic-model" not in models
        assert "tools-only" in models
        assert "full-model" in models

    def test_no_capabilities_returns_any_model(self, router):
        """Without required_capabilities, all models are candidates."""
        model = router.route(
            [{"role": "user", "content": "Hello"}],
            strategy=RoutingStrategy.DIRECT,
        )
        # Should pick cheapest (basic-model) since no quality threshold
        assert model in ["full-model", "tools-only", "basic-model"]

    def test_capabilities_applied_before_quality_prediction(self, router):
        """Capabilities filter happens before quality prediction."""
        # With high quality threshold + tool_calling requirement,
        # basic-model is excluded by capability, not quality
        model = router.route(
            [{"role": "user", "content": "Complex tool task"}],
            strategy=RoutingStrategy.DIRECT,
            min_quality=0.90,
            required_capabilities={"tool_calling"},
        )
        assert model == "full-model"


class TestRouterEdgeCases:
    """Edge case tests for router."""

    def test_route_with_empty_messages(self):
        """Test routing handles empty messages list."""
        registry = ModelRegistry()
        registry.register("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002)

        router = Router(RouteSmithConfig(), registry)
        # Should not raise, just route based on priors
        model = router.route([])
        assert model == "model"

    def test_route_with_long_conversation(self):
        """Test routing handles long message history."""
        registry = ModelRegistry()
        registry.register("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002)

        router = Router(RouteSmithConfig(), registry)

        # Create a long conversation
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(100)
        ]

        model = router.route(messages)
        assert model == "model"

    def test_route_with_special_characters_in_content(self):
        """Test routing handles special characters in messages."""
        registry = ModelRegistry()
        registry.register("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002)

        router = Router(RouteSmithConfig(), registry)

        messages = [{"role": "user", "content": "Hello! 🎉 Special chars: <>&\"'"}]
        model = router.route(messages)
        assert model == "model"
