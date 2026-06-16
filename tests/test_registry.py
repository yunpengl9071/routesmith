"""Tests for model registry."""

import pytest

from routesmith.registry.models import ModelConfig, ModelRegistry


class TestModelConfig:
    def test_cost_per_1k_total(self):
        config = ModelConfig(
            model_id="test-model",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
        )
        assert config.cost_per_1k_total == 0.02

    def test_capabilities_auto_populated_from_flags(self):
        config = ModelConfig(
            model_id="test",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_function_calling=True,
            supports_vision=True,
            supports_json_mode=True,
            supports_streaming=True,
        )
        assert "tool_calling" in config.capabilities
        assert "vision" in config.capabilities
        assert "json_mode" in config.capabilities
        assert "streaming" in config.capabilities

    def test_capabilities_not_added_when_flags_false(self):
        config = ModelConfig(
            model_id="test",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_function_calling=False,
            supports_vision=False,
            supports_json_mode=False,
            supports_streaming=False,
        )
        assert config.capabilities == set()

    def test_explicit_capabilities_merged_with_flags(self):
        config = ModelConfig(
            model_id="test",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_function_calling=True,
            capabilities={"custom_cap"},
        )
        assert "tool_calling" in config.capabilities
        assert "custom_cap" in config.capabilities


class TestModelRegistry:
    @pytest.fixture
    def registry(self):
        reg = ModelRegistry()
        reg.register(
            "gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            quality_score=0.95,
        )
        reg.register(
            "gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
        )
        reg.register(
            "claude-3-haiku",
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            quality_score=0.80,
        )
        return reg

    def test_register_and_get(self, registry):
        model = registry.get("gpt-4o")
        assert model is not None
        assert model.model_id == "gpt-4o"
        assert model.quality_score == 0.95

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_filter_by_quality(self, registry):
        high_quality = registry.filter_by_quality(0.9)
        assert len(high_quality) == 1
        assert high_quality[0].model_id == "gpt-4o"

    def test_get_cheapest(self, registry):
        cheapest = registry.get_cheapest()
        assert cheapest is not None
        assert cheapest.model_id == "gpt-4o-mini"

    def test_get_cheapest_with_quality_threshold(self, registry):
        cheapest = registry.get_cheapest(min_quality=0.9)
        assert cheapest is not None
        assert cheapest.model_id == "gpt-4o"

    def test_get_best_quality(self, registry):
        best = registry.get_best_quality()
        assert best is not None
        assert best.model_id == "gpt-4o"

    def test_sorted_by_cost(self, registry):
        sorted_models = registry.sorted_by_cost()
        assert sorted_models[0].model_id == "gpt-4o-mini"
        assert sorted_models[-1].model_id == "gpt-4o"

    def test_len(self, registry):
        assert len(registry) == 3

    def test_contains(self, registry):
        assert "gpt-4o" in registry
        assert "nonexistent" not in registry

    def test_register_with_capabilities(self):
        reg = ModelRegistry()
        reg.register(
            "capable-model",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            supports_function_calling=True,
            supports_vision=True,
        )
        model = reg.get("capable-model")
        assert model is not None
        assert "tool_calling" in model.capabilities
        assert "vision" in model.capabilities

    def test_register_without_function_calling(self):
        reg = ModelRegistry()
        reg.register(
            "limited-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            supports_function_calling=False,
            supports_json_mode=False,
        )
        model = reg.get("limited-model")
        assert model is not None
        assert "tool_calling" not in model.capabilities
        assert "json_mode" not in model.capabilities
        assert "streaming" in model.capabilities  # default True

    def test_filter_by_capabilities(self):
        reg = ModelRegistry()
        reg.register("full", cost_per_1k_input=0.01, cost_per_1k_output=0.03,
                      supports_function_calling=True, supports_vision=True)
        reg.register("no-vision", cost_per_1k_input=0.001, cost_per_1k_output=0.002,
                      supports_function_calling=True, supports_vision=False)
        reg.register("limited", cost_per_1k_input=0.0001, cost_per_1k_output=0.0002,
                      supports_function_calling=False, supports_vision=False)

        # Only "full" has both tool_calling and vision
        result = reg.filter_by_capabilities({"tool_calling", "vision"})
        assert len(result) == 1
        assert result[0].model_id == "full"

        # Two models have tool_calling
        result = reg.filter_by_capabilities({"tool_calling"})
        assert len(result) == 2

        # Empty set returns all
        result = reg.filter_by_capabilities(set())
        assert len(result) == 3

    def test_get_by_capability_uses_capabilities_set(self):
        reg = ModelRegistry()
        reg.register("with-tools", cost_per_1k_input=0.01, cost_per_1k_output=0.03,
                      supports_function_calling=True)
        reg.register("no-tools", cost_per_1k_input=0.001, cost_per_1k_output=0.002,
                      supports_function_calling=False)

        result = reg.get_by_capability("tool_calling")
        assert len(result) == 1
        assert result[0].model_id == "with-tools"


class TestCapacityTrackerIntegration:
    def test_registry_creates_tracker_for_provisioned(self):
        from routesmith.config import CostModel
        from routesmith.registry.models import ModelRegistry

        reg = ModelRegistry()
        reg.register(
            "provisioned-model",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=10,
        )
        tracker = reg.get_capacity_tracker("provisioned-model")
        assert tracker is not None
        assert tracker.max_rpm == 10

    def test_registry_returns_none_for_on_demand(self):
        from routesmith.registry.models import ModelRegistry

        reg = ModelRegistry()
        reg.register("on-demand-model", 0.001, 0.002)
        tracker = reg.get_capacity_tracker("on-demand-model")
        assert tracker is None

    def test_registry_returns_none_for_unknown_model(self):
        from routesmith.registry.models import ModelRegistry

        reg = ModelRegistry()
        assert reg.get_capacity_tracker("nonexistent") is None

    def test_capacity_tracker_reused_same_model(self):
        from routesmith.config import CostModel
        from routesmith.registry.models import ModelRegistry

        reg = ModelRegistry()
        reg.register(
            "provisioned-model",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=10,
        )
        t1 = reg.get_capacity_tracker("provisioned-model")
        t2 = reg.get_capacity_tracker("provisioned-model")
        assert t1 is t2  # Same tracker instance reused

    def test_deregister_cleans_up_tracker(self):
        from routesmith.config import CostModel
        from routesmith.registry.models import ModelRegistry

        reg = ModelRegistry()
        reg.register(
            "provisioned-model",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=10,
        )
        _ = reg.get_capacity_tracker("provisioned-model")
        reg.deregister("provisioned-model")
        # After deregister, get should return None
        assert reg.get_capacity_tracker("provisioned-model") is None


class TestComplianceFiltering:
    def test_model_has_compliance_tags_default_empty(self):
        from routesmith.registry.models import ModelConfig
        model = ModelConfig(model_id="test", cost_per_1k_input=0.001, cost_per_1k_output=0.002)
        assert model.compliance_tags == set()

    def test_model_compliance_tags_settable(self):
        from routesmith.registry.models import ModelConfig
        model = ModelConfig(
            model_id="test",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            compliance_tags={"hipaa", "soc2"},
        )
        assert model.compliance_tags == {"hipaa", "soc2"}

    def test_filter_by_compliance_exact_match(self):
        from routesmith.registry.models import ModelRegistry
        reg = ModelRegistry()
        reg.register("hipaa-model", 0.001, 0.002, compliance_tags={"hipaa", "soc2"})
        reg.register("no-compliance", 0.001, 0.002)

        results = reg.filter_by_compliance({"hipaa"})
        assert len(results) == 1
        assert results[0].model_id == "hipaa-model"

    def test_filter_by_compliance_requires_all_tags(self):
        from routesmith.registry.models import ModelRegistry
        reg = ModelRegistry()
        reg.register("hipaa-only", 0.001, 0.002, compliance_tags={"hipaa"})
        reg.register("hipaa-soc2", 0.001, 0.002, compliance_tags={"hipaa", "soc2"})

        results = reg.filter_by_compliance({"hipaa", "soc2"})
        assert len(results) == 1
        assert results[0].model_id == "hipaa-soc2"

    def test_filter_by_compliance_empty_required_returns_all(self):
        from routesmith.registry.models import ModelRegistry
        reg = ModelRegistry()
        reg.register("model-a", 0.001, 0.002, compliance_tags={"hipaa"})
        reg.register("model-b", 0.001, 0.002)

        results = reg.filter_by_compliance(set())
        assert len(results) == 2

    def test_filter_by_compliance_no_match_returns_empty(self):
        from routesmith.registry.models import ModelRegistry
        reg = ModelRegistry()
        reg.register("model-a", 0.001, 0.002, compliance_tags={"soc2"})

        results = reg.filter_by_compliance({"hipaa"})
        assert len(results) == 0

    def test_register_accepts_compliance_tags(self):
        from routesmith.registry.models import ModelRegistry
        reg = ModelRegistry()
        config = reg.register(
            "test-model",
            0.001,
            0.002,
            compliance_tags={"hipaa", "us-east-1"},
        )
        assert config.compliance_tags == {"hipaa", "us-east-1"}
