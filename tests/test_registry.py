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
