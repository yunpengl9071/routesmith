"""Tests for RouteSmith.with_free_models() factory preset."""


from routesmith import RouteSmith


class TestFreeModelsPreset:
    """RouteSmith.with_free_models() provides zero-config access to free-tier models."""

    def test_with_free_models_returns_client(self):
        """Factory returns a configured RouteSmith instance."""
        rs = RouteSmith.with_free_models()
        assert isinstance(rs, RouteSmith)

    def test_registers_free_models(self):
        """Factory registers at least five free models."""
        rs = RouteSmith.with_free_models()
        models = rs.registry.list_models()
        assert len(models) >= 5, f"Expected at least 5 free models, got {len(models)}"

    def test_free_models_have_zero_cost(self):
        """All registered models should report zero cost per token."""
        rs = RouteSmith.with_free_models()
        for model in rs.registry.list_models():
            assert model.cost_per_1k_input == 0.0, (
                f"{model.model_id} should have zero cost"
            )
            assert model.cost_per_1k_output == 0.0, (
                f"{model.model_id} should have zero cost"
            )

    def test_free_models_include_known_names(self):
        """Factory should include recognizable free-model identifiers."""
        rs = RouteSmith.with_free_models()
        model_ids = [m.model_id for m in rs.registry.list_models()]
        free_indicators = ["gemini", "llama", "qwen", "gemma", "ministral", "nemotron"]
        matched = [
            m
            for m in model_ids
            if any(indicator in m.lower() for indicator in free_indicators)
        ]
        assert len(matched) >= 3, (
            f"Expected at least 3 recognizable free models in {model_ids}"
        )

    def test_client_config_is_valid(self):
        """Resulting client should have a valid config."""
        rs = RouteSmith.with_free_models()
        assert rs.config is not None
