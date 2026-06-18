"""Tests for model discovery auto-registration."""

from unittest.mock import patch

import pytest

from routesmith.registry.discovery import discover_models


class TestDiscoverModels:
    def test_discovers_from_openrouter(self):
        """discover_models returns models from OpenRouter API."""
        mock_response = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "pricing": {"prompt": "0.000005", "completion": "0.000015"},
                    "context_length": 128000,
                    "architecture": {"modality": "text+image"},
                }
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            models = discover_models(api_key="sk-test")

        assert len(models) > 0
        assert models[0]["model_id"] == "openai/gpt-4o"
        assert models[0]["cost_per_1k_input"] == pytest.approx(0.005)
        assert models[0]["cost_per_1k_output"] == pytest.approx(0.015)
        assert models[0]["context_window"] == 128000

    def test_falls_back_to_curated_when_no_key(self):
        """discover_models falls back to curated list without API key."""
        models = discover_models(api_key=None)
        assert len(models) > 0
        assert any(m["model_id"] == "openai/gpt-4o" for m in models)

    def test_respects_provider_filter(self):
        """discover_models filters by provider whitelist."""
        models = discover_models(
            api_key=None, providers=["anthropic"]
        )
        for m in models:
            assert m["model_id"].startswith("anthropic/")

    def test_seeds_quality_from_benchmarks(self):
        """discover_models includes benchmark quality seeds when available."""
        mock_response = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "pricing": {"prompt": "0.000005", "completion": "0.000015"},
                    "context_length": 128000,
                    "benchmarks": {"mmlu": 0.887, "humaneval": 0.902},
                }
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            models = discover_models(api_key="sk-test")

        assert 0.7 <= models[0]["quality_score"] <= 1.0