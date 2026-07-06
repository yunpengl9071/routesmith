"""Tests for benchmark-grounded warm-start priors."""
from __future__ import annotations

import pytest

from routesmith import RouteSmith
from routesmith.registry.priors import load_default_priors, lookup_prior


class TestLoadDefaultPriors:
    def test_load_default_priors_has_20_entries(self):
        priors = load_default_priors()
        assert len(priors) == 20


class TestLookupPrior:
    def test_lookup_exact(self):
        priors = load_default_priors()
        assert lookup_prior("openai/gpt-4o", priors) == 0.92

    def test_lookup_openrouter_prefixed(self):
        priors = load_default_priors()
        assert lookup_prior("openrouter/openai/gpt-4o", priors) == 0.92

    def test_lookup_tail_unique(self):
        priors = load_default_priors()
        assert lookup_prior("gpt-4o-mini", priors) == 0.82

    def test_lookup_ambiguous_returns_none(self):
        """'large' matches multiple tails (mistralai/mistral-large, etc.)."""
        priors = load_default_priors()
        assert lookup_prior("large", priors) is None

    def test_lookup_unknown_returns_none(self):
        priors = load_default_priors()
        assert lookup_prior("nonexistent-model", priors) is None


class TestRegisterModelUsesPrior:
    @pytest.fixture
    def client(self):
        return RouteSmith()

    def test_register_model_uses_prior(self, client):
        client.register_model(
            "openai/gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )
        model = client.registry.get("openai/gpt-4o")
        assert model is not None
        assert model.quality_score == 0.92

    def test_register_model_explicit_score_wins(self, client):
        client.register_model(
            "openai/gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            quality_score=0.5,
        )
        model = client.registry.get("openai/gpt-4o")
        assert model is not None
        assert model.quality_score == 0.5

    def test_register_model_unknown_falls_back_to_08(self, client):
        client.register_model(
            "some/unknown-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        model = client.registry.get("some/unknown-model")
        assert model is not None
        assert model.quality_score == 0.8
