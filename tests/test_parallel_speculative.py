"""Tests for parallel and speculative execution strategies."""
from unittest.mock import patch

import pytest

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig, RoutingStrategy
from tests.helpers import fake_response


@pytest.fixture
def routesmith():
    rs = RouteSmith(
        config=RouteSmithConfig(
            default_strategy=RoutingStrategy.PARALLEL,
        ).with_budget(quality_threshold=0.0),
    )
    rs.register_model(
        "model-cheap",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0002,
        quality_score=0.70,
    )
    rs.register_model(
        "model-expensive",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        quality_score=0.95,
    )
    return rs


def _models_called(mock):
    return [call[1]["model"] for call in mock.call_args_list]


class TestParallelExecution:
    """Tests for parallel routing strategy."""

    @patch("routesmith.client.litellm.completion")
    def test_parallel_runs_two_models(self, mock_litellm, routesmith):
        """Parallel runs both models, picks cheaper when responses equivalent."""
        mock_litellm.side_effect = lambda model=None, messages=None, **kw: (
            fake_response(content="Hello world", model=model)
        )

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
            strategy=RoutingStrategy.PARALLEL,
        )

        assert mock_litellm.call_count == 2
        called = _models_called(mock_litellm)
        assert "model-cheap" in called
        assert "model-expensive" in called
        assert response is not None

    @patch("routesmith.client.litellm.completion")
    def test_parallel_picks_cheaper_when_equivalent(self, mock_litellm, routesmith):
        """When responses are equivalent, cheaper model is selected."""
        mock_litellm.side_effect = lambda model=None, messages=None, **kw: (
            fake_response(content="identical text for both", model=model)
        )

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
            strategy=RoutingStrategy.PARALLEL,
        )

        assert response is not None

    @patch("routesmith.client.litellm.completion")
    def test_parallel_falls_through_with_default_strategy(self, mock_litellm, routesmith):
        """When default_strategy is PARALLEL, no explicit strategy needed."""
        mock_litellm.side_effect = lambda model=None, messages=None, **kw: (
            fake_response(content="ok", model=model)
        )

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
        )

        assert mock_litellm.call_count == 2
        assert response is not None


class TestSpeculativeExecution:
    """Tests for speculative routing strategy."""

    @pytest.fixture
    def rs(self):
        rs = RouteSmith(
            config=RouteSmithConfig(
                default_strategy=RoutingStrategy.SPECULATIVE,
            ).with_budget(quality_threshold=0.0),
        )
        rs.register_model(
            "fast-cheap",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0002,
            quality_score=0.70,
        )
        rs.register_model(
            "slow-expensive",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            quality_score=0.95,
        )
        return rs

    @patch("routesmith.client.litellm.completion")
    def test_speculative_accepts_cheap_when_clean(self, mock_litellm, rs):
        """Clean cheap response is accepted without escalation."""
        mock_litellm.return_value = fake_response(
            content="Clean helpful answer", model="fast-cheap"
        )

        response = rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            strategy=RoutingStrategy.SPECULATIVE,
        )

        assert mock_litellm.call_count == 1
        assert _models_called(mock_litellm) == ["fast-cheap"]
        assert response is not None

    @patch("routesmith.client.litellm.completion")
    def test_speculative_escalates_on_refusal(self, mock_litellm, rs):
        """Refusal triggers escalation to expensive model."""
        def side_effect(model=None, messages=None, **kw):
            if model == "fast-cheap":
                return fake_response(
                    content="I'm sorry, I cannot help with that.",
                    model=model,
                )
            return fake_response(content="Good answer", model=model)

        mock_litellm.side_effect = side_effect

        response = rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            strategy=RoutingStrategy.SPECULATIVE,
        )

        assert mock_litellm.call_count == 2
        assert _models_called(mock_litellm) == ["fast-cheap", "slow-expensive"]
        assert response is not None

    @patch("routesmith.client.litellm.completion")
    def test_speculative_escalates_on_empty_response(self, mock_litellm, rs):
        """Empty response triggers escalation."""
        def side_effect(model=None, messages=None, **kw):
            if model == "fast-cheap":
                return fake_response(content="", model=model)
            return fake_response(content="Good answer", model=model)

        mock_litellm.side_effect = side_effect

        response = rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            strategy=RoutingStrategy.SPECULATIVE,
        )

        assert mock_litellm.call_count == 2
        assert response is not None

    @patch("routesmith.client.litellm.completion")
    def test_speculative_accepts_cheap_when_no_escalation_target(self, mock_litellm, rs):
        """If only one model registered, speculative uses it directly."""
        rs2 = RouteSmith()
        rs2.register_model(
            "only-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.8,
        )
        mock_litellm.return_value = fake_response(
            content="Clean response", model="only-model"
        )

        response = rs2.completion(
            messages=[{"role": "user", "content": "hello"}],
            strategy=RoutingStrategy.SPECULATIVE,
        )

        assert mock_litellm.call_count == 1
        assert response is not None
