"""Tests for cascade execution with verification (P2.4)."""

from unittest.mock import MagicMock, patch

import pytest

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig, RoutingStrategy
from tests.helpers import fake_response


@pytest.fixture
def routesmith():
    rs = RouteSmith(
        config=RouteSmithConfig(
            default_strategy=RoutingStrategy.CASCADE,
        ).with_budget(quality_threshold=0.6),
    )
    rs.register_model(
        "model-cheap",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0002,
        quality_score=0.70,
    )
    rs.register_model(
        "model-medium",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        quality_score=0.85,
    )
    rs.register_model(
        "model-expensive",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        quality_score=0.95,
    )
    return rs


def _count_calls(mock, model_id=None):
    """Count how many times mock was called with a specific model."""
    count = 0
    for call_args in mock.call_args_list:
        if call_args[1].get("model") == model_id:
            count += 1
    return count


def _models_called(mock):
    """Return list of model IDs that were called, in order."""
    return [call[1]["model"] for call in mock.call_args_list]


class TestCascadeExecution:
    """Tests for cascade routing with verification."""

    @patch("routesmith.client.litellm.completion")
    def test_cascade_accepts_first_tier_when_clean(self, mock_litellm, routesmith):
        """All responses clean -> first tier accepted, single call."""
        mock_litellm.side_effect = lambda model=None, messages=None, **kw: (
            fake_response(content="Clean response", model=model)
        )

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
        )

        assert mock_litellm.call_count == 1
        assert _models_called(mock_litellm) == ["model-cheap"]
        assert response is not None

    @patch("routesmith.client.litellm.completion")
    def test_cascade_escalates_on_refusal(self, mock_litellm, routesmith):
        """Tier1 returns refusal -> escalates to tier2 -> accepted."""
        def side_effect(model=None, messages=None, **kw):
            if model == "model-cheap":
                return fake_response(
                    content="I'm sorry, I cannot help with that.",
                    model=model,
                )
            return fake_response(content="Clean response", model=model)

        mock_litellm.side_effect = side_effect

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
            include_metadata=True,
        )

        assert mock_litellm.call_count == 2
        assert _models_called(mock_litellm) == ["model-cheap", "model-medium"]
        md = getattr(response, "routesmith_metadata", {})
        assert md.get("escalations") == 1
        assert len(md.get("tiers_tried", [])) == 2

    @patch("routesmith.client.litellm.completion")
    def test_cascade_escalates_on_judge_reject(self, mock_litellm, routesmith):
        """Judge rejects tier1 -> escalates to tier2."""
        # Set up a mock judge
        mock_judge = MagicMock()
        mock_judge.score.side_effect = [0.3, 0.9]
        routesmith._judge = mock_judge

        mock_litellm.side_effect = lambda model=None, messages=None, **kw: (
            fake_response(content="Looks fine but judge says no", model=model)
        )

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
            include_metadata=True,
        )

        assert mock_litellm.call_count == 2
        assert _models_called(mock_litellm) == ["model-cheap", "model-medium"]
        assert mock_judge.score.call_count == 2
        md = getattr(response, "routesmith_metadata", {})
        assert md.get("escalations") == 1

    @patch("routesmith.client.litellm.completion")
    def test_cascade_exhausted_returns_last(self, mock_litellm, routesmith):
        """All tiers refuse -> last response returned with exhausted flag."""
        def side_effect(model=None, messages=None, **kw):
            return fake_response(
                content="I'm sorry, I cannot help with that.",
                model=model,
            )

        mock_litellm.side_effect = side_effect

        response = routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
            include_metadata=True,
        )

        assert mock_litellm.call_count == 3
        md = getattr(response, "routesmith_metadata", {})
        assert md.get("cascade_exhausted") is True

    @patch("routesmith.client.litellm.completion")
    def test_cascade_cost_accumulates_all_attempts(self, mock_litellm, routesmith):
        """2-tier escalation -> total cost includes both attempts."""
        def side_effect(model=None, messages=None, **kw):
            if model == "model-cheap":
                return fake_response(
                    content="I'm sorry, I cannot help with that.",
                    model=model,
                    prompt_tokens=10,
                    completion_tokens=20,
                )
            return fake_response(
                content="Clean response",
                model=model,
                prompt_tokens=10,
                completion_tokens=20,
            )

        mock_litellm.side_effect = side_effect

        routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
        )

        # model-cheap cost: (10/1000)*0.0001 + (20/1000)*0.0002 = 0.000001 + 0.000004 = 0.000005
        # model-medium cost: (10/1000)*0.001 + (20/1000)*0.002 = 0.00001 + 0.00004 = 0.00005
        # Total: 0.000055
        expected_cost = 0.000055
        assert routesmith.stats["total_cost_usd"] == pytest.approx(expected_cost, abs=1e-8)

    @patch("routesmith.client.litellm.completion")
    def test_cascade_rejected_tiers_get_negative_feedback(self, mock_litellm, routesmith):
        """Predictor receives updates for rejected tiers."""
        def side_effect(model=None, messages=None, **kw):
            if model == "model-cheap":
                return fake_response(
                    content="I'm sorry, I cannot help with that.",
                    model=model,
                )
            return fake_response(content="Clean response", model=model)

        mock_litellm.side_effect = side_effect

        initial_updates = getattr(routesmith.router.predictor, "_total_updates", 0)

        routesmith.completion(
            messages=[{"role": "user", "content": "hello"}],
        )

        # model-cheap rejected -> predictor update with score=0.1
        # model-medium accepted -> implicit feedback update (via feedback.record)
        final_updates = getattr(routesmith.router.predictor, "_total_updates", 0)
        assert final_updates > initial_updates
