"""Tests for A/B test framework."""

import pytest
from unittest.mock import patch, MagicMock

from routesmith import RouteSmith, RouteSmithConfig, ABTestRunner
from routesmith.strategy.ab_test import ABTestResults, VariantStats, MIN_SAMPLES_FOR_WINNER


def _make_mock_response(request_id: str, model: str = "gpt-4o-mini"):
    """Create a mock ModelResponse with routesmith metadata."""
    response = MagicMock()
    response._routesmith_request_id = request_id
    response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return response


def _make_client_with_models():
    """Create a RouteSmith client with test models registered."""
    config = RouteSmithConfig(feedback_sample_rate=1.0)
    rs = RouteSmith(config=config)
    rs.register_model(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
    )
    rs.register_model(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.80,
    )
    return rs


class TestABTestRunnerInit:
    """Tests for ABTestRunner initialization."""

    def test_init_defaults(self):
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        assert ab.traffic_split == 0.5
        assert ab.name == ""
        assert ab._variant_counts == {"A": 0, "B": 0}

    def test_init_custom_split(self):
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.3, name="test-1")

        assert ab.traffic_split == 0.3
        assert ab.name == "test-1"

    def test_invalid_traffic_split(self):
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()

        with pytest.raises(ValueError, match="traffic_split"):
            ABTestRunner(client_a, client_b, traffic_split=1.5)

        with pytest.raises(ValueError, match="traffic_split"):
            ABTestRunner(client_a, client_b, traffic_split=-0.1)


class TestTrafficSplit:
    """Tests for traffic assignment."""

    def test_all_traffic_to_a(self):
        """traffic_split=0.0 sends everything to A."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.0)

        variants = [ab._assign_variant() for _ in range(100)]
        assert all(v == "A" for v in variants)

    def test_all_traffic_to_b(self):
        """traffic_split=1.0 sends everything to B."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=1.0)

        variants = [ab._assign_variant() for _ in range(100)]
        assert all(v == "B" for v in variants)

    @patch("routesmith.strategy.ab_test.random.random")
    def test_deterministic_assignment(self, mock_random):
        """Verify assignment logic with controlled random values."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.5)

        # random() returns 0.3 < 0.5 → B
        mock_random.return_value = 0.3
        assert ab._assign_variant() == "B"

        # random() returns 0.7 >= 0.5 → A
        mock_random.return_value = 0.7
        assert ab._assign_variant() == "A"

        # random() returns exactly 0.5 → A (not strictly less than)
        mock_random.return_value = 0.5
        assert ab._assign_variant() == "A"


class TestCompletion:
    """Tests for completion routing through A/B runner."""

    @patch("litellm.completion")
    @patch("routesmith.strategy.ab_test.random.random")
    def test_routes_to_correct_client(self, mock_random, mock_litellm):
        """Requests are routed to the correct variant's client."""
        mock_response = _make_mock_response("req-1")
        mock_litellm.return_value = mock_response

        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.5)

        messages = [{"role": "user", "content": "Hello"}]

        # Force variant A
        mock_random.return_value = 0.8
        ab.completion(messages=messages)
        assert ab._variant_counts["A"] == 1
        assert ab._variant_counts["B"] == 0

        # Force variant B
        mock_random.return_value = 0.2
        mock_response._routesmith_request_id = "req-2"
        ab.completion(messages=messages)
        assert ab._variant_counts["A"] == 1
        assert ab._variant_counts["B"] == 1

    @patch("litellm.completion")
    def test_request_id_mapping(self, mock_litellm):
        """Request IDs are mapped to the correct variant."""
        mock_response = _make_mock_response("placeholder")
        mock_litellm.return_value = mock_response

        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.0)  # all to A

        messages = [{"role": "user", "content": "test"}]
        response = ab.completion(messages=messages)

        # The client generates its own request_id
        req_id = response._routesmith_request_id
        assert req_id in ab._request_map
        assert ab._request_map[req_id] == "A"


class TestRecordOutcome:
    """Tests for outcome recording."""

    @patch("litellm.completion")
    def test_record_outcome_routes_to_correct_client(self, mock_litellm):
        """record_outcome delegates to the correct variant's client."""
        mock_response = _make_mock_response("placeholder")
        mock_litellm.return_value = mock_response

        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.0)  # all to A

        messages = [{"role": "user", "content": "test"}]
        response = ab.completion(messages=messages)
        req_id = response._routesmith_request_id

        result = ab.record_outcome(req_id, score=0.9)
        assert result is True
        assert ab._variant_quality["A"] == [0.9]
        assert ab._variant_quality["B"] == []

    def test_record_outcome_unknown_request(self):
        """Recording outcome for unknown request returns False."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        assert ab.record_outcome("nonexistent", score=0.5) is False

    @patch("litellm.completion")
    def test_quality_scores_tracked(self, mock_litellm):
        """Quality scores are accumulated per variant."""
        mock_litellm.return_value = _make_mock_response("placeholder")

        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b, traffic_split=0.0)

        messages = [{"role": "user", "content": "test"}]
        resp1 = ab.completion(messages=messages)
        ab.record_outcome(resp1._routesmith_request_id, score=0.8)

        resp2 = ab.completion(messages=messages)
        ab.record_outcome(resp2._routesmith_request_id, score=0.9)

        assert ab._variant_quality["A"] == [0.8, 0.9]


class TestResults:
    """Tests for results aggregation."""

    def test_empty_results(self):
        """Results with no requests."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        results = ab.results()
        assert results.variant_a.request_count == 0
        assert results.variant_b.request_count == 0
        assert results.winner is None
        assert results.quality_diff is None

    def test_insufficient_data_no_winner(self):
        """Winner is None when fewer than MIN_SAMPLES_FOR_WINNER scored samples."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        # Add some quality scores, but fewer than threshold
        for i in range(10):
            ab._variant_quality["A"].append(0.8)
            ab._variant_quality["B"].append(0.9)
            ab._variant_counts["A"] += 1
            ab._variant_counts["B"] += 1

        results = ab.results()
        assert results.winner is None

    def test_winner_with_sufficient_data(self):
        """Winner is determined when enough scored samples exist."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        for _ in range(MIN_SAMPLES_FOR_WINNER):
            ab._variant_quality["A"].append(0.7)
            ab._variant_quality["B"].append(0.9)
            ab._variant_counts["A"] += 1
            ab._variant_counts["B"] += 1

        results = ab.results()
        assert results.winner == "B"
        assert results.quality_diff is not None
        assert results.quality_diff > 0

    def test_winner_a(self):
        """Variant A wins when it has higher quality."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        for _ in range(MIN_SAMPLES_FOR_WINNER):
            ab._variant_quality["A"].append(0.95)
            ab._variant_quality["B"].append(0.6)
            ab._variant_counts["A"] += 1
            ab._variant_counts["B"] += 1

        results = ab.results()
        assert results.winner == "A"
        assert results.quality_diff is not None
        assert results.quality_diff < 0

    def test_equal_quality_no_winner(self):
        """No winner when quality is identical."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        for _ in range(MIN_SAMPLES_FOR_WINNER):
            ab._variant_quality["A"].append(0.8)
            ab._variant_quality["B"].append(0.8)
            ab._variant_counts["A"] += 1
            ab._variant_counts["B"] += 1

        results = ab.results()
        assert results.winner is None
        assert results.quality_diff == 0.0

    def test_cost_diff_percentage(self):
        """Cost difference percentage is computed correctly."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        ab._variant_counts = {"A": 10, "B": 10}
        ab._variant_costs = {"A": 1.0, "B": 0.5}  # B is 50% cheaper

        results = ab.results()
        assert results.cost_diff_pct == -50.0

    def test_cost_diff_none_when_a_zero(self):
        """Cost diff is None when A has zero cost (avoid division by zero)."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        ab._variant_counts = {"A": 0, "B": 5}
        results = ab.results()
        assert results.cost_diff_pct is None

    def test_models_used_tracking(self):
        """Model usage is tracked per variant."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        ab._variant_models["A"]["gpt-4o-mini"] = 8
        ab._variant_models["A"]["gpt-4o"] = 2
        ab._variant_counts["A"] = 10

        results = ab.results()
        assert results.variant_a.models_used == {"gpt-4o-mini": 8, "gpt-4o": 2}


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_all_state(self):
        """Reset clears all tracked state."""
        client_a = _make_client_with_models()
        client_b = _make_client_with_models()
        ab = ABTestRunner(client_a, client_b)

        # Add some state
        ab._request_map["req-1"] = "A"
        ab._variant_counts = {"A": 5, "B": 3}
        ab._variant_costs = {"A": 1.0, "B": 0.5}
        ab._variant_quality = {"A": [0.8, 0.9], "B": [0.7]}
        ab._variant_models["A"]["gpt-4o"] = 5

        ab.reset()

        assert ab._request_map == {}
        assert ab._variant_counts == {"A": 0, "B": 0}
        assert ab._variant_costs == {"A": 0.0, "B": 0.0}
        assert ab._variant_quality == {"A": [], "B": []}
        results = ab.results()
        assert results.variant_a.request_count == 0
        assert results.variant_b.request_count == 0


class TestABTestResultsDataclass:
    """Tests for the result dataclasses."""

    def test_variant_stats_avg_quality_none(self):
        """avg_quality is None when no quality scores."""
        stats = VariantStats(
            name="A",
            request_count=5,
            total_cost_usd=0.01,
            avg_cost_per_request=0.002,
            quality_scores=[],
            avg_quality=None,
            models_used={"gpt-4o": 5},
        )
        assert stats.avg_quality is None

    def test_ab_results_structure(self):
        """ABTestResults has expected fields."""
        stats_a = VariantStats("A", 10, 0.1, 0.01, [0.8], 0.8, {"gpt-4o": 10})
        stats_b = VariantStats("B", 10, 0.05, 0.005, [0.9], 0.9, {"gpt-4o-mini": 10})

        results = ABTestResults(
            variant_a=stats_a,
            variant_b=stats_b,
            winner="B",
            quality_diff=0.1,
            cost_diff_pct=-50.0,
        )
        assert results.winner == "B"
        assert results.quality_diff == 0.1
        assert results.cost_diff_pct == -50.0
