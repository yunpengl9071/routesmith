"""Integration tests combining all v0.3.0 enterprise features."""
import pytest
from unittest.mock import patch

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.config import BudgetBehavior, CostModel, RoutingStrategy
from routesmith.exceptions import BudgetExceededError, CapacityExhaustedError, NoCompliantModelError


def _make_response(model_id):
    from litellm import ModelResponse
    resp = ModelResponse()
    resp.model = model_id
    resp.usage = type(
        "Usage",
        (),
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )()
    resp.choices = [
        type(
            "Choice",
            (),
            {"message": type("Msg", (), {"content": "ok"})()},
        )()
    ]
    return resp


class TestV030Integration:
    """Smoke test: all v0.3.0 features work together."""

    def test_provisioned_first_with_compliance_and_budget(self):
        """Enterprise scenario: HIPAA-compliant provisioned routing with budget."""
        config = RouteSmithConfig()
        config.budget_behavior = BudgetBehavior.FAIL
        rs = RouteSmith(config=config, project="healthcare-app")

        # HIPAA-compliant provisioned model
        rs.register_model(
            "bedrock/claude-sonnet-provisioned",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            quality_score=0.95,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=10,
            provisioned_hourly_cost=66.0,
            provisioned_units=2,
            compliance_tags={"hipaa", "soc2"},
        )

        # On-demand HIPAA model for overflow
        rs.register_model(
            "bedrock/claude-sonnet",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            quality_score=0.92,
            cost_model=CostModel.ON_DEMAND,
            compliance_tags={"hipaa"},
        )

        # Non-HIPAA model (should never be selected with compliance filter)
        rs.register_model(
            "gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _make_response(
                "bedrock/claude-sonnet-provisioned"
            )
            rs.completion(
                messages=[{"role": "user", "content": "Patient summary"}],
                strategy=RoutingStrategy.PROVISIONED_FIRST,
                required_compliance={"hipaa"},
            )

        # Verify provisioned model was selected
        called_model = mock_completion.call_args[1]["model"]
        assert called_model == "bedrock/claude-sonnet-provisioned"

        # Verify stats
        stats = rs.stats
        assert stats["project"] == "healthcare-app"
        assert stats["request_count"] == 1

    def test_budget_fallback_uses_cheapest(self):
        """When budget exhausted with FALLBACK, cheapest model is selected."""
        config = RouteSmithConfig().with_budget(max_cost_per_day=0.001)
        config.budget_behavior = BudgetBehavior.FALLBACK
        rs = RouteSmith(config=config, project="side-project")

        rs.register_model("expensive", 0.005, 0.015, quality_score=0.95)
        rs.register_model("cheap", 0.00015, 0.0006, quality_score=0.70)

        # Simulate budget exhausted
        rs._total_cost = 1.0

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _make_response("cheap")
            rs.completion(
                messages=[{"role": "user", "content": "hello"}],
            )

        assert mock_completion.call_args[1]["model"] == "cheap"
        assert rs.stats["budget_events"]["fallbacks"] == 1

    def test_budget_fail_raises_error(self):
        """When budget exhausted with FAIL, BudgetExceededError is raised."""
        config = RouteSmithConfig().with_budget(max_cost_per_day=0.001)
        config.budget_behavior = BudgetBehavior.FAIL
        rs = RouteSmith(config=config)

        rs.register_model("gpt-4o", 0.005, 0.015)
        rs._total_cost = 1.0  # Force budget exceeded

        with pytest.raises(BudgetExceededError):
            rs.completion(
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_compliance_filter_combined_with_capabilities(self):
        """HIPAA compliance combined with tool_calling capability."""
        config = RouteSmithConfig()
        rs = RouteSmith(config=config)

        # HIPAA model WITHOUT tool support
        rs.register_model(
            "hipaa-no-tools",
            0.001,
            0.002,
            compliance_tags={"hipaa"},
            supports_function_calling=False,
            capabilities=set(),
        )

        # Non-HIPAA model WITH tools
        rs.register_model(
            "no-hipaa-tools",
            0.001,
            0.002,
            compliance_tags=set(),
            supports_function_calling=True,
        )

        # When HIPAA is required but model lacks tools, it should fail
        with pytest.raises(NoCompliantModelError):
            rs.completion(
                messages=[{"role": "user", "content": "use a tool"}],
                required_compliance={"hipaa"},
                tools=[{"type": "function", "function": {"name": "test"}}],
            )

    def test_cost_model_enum_on_registered_models(self):
        """Models retain cost_model and provisioned fields after registration."""
        rs = RouteSmith()
        rs.register_model(
            "provisioned-model",
            0.0,
            0.0,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=15,
            provisioned_hourly_cost=66.0,
            provisioned_units=2,
        )

        model = rs.registry.get("provisioned-model")
        assert model is not None
        assert model.cost_model == CostModel.PROVISIONED
        assert model.capacity_requests_per_min == 15
        assert model.provisioned_hourly_cost == 66.0
        assert model.provisioned_units == 2

    def test_capacity_tracker_tracks_overflow(self):
        """CapacityTracker correctly reports overflow count."""
        config = RouteSmithConfig()
        rs = RouteSmith(config=config)

        rs.register_model(
            "limited-model",
            0.0,
            0.0,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=3,
        )
        rs.register_model(
            "unlimited-model",
            0.001,
            0.002,
            cost_model=CostModel.ON_DEMAND,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _make_response("limited-model")

            # Use all 3 capacity slots
            for i in range(3):
                rs.completion(
                    messages=[{"role": "user", "content": f"msg {i}"}],
                    strategy=RoutingStrategy.PROVISIONED_FIRST,
                )

        # 4th should overflow to on-demand
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _make_response("unlimited-model")
            rs.completion(
                messages=[{"role": "user", "content": "overflow"}],
                strategy=RoutingStrategy.PROVISIONED_FIRST,
            )
            assert mock_completion.call_args[1]["model"] == "unlimited-model"

        tracker = rs.registry.get_capacity_tracker("limited-model")
        assert tracker is not None
        assert tracker.total_requests >= 3
        assert tracker.overflow_count >= 1
        assert not tracker.available()

    def test_project_isolation_between_instances(self):
        """Two instances with different projects have isolated stats."""
        config = RouteSmithConfig()
        rs_a = RouteSmith(config=config, project="project-a")
        rs_b = RouteSmith(config=config, project="project-b")

        rs_a.register_model("gpt-4o-mini", 0.00015, 0.0006)
        rs_b.register_model("gpt-4o-mini", 0.00015, 0.0006)

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _make_response("gpt-4o-mini")
            rs_a.completion(messages=[{"role": "user", "content": "a"}])

        assert rs_a.stats["request_count"] == 1
        assert rs_b.stats["request_count"] == 0
        assert rs_a.stats["project"] == "project-a"
        assert rs_b.stats["project"] == "project-b"

    def test_stats_contains_new_fields(self):
        """Stats dict includes budget_events, by_cost_model, provisioned_utilization."""
        config = RouteSmithConfig()
        rs = RouteSmith(config=config, project="test-project")
        rs.register_model("gpt-4o-mini", 0.00015, 0.0006, quality_score=0.85)

        stats = rs.stats
        assert stats["project"] == "test-project"
        assert "budget_events" in stats
        assert stats["budget_events"]["failures"] == 0
        assert stats["budget_events"]["fallbacks"] == 0
        assert stats["budget_events"]["queued"] == 0
        assert "by_cost_model" in stats
        assert "provisioned_utilization" in stats

    def test_provisioned_first_exhausted_raises(self):
        """PROVISIONED_FIRST raises CapacityExhaustedError when no fallback."""
        config = RouteSmithConfig()
        rs = RouteSmith(config=config)

        rs.register_model(
            "small-capacity",
            0.0,
            0.0,
            cost_model=CostModel.PROVISIONED,
            capacity_requests_per_min=1,
        )

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = _make_response("small-capacity")
            rs.completion(
                messages=[{"role": "user", "content": "first"}],
                strategy=RoutingStrategy.PROVISIONED_FIRST,
            )

        with pytest.raises(CapacityExhaustedError):
            rs.completion(
                messages=[{"role": "user", "content": "second"}],
                strategy=RoutingStrategy.PROVISIONED_FIRST,
            )