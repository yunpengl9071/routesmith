"""Tests for CostModel enum and related config."""
import pytest
from routesmith.config import BudgetBehavior, CostModel


class TestCostModelEnum:
    def test_enum_values(self):
        assert CostModel.ON_DEMAND.value == "on_demand"
        assert CostModel.PROVISIONED.value == "provisioned"
        assert CostModel.SELF_HOSTED.value == "self_hosted"

    def test_enum_from_string(self):
        assert CostModel("on_demand") == CostModel.ON_DEMAND
        assert CostModel("provisioned") == CostModel.PROVISIONED
        assert CostModel("self_hosted") == CostModel.SELF_HOSTED

    def test_default_is_on_demand(self):
        from routesmith.registry.models import ModelConfig
        model = ModelConfig(
            model_id="test-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        assert model.cost_model == CostModel.ON_DEMAND

    def test_provisioned_fields_default_zero(self):
        from routesmith.registry.models import ModelConfig
        model = ModelConfig(
            model_id="test-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        assert model.capacity_requests_per_min == 0
        assert model.provisioned_hourly_cost == 0.0
        assert model.provisioned_units == 0


class TestBudgetBehaviorEnum:
    def test_enum_values(self):
        assert BudgetBehavior.FAIL.value == "fail"
        assert BudgetBehavior.FALLBACK.value == "fallback"
        assert BudgetBehavior.QUEUE.value == "queue"

    def test_enum_from_string(self):
        assert BudgetBehavior("fail") == BudgetBehavior.FAIL
        assert BudgetBehavior("fallback") == BudgetBehavior.FALLBACK
        assert BudgetBehavior("queue") == BudgetBehavior.QUEUE


class TestRegistryFiltersByCostModel:
    def test_filter_by_cost_model_returns_only_matching(self):
        from routesmith.registry.models import ModelRegistry
        from routesmith.config import CostModel

        reg = ModelRegistry()
        reg.register("on-demand-model", 0.001, 0.002, cost_model=CostModel.ON_DEMAND)
        reg.register("provisioned-model", 0.0, 0.0, cost_model=CostModel.PROVISIONED)
        reg.register("self-hosted-model", 0.0, 0.0, cost_model=CostModel.SELF_HOSTED)

        on_demand = reg.filter_by_cost_model(CostModel.ON_DEMAND)
        assert len(on_demand) == 1
        assert on_demand[0].model_id == "on-demand-model"

        provisioned = reg.filter_by_cost_model(CostModel.PROVISIONED)
        assert len(provisioned) == 1
        assert provisioned[0].model_id == "provisioned-model"

    def test_filter_by_cost_model_returns_empty_if_none_match(self):
        from routesmith.registry.models import ModelRegistry
        from routesmith.config import CostModel

        reg = ModelRegistry()
        reg.register("on-demand-model", 0.001, 0.002, cost_model=CostModel.ON_DEMAND)

        provisioned = reg.filter_by_cost_model(CostModel.PROVISIONED)
        assert len(provisioned) == 0