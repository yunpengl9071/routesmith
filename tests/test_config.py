"""Tests for configuration."""

import pytest
from routesmith.config import (
    BudgetConfig,
    CacheConfig,
    RouteContext,
    RouteSmithConfig,
    RoutingStrategy,
)


class TestRouteSmithConfig:
    def test_default_config(self):
        config = RouteSmithConfig()
        assert config.default_strategy == RoutingStrategy.DIRECT
        assert config.cache.enabled is False
        assert config.budget.quality_threshold == 0.8

    def test_with_cache(self):
        config = RouteSmithConfig()
        new_config = config.with_cache(enabled=True, similarity_threshold=0.9)
        assert new_config.cache.enabled is True
        assert new_config.cache.similarity_threshold == 0.9
        # Original unchanged
        assert config.cache.enabled is False

    def test_with_budget(self):
        config = RouteSmithConfig()
        new_config = config.with_budget(max_cost_per_request=0.01)
        assert new_config.budget.max_cost_per_request == 0.01
        # Original unchanged
        assert config.budget.max_cost_per_request is None


class TestRoutingStrategy:
    def test_strategy_values(self):
        assert RoutingStrategy.DIRECT.value == "direct"
        assert RoutingStrategy.CASCADE.value == "cascade"
        assert RoutingStrategy.PARALLEL.value == "parallel"
        assert RoutingStrategy.SPECULATIVE.value == "speculative"


class TestRouteContext:
    def test_all_optional(self):
        ctx = RouteContext()
        assert ctx.agent_id is None
        assert ctx.agent_role is None
        assert ctx.conversation_id is None
        assert ctx.turn_index is None
        assert ctx.metadata == {}

    def test_partial_construction(self):
        ctx = RouteContext(agent_role="research", turn_index=3)
        assert ctx.agent_role == "research"
        assert ctx.turn_index == 3
        assert ctx.agent_id is None


class TestRouteSmithConfigExtensions:
    def test_reward_fns_default_empty(self):
        config = RouteSmithConfig()
        assert config.reward_fns == {}

    def test_business_rules_default_empty(self):
        config = RouteSmithConfig()
        assert config.business_rules == []

    def test_reward_fns_set(self):
        fn = lambda r, m: 0.9
        config = RouteSmithConfig(reward_fns={"research": fn})
        assert config.reward_fns["research"] is fn
