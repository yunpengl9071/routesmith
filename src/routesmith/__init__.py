"""
Routesmith - Adaptive LLM Execution Engine

Intelligent routing, cascading, semantic caching, and budget management for LLM applications.
"""

from routesmith.config import (
    BudgetBehavior,
    CostModel,
    RouteContext,
    RouteSmithConfig,
    RoutingStrategy,
)
from routesmith.exceptions import (
    BudgetExceededError,
    CapacityExhaustedError,
    CircuitOpenError,
    NoCapableModelError,
    NoCompliantModelError,
    NoProviderDetectedError,
    ProviderUnavailableError,
    RouteSmithError,
)
from routesmith.predictor.lints import LinTSPredictor, LinTSRouter
from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.registry.models import ModelConfig, ModelRegistry
from routesmith.strategy.ab_test import ABTestRunner

__version__ = "0.9.3"


def __getattr__(name):
    # RouteSmith/RoutingMetadata live in routesmith.client, which imports litellm (a
    # heavy, native-built dependency). Load them lazily (PEP 562) so lightweight
    # submodules — notably routesmith.research (the OPE instrument) — can be imported
    # without pulling in the full LLM-calling stack. `from routesmith import RouteSmith`
    # and attribute access still work exactly as before, resolving on first use.
    if name in ("RouteSmith", "RoutingMetadata"):
        from routesmith.client import RouteSmith, RoutingMetadata
        globals()["RouteSmith"] = RouteSmith
        globals()["RoutingMetadata"] = RoutingMetadata
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ABTestRunner",
    "BudgetBehavior",
    "BudgetExceededError",
    "CapacityExhaustedError",
    "CircuitOpenError",
    "CostModel",
    "LinTSPredictor",
    "LinTSRouter",
    "LinUCBPredictor",
    "ModelConfig",
    "ModelRegistry",
    "NoCapableModelError",
    "NoCompliantModelError",
    "NoProviderDetectedError",
    "ProviderUnavailableError",
    "RouteContext",
    "RouteSmith",
    "RouteSmithConfig",
    "RouteSmithError",
    "RoutingMetadata",
    "RoutingStrategy",
]
