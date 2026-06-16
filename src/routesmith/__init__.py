"""
Routesmith - Adaptive LLM Execution Engine

Intelligent routing, cascading, semantic caching, and budget management for LLM applications.
"""

from routesmith.client import RouteSmith, RoutingMetadata
from routesmith.config import BudgetBehavior, CostModel, RouteContext, RouteSmithConfig, RoutingStrategy
from routesmith.exceptions import (
    BudgetExceededError,
    CircuitOpenError,
    NoCapableModelError,
    ProviderUnavailableError,
    RouteSmithError,
)
from routesmith.predictor.lints import LinTSPredictor, LinTSRouter
from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.registry.models import ModelConfig, ModelRegistry
from routesmith.strategy.ab_test import ABTestRunner

__version__ = "0.2.0"

__all__ = [
    "ABTestRunner",
    "BudgetBehavior",
    "BudgetExceededError",
    "CircuitOpenError",
    "CostModel",
    "LinTSPredictor",
    "LinTSRouter",
    "LinUCBPredictor",
    "ModelConfig",
    "ModelRegistry",
    "NoCapableModelError",
    "ProviderUnavailableError",
    "RouteContext",
    "RouteSmith",
    "RouteSmithConfig",
    "RouteSmithError",
    "RoutingMetadata",
    "RoutingStrategy",
]
