"""
Routesmith - Adaptive LLM Execution Engine

Intelligent routing, cascading, semantic caching, and budget management for LLM applications.
"""

from routesmith.client import RouteSmith, RoutingMetadata
from routesmith.config import (
    BudgetBehavior,
    BudgetConfig,
    CacheConfig,
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
    ProviderUnavailableError,
    RouteSmithError,
)
from routesmith.predictor.lints import LinTSPredictor, LinTSRouter
from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.predictor.neural_ucb import NeuralUCBPredictor
from routesmith.predictor.reinforce import ReinforcePredictor
from routesmith.predictor.warmstart_linucb import WarmStartLinUCBPredictor
from routesmith.registry.models import ModelConfig, ModelRegistry
from routesmith.strategy.ab_test import ABTestRunner

__version__ = "0.5.0"

__all__ = [
    "ABTestRunner",
    "BudgetBehavior",
    "BudgetConfig",
    "BudgetExceededError",
    "CacheConfig",
    "CapacityExhaustedError",
    "CircuitOpenError",
    "CostModel",
    "LinTSPredictor",
    "LinTSRouter",
    "LinUCBPredictor",
    "ModelConfig",
    "ModelRegistry",
    "NeuralUCBPredictor",
    "NoCapableModelError",
    "NoCompliantModelError",
    "ProviderUnavailableError",
    "ReinforcePredictor",
    "RouteContext",
    "RouteSmith",
    "RouteSmithConfig",
    "RouteSmithError",
    "RoutingMetadata",
    "RoutingStrategy",
    "WarmStartLinUCBPredictor",
]
