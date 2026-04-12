"""
Routesmith - Adaptive LLM Execution Engine

Intelligent routing, cascading, semantic caching, and budget management for LLM applications.
"""

from routesmith.client import RouteSmith, RoutingMetadata
from routesmith.config import RouteContext, RouteSmithConfig, RoutingStrategy
from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.predictor.lints import LinTSPredictor, LinTSRouter
from routesmith.registry.models import ModelConfig, ModelRegistry
from routesmith.strategy.ab_test import ABTestRunner

__version__ = "0.1.0"

__all__ = [
    "ABTestRunner",
    "LinTSPredictor",
    "LinTSRouter",
    "LinUCBPredictor",
    "ModelConfig",
    "ModelRegistry",
    "RouteContext",
    "RouteSmith",
    "RouteSmithConfig",
    "RoutingMetadata",
    "RoutingStrategy",
]
