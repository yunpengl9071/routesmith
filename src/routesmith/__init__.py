"""
Routesmith - Adaptive LLM Execution Engine

Intelligent routing, cascading, semantic caching, and budget management for LLM applications.
"""

from routesmith.client import RouteSmith, RoutingMetadata
from routesmith.config import RouteSmithConfig, RoutingStrategy
from routesmith.registry.models import ModelConfig, ModelRegistry
from routesmith.strategy.ab_test import ABTestRunner

__version__ = "0.1.0"

__all__ = [
    "ABTestRunner",
    "RouteSmith",
    "RouteSmithConfig",
    "RoutingStrategy",
    "RoutingMetadata",
    "ModelConfig",
    "ModelRegistry",
]
