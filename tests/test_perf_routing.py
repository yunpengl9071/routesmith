"""Routing-overhead performance guardrail (G3)."""

import os
import time

import pytest

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig, RoutingStrategy

PERF_MULTIPLIER = float(os.environ.get("PERF_MULTIPLIER", "1.0"))


@pytest.fixture
def five_model_registry():
    rs = RouteSmith(config=RouteSmithConfig(predictor_type="lints"))
    rs.register_model("model-a", 0.0001, 0.0002, quality_score=0.70)
    rs.register_model("model-b", 0.0005, 0.0010, quality_score=0.78)
    rs.register_model("model-c", 0.0010, 0.0020, quality_score=0.85)
    rs.register_model("model-d", 0.0050, 0.0100, quality_score=0.92)
    rs.register_model("model-e", 0.0100, 0.0200, quality_score=0.96)
    return rs


@pytest.mark.perf
def test_routing_overhead_below_threshold(five_model_registry):
    """router.route() must complete within 5ms p50 / 15ms p99."""
    rs = five_model_registry
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Explain your reasoning step by step."},
    ]

    # Warmup
    for _ in range(20):
        rs.router.route(messages, strategy=RoutingStrategy.DIRECT, min_quality=0.0)

    # Measure
    deltas = []
    for _ in range(200):
        t0 = time.perf_counter()
        rs.router.route(messages, strategy=RoutingStrategy.DIRECT, min_quality=0.0)
        deltas.append(time.perf_counter() - t0)

    deltas.sort()
    p50 = deltas[len(deltas) // 2]
    p99 = deltas[int(len(deltas) * 0.99)]

    p50_limit = 0.005 * PERF_MULTIPLIER
    p99_limit = 0.015 * PERF_MULTIPLIER

    assert p50 < p50_limit, f"p50={p50*1000:.2f}ms >= {p50_limit*1000:.2f}ms"
    assert p99 < p99_limit, f"p99={p99*1000:.2f}ms >= {p99_limit*1000:.2f}ms"
