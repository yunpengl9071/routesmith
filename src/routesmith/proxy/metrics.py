"""Prometheus metrics for RouteSmith."""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
from typing import Optional

ROUTING_REQUESTS: Counter
ROUTING_LATENCY: Histogram
COST_USD: Counter
CACHE_HITS: Counter
ACTIVE_CIRCUITS: Gauge

def init_metrics(registry: Optional[CollectorRegistry] = None) -> CollectorRegistry:
    global ROUTING_REQUESTS, ROUTING_LATENCY, COST_USD, CACHE_HITS, ACTIVE_CIRCUITS
    if registry is None:
        registry = REGISTRY
    # Guard against re-registration on the same registry
    try:
        ROUTING_REQUESTS = Counter(
            "routesmith_requests_total", "Total requests routed",
            ["model", "strategy", "project"], registry=registry,
        )
        ROUTING_LATENCY = Histogram(
            "routesmith_routing_latency_seconds", "Routing decision latency",
            ["strategy"], registry=registry,
            buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
        )
        COST_USD = Counter(
            "routesmith_cost_usd_total", "Cumulative cost in USD",
            ["model", "project"], registry=registry,
        )
        CACHE_HITS = Counter(
            "routesmith_cache_hits_total", "Cache hits",
            ["type"], registry=registry,
        )
        ACTIVE_CIRCUITS = Gauge(
            "routesmith_active_circuits", "Number of open circuit breakers",
            registry=registry,
        )
    except ValueError:
        pass  # Already registered on this registry
    return registry