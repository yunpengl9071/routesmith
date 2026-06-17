"""Real use case: Prometheus metrics expose all routing data."""
from prometheus_client import CollectorRegistry

from routesmith.proxy.metrics import init_metrics


class TestMetricsRegistration:
    def test_metrics_are_registered(self):
        registry = CollectorRegistry()
        init_metrics(registry)
        names = {m.name for m in registry.collect()}
        assert "routesmith_requests" in names
        assert "routesmith_routing_latency_seconds" in names
        assert "routesmith_cost_usd" in names
        assert "routesmith_cache_hits" in names
        assert "routesmith_active_circuits" in names

    def test_requests_counter_has_labels(self):
        registry = CollectorRegistry()
        init_metrics(registry)
        from routesmith.proxy.metrics import ROUTING_REQUESTS
        ROUTING_REQUESTS.labels(model="gpt-4o", strategy="direct", project="default").inc()

    def test_latency_histogram_has_buckets(self):
        registry = CollectorRegistry()
        init_metrics(registry)
        from routesmith.proxy.metrics import ROUTING_LATENCY
        ROUTING_LATENCY.labels(strategy="cascade").observe(0.003)
        for metric in registry.collect():
            if metric.name == "routesmith_routing_latency_seconds":
                assert len(metric.samples) > 0

    def test_metrics_registry_is_singleton(self):
        reg1 = init_metrics()
        assert init_metrics() is not None
        assert reg1 is not None
