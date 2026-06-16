"""Real use case: P99 routing latency <5ms under real routing load."""
import os
import time
import pytest
from routesmith import RouteSmith

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY. Export it to run performance tests."
)


def make_client():
    rs = RouteSmith()
    rs.register_model("groq/llama-3.3-70b-versatile",
                      cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
                      quality_score=0.90)
    rs.register_model("groq/llama-3.1-8b-instant",
                      cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
                      quality_score=0.60)
    return rs


class TestRoutingLatency:
    def test_routing_decision_latency(self):
        """Measure pure routing decision time. P99 must be <5ms."""
        rs = make_client()
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            candidate = rs.router.route(messages)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            assert candidate is not None
        latencies.sort()
        p99 = latencies[99]
        avg = sum(latencies) / len(latencies)
        print(f"\nRouting Latency: avg={avg:.2f}ms, P99={p99:.2f}ms")
        assert p99 < 5.0, f"P99 latency {p99:.2f}ms exceeds 5ms threshold"

    def test_route_output_contains_model_info(self):
        """Verify route() output carries enough metadata for observability."""
        rs = make_client()
        messages = [{"role": "user", "content": "hello"}]
        candidate = rs.router.route(messages)
        assert candidate.model_id is not None
        assert candidate.quality_score is not None