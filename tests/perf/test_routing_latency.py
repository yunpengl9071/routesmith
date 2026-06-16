"""Real use case: P99 routing latency <5ms under real routing load."""
import os
import time

import pytest

_live = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY for live API calls."
)


class TestRoutingLatency:
    """Pure routing latency - no API key needed."""

    def test_routing_decision_latency(self):
        """Measure pure routing decision time. P99 must be <5ms."""
        from routesmith import RouteSmith
        rs = RouteSmith()
        rs.register_model("groq/llama-3.3-70b-versatile",
                          cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
                          quality_score=0.90)
        rs.register_model("groq/llama-3.1-8b-instant",
                          cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
                          quality_score=0.60)
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            candidate = rs.router.route(messages)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            assert isinstance(candidate, str) and len(candidate) > 0
        latencies.sort()
        p99 = latencies[99]
        avg = sum(latencies) / len(latencies)
        print(f"\nRouting Latency: avg={avg:.2f}ms, P99={p99:.2f}ms")
        assert p99 < 5.0, f"P99 latency {p99:.2f}ms exceeds 5ms threshold"

    def test_route_output_contains_model_info(self):
        """Verify route() output carries enough metadata for observability."""
        from routesmith import RouteSmith
        rs = RouteSmith()
        rs.register_model("groq/llama-3.3-70b-versatile",
                          cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
                          quality_score=0.90)
        rs.register_model("groq/llama-3.1-8b-instant",
                          cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
                          quality_score=0.60)
        messages = [{"role": "user", "content": "hello"}]
        candidate = rs.router.route(messages)
        assert isinstance(candidate, str)
        assert len(candidate) > 0

    @_live
    def test_end_to_end_latency_under_load(self):
        """Measure end-to-end routing + LLM call latency for 20 real queries."""
        from routesmith import RouteSmith
        rs = RouteSmith()
        rs.register_model("groq/llama-3.3-70b-versatile",
                          cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
                          quality_score=0.90)
        rs.register_model("groq/llama-3.1-8b-instant",
                          cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
                          quality_score=0.60)
        queries = [
            "What is 2+2?", "Capital of Japan?", "Explain gravity in one sentence.",
            "Write 'hello' in French.", "What year was Python created?",
            "Name three primary colors.", "What is the speed of light?",
            "Define photosynthesis briefly.", "Who wrote Romeo and Juliet?",
            "What is H2O?", "Count from 1 to 10.", "What does CPU stand for?",
            "Largest planet in solar system?", "Boiling point of water?",
            "What is JSON?", "Name a Shakespeare play.", "What is an API?",
            "Square root of 144?", "What language is spoken in Brazil?",
            "How many continents are there?",
        ]
        latencies = []
        for q in queries:
            t0 = time.perf_counter()
            resp = rs.completion(messages=[{"role": "user", "content": q}],
                                min_quality=0.5)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            assert resp.choices[0].message.content is not None
        latencies.sort()
        p99 = latencies[min(19, len(latencies) - 1)]
        avg = sum(latencies) / len(latencies)
        print(f"\nEnd-to-End Latency (20 queries): avg={avg:.0f}ms, P99={p99:.0f}ms")
        assert p99 < 5000, f"P99 e2e latency {p99:.0f}ms exceeds 5s"
