"""Real use case: no memory leaks under sustained routing load."""
import os

import pytest

_live = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY."
)


@_live
def test_no_memory_leak_over_sustained_load():
    """Route 50 queries and verify memory doesn't grow unbounded."""
    import tracemalloc

    from routesmith import RouteSmith

    rs = RouteSmith()
    rs.register_model("groq/llama-3.3-70b-versatile",
                      cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
                      quality_score=0.90)
    rs.register_model("groq/llama-3.1-8b-instant",
                      cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
                      quality_score=0.60)

    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    for i in range(50):
        rs.completion(messages=messages, min_quality=0.5)
        if i % 10 == 0 and i > 0:
            print(f"  Completed {i} queries...")

    snapshot_after = tracemalloc.take_snapshot()
    stats = snapshot_after.compare_to(snapshot_before, "lineno")

    print("\nTop 5 memory diffs:")
    for stat in stats[:5]:
        print(f"  {stat}")

    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent: {current / 1024:.1f}KB, Peak: {peak / 1024:.1f}KB")
    assert current < 100 * 1024 * 1024  # <100MB
    tracemalloc.stop()
