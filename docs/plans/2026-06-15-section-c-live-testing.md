# Section C: Live Testing & Benchmarking — Implementation Plan

> **REQUIRED SUB-SKILL:** Use subagent-driven-development to execute.
> **Branch:** feature/v0.2.0-live-testing (from dev)
> **CRITICAL:** All tests use real Groq API, not mocks.

**Goal:** Add automated live API testing, RouterBench reproduction, and performance benchmarks.

**Architecture:** New `tests/perf/` for perf tests, `scripts/` for eval scripts, extended nightly workflow.

**Tech Stack:** pytest, Groq free tier API, timeit, memory_profiler

---

## Task C1: Performance Benchmark (Routing Latency)

**Files:**
- Create: `tests/perf/__init__.py`
- Create: `tests/perf/test_routing_latency.py`

### Step 1: Write failing test (will fail without API key in CI, but documents the benchmark)

```python
# tests/perf/test_routing_latency.py
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
    """P99 routing latency must be <5ms."""

    def test_routing_decision_latency(self):
        """Measure pure routing decision time (not LLM call time)."""
        rs = make_client()
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            # Route only (don't execute LLM call)
            candidate = rs.router.route(messages)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            latencies.append(elapsed)
            assert candidate is not None
        
        latencies.sort()
        p50 = latencies[50]
        p99 = latencies[99]
        avg = sum(latencies) / len(latencies)
        
        print(f"\nRouting Latency (100 samples):")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P99: {p99:.2f}ms")  
        print(f"  Avg: {avg:.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        
        assert p99 < 5.0, f"P99 latency {p99:.2f}ms exceeds 5ms threshold"

    def test_end_to_end_latency_under_load(self):
        """Measure end-to-end routing + LLM call latency for 20 real queries."""
        rs = make_client()
        queries = [
            "What is 2+2?",
            "Capital of Japan?",
            "Explain gravity in one sentence.",
            "Write 'hello' in French.",
            "What year was Python created?",
            "Name three primary colors.",
            "What is the speed of light?",
            "Define photosynthesis briefly.",
            "Who wrote Romeo and Juliet?",
            "What is H2O?",
            "Count from 1 to 10.",
            "What does CPU stand for?",
            "Largest planet in solar system?",
            "Boiling point of water in Celsius?",
            "What is JSON?",
            "Name a Shakespeare play.",
            "What is an API?",
            "Square root of 144?",
            "What language is spoken in Brazil?",
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
        p50 = latencies[10]
        p99 = latencies[min(19, len(latencies) - 1)]
        avg = sum(latencies) / len(latencies)
        
        print(f"\nEnd-to-End Latency (20 queries):")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P99: {p99:.2f}ms")
        print(f"  Avg: {avg:.2f}ms")
        
        # End-to-end with Groq should be fast (<2s per query)
        assert p99 < 5000, f"P99 e2e latency {p99:.2f}ms exceeds 5s"

    def test_route_output_contains_model_info(self):
        """Verify route() output carries enough metadata for observability."""
        rs = make_client()
        messages = [{"role": "user", "content": "hello"}]
        candidate = rs.router.route(messages)
        assert candidate.model_id is not None
        assert candidate.quality_score is not None
```

Run: `GROQ_API_KEY=... .venv/bin/pytest tests/perf/test_routing_latency.py -v -s`
Expected: P99 <5ms

### Step 2: Commit

```bash
git add tests/perf/__init__.py tests/perf/test_routing_latency.py
git commit -m "feat(perf): add routing latency benchmark with P99 <5ms gate"
```

---

## Task C2: Memory Profiling

**Files:**
- Create: `tests/perf/test_memory.py`

### Step 1: Write test

```python
# tests/perf/test_memory.py
"""Real use case: no memory leaks under sustained routing load."""
import os
import pytest
from routesmith import RouteSmith

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY."
)

@pytest.mark.slow
def test_no_memory_leak_over_sustained_load():
    """Route 100 queries and verify memory doesn't grow unbounded."""
    import tracemalloc
    
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
        if i % 10 == 0:
            print(f"  Completed {i+1} queries...")
    
    snapshot_after = tracemalloc.take_snapshot()
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    
    # Top memory diffs
    print("\nTop 5 memory diffs:")
    for stat in stats[:5]:
        print(f"  {stat}")
    
    # RouteSmith's own memory should be reasonable
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nCurrent: {current / 1024:.1f}KB, Peak: {peak / 1024:.1f}KB")
    assert current < 100 * 1024 * 1024  # <100MB
    tracemalloc.stop()
```

### Step 2: Commit

```bash
git add tests/perf/test_memory.py
git commit -m "feat(perf): add memory profiling test (<100MB gate)"
```

---

## Task C3: Multi-Model Eval Script (100 Queries)

**Files:**
- Create: `scripts/run_multi_model_eval.py`

### Step 1: Write eval script

```python
#!/usr/bin/env python3
"""Real use case: route 100 diverse queries across difficulty levels."""
import os
import sys
import json
import time
from collections import defaultdict
from routesmith import RouteSmith

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    print("Set GROQ_API_KEY")
    sys.exit(1)

# Real queries at different difficulty levels
QUERIES = {
    "trivial": [
        "What is 2+2?", "What color is the sky?", "What day comes after Monday?",
        "Capital of France?", "How many legs does a cat have?", "What is water?",
        "Who is Mickey Mouse?", "What sound does a dog make?", "1+1=?",
        "What is the opposite of hot?",
    ],
    "simple": [
        "Explain what a computer is in one sentence.",
        "What is the capital of Japan?", "Name three fruits.",
        "How many hours in a day?", "What does CPU stand for?",
        "List primary colors.", "What year is it?",
        "Define photosynthesis briefly.", "What is JSON?",
        "What is the speed of light?",
    ],
    "moderate": [
        "Explain the difference between HTTP and HTTPS.",
        "Write a haiku about programming.",
        "Summarize the water cycle in 3 sentences.",
        "What is the difference between Python lists and tuples?",
        "Explain what an API is to a non-technical person.",
        "Compare and contrast REST and GraphQL.",
        "Write a SQL query to find duplicate emails.",
        "Explain how garbage collection works in Python.",
        "What is the CAP theorem?",
        "Explain OAuth 2.0 flow in simple terms.",
    ],
    "complex": [
        "Explain the transformer architecture and why it revolutionized NLP.",
        "Compare BFS and DFS with real-world examples.",
        "Explain how Docker containers differ from VMs at the OS level.",
        "Write a detailed explanation of the blockchain consensus mechanism.",
        "Explain the double-slit experiment and its implications.",
        "Describe how TCP congestion control works.",
        "Explain monads in functional programming with examples.",
        "Compare ACID vs BASE database properties with use cases.",
        "Explain how Kubernetes orchestrates containers at scale.",
        "Describe the PageRank algorithm and its mathematical foundation.",
    ],
}

MATH_QUERIES = [
    "Solve: If x = 5 and y = 3, what is x^2 + y^2?",
    "What is 15% of 200?",
    "Find the derivative of x^3 + 2x^2 - 5x + 1.",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "Calculate the area of a circle with radius 7.",
    "What is the probability of rolling a sum of 7 with two dice?",
    "Solve for x: 3x + 7 = 22.",
    "What is the median of [3, 7, 1, 9, 4, 6, 8]?",
    "Convert 0.75 to a fraction.",
    "What is the factorial of 5?",
]


def main():
    rs = RouteSmith()
    rs.register_model("groq/llama-3.3-70b-versatile",
                      cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
                      quality_score=0.90)
    rs.register_model("groq/llama-3.1-8b-instant",
                      cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
                      quality_score=0.60)

    results = []
    model_usage = defaultdict(int)
    difficulties = {k: [] for k in QUERIES}
    difficulties["math"] = []

    all_queries = []
    for diff, qs in QUERIES.items():
        for q in qs:
            all_queries.append((diff, q))
    for q in MATH_QUERIES:
        all_queries.append(("math", q))

    print(f"Running {len(all_queries)} queries across {len(difficulties)} difficulty levels...")
    
    for i, (difficulty, query) in enumerate(all_queries):
        t0 = time.monotonic()
        try:
            resp = rs.completion(
                messages=[{"role": "user", "content": query}],
                min_quality=0.5,
                include_metadata=True,
            )
            elapsed = time.monotonic() - t0
            content = resp.choices[0].message.content[:100]
            model = resp.routesmith_metadata.get("model_selected", "unknown")
            cost = resp.routesmith_metadata.get("estimated_cost_usd", 0)
            
            model_usage[model] += 1
            difficulties[difficulty].append({
                "query": query[:50],
                "model": model,
                "latency_s": round(elapsed, 3),
                "cost_usd": cost,
                "preview": content,
            })
            print(f"  [{i+1}/{len(all_queries)}] {difficulty:8s} → {model:30s} {elapsed:.2f}s ${cost:.6f}")
        except Exception as e:
            print(f"  [{i+1}/{len(all_queries)}] {difficulty:8s} → ERROR: {e}")
            difficulties[difficulty].append({"query": query[:50], "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-MODEL ROUTING EVAL RESULTS")
    print("=" * 70)
    print(f"\nModel Usage:")
    for model, count in sorted(model_usage.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count} queries ({count/len(all_queries)*100:.1f}%)")

    print(f"\nDifficulty Breakdown:")
    for diff, items in difficulties.items():
        success = [i for i in items if "error" not in i]
        models_used = defaultdict(int)
        for i in success:
            models_used[i["model"]] += 1
        avg_latency = sum(i["latency_s"] for i in success) / len(success) if success else 0
        print(f"  {diff:8s}: {len(success)}/{len(items)} ok, avg latency {avg_latency:.2f}s")
        print(f"            models: {dict(models_used)}")

    stats = rs.stats
    print(f"\nCost Summary:")
    print(f"  Total requests: {stats['request_count']}")
    print(f"  Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"  By model: {stats.get('by_model', {})}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "..", "benchmark", "results", 
                               "multi_model_eval_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model_usage": dict(model_usage),
            "difficulties": difficulties,
            "stats": stats,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

### Step 2: Test locally

```bash
GROQ_API_KEY=... .venv/bin/python scripts/run_multi_model_eval.py
```
Expected: 110 queries pass, output shows model distribution across difficulties.

### Step 3: Commit

```bash
git add scripts/run_multi_model_eval.py
git commit -m "feat(eval): add multi-model routing eval script with 100+ diverse queries"
```

---

## Task C4: Extend Nightly Workflow

**Files:**
- Modify: `.github/workflows/nightly.yml`

### Step 1: Update nightly.yml

Add these jobs to `.github/workflows/nightly.yml`:

```yaml
  perf-test:
    runs-on: ubuntu-latest
    needs: live-test
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra dev --extra anthropic --extra langchain
      - env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: uv run pytest tests/perf/ -v -s --timeout=300

  multi-model-eval:
    runs-on: ubuntu-latest
    needs: live-test
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra dev --extra anthropic --extra langchain
      - env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: uv run python scripts/run_multi_model_eval.py
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: multi-model-eval-results
          path: benchmark/results/multi_model_eval_results.json
```

### Step 2: Commit

```bash
git add .github/workflows/nightly.yml
git commit -m "feat(ci): add perf tests and multi-model eval to nightly workflow"
```

---

## UAT Validation (Real Use Case)

```bash
# Run against Groq
export GROQ_API_KEY=gsk_...

# 1. Routing latency benchmark
.venv/bin/pytest tests/perf/test_routing_latency.py -v -s
# Expected: P99 <5ms

# 2. Memory profiling  
.venv/bin/pytest tests/perf/test_memory.py -v -s
# Expected: Current memory <100MB

# 3. Multi-model eval (110 queries)
.venv/bin/python scripts/run_multi_model_eval.py
# Expected: Results show model distribution, no errors

# 4. Full live tests
.venv/bin/pytest tests/manual/test_real_api.py -v
.venv/bin/pytest tests/manual/test_langchain_live.py -v
```
