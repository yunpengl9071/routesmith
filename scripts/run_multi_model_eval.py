#!/usr/bin/env python3
"""Real use case: route 110 diverse queries across difficulty levels with Groq."""
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
            print(f"  [{i+1}/{len(all_queries)}] {difficulty:8s} \u2192 {model:30s} {elapsed:.2f}s ${cost:.6f}")
        except Exception as e:
            print(f"  [{i+1}/{len(all_queries)}] {difficulty:8s} \u2192 ERROR: {e}")

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

    output_path = os.path.join(os.path.dirname(__file__), "..", "benchmark", "results",
                               "multi_model_eval_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model_usage": dict(model_usage),
            "difficulties": dict(difficulties),
            "stats": stats,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()