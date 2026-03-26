"""
Run RouteLLM smart router (sw_ranking) on our benchmark queries.
Compare fairly against RouteSmith results.
"""
import os
import json
import sys

# Set up OpenRouter as the API
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")

# Configure litellm for OpenRouter
os.environ["LITELLM_MASTER_KEY"] = os.environ.get("OPENROUTER_API_KEY", "")
os.environ["LITELLM_PROVIDER"] = "openrouter"

from routellm.controller import Controller

# Use a strong model pair that works with OpenRouter
# Strong: GPT-4o mini (cheap but capable)
# Weak: Llama 3.1 8B (much cheaper)
client = Controller(
    routers=["sw_ranking"],  # Use similarity-weighted ranking
    strong_model="openai/gpt-4o-mini",  # Use OpenRouter model name
    weak_model="meta-llama/llama-3.1-8b-instruct",
)

# Load our benchmark queries
benchmark_file = "/home/yliulupo/projects/routesmith/report/benchmark_queries.json"
if os.path.exists(benchmark_file):
    with open(benchmark_file) as f:
        queries = json.load(f)
else:
    # Use some sample queries if benchmark not found
    queries = [
        {"id": 1, "query": "What is the capital of France?", "category": "factual"},
        {"id": 2, "query": "Write a Python function to reverse a string", "category": "coding"},
        {"id": 3, "query": "Explain quantum entanglement in simple terms", "category": "science"},
    ]

print(f"Running RouteLLM sw_ranking on {len(queries)} queries...")
print(f"Model pair: {client.strong_model} (strong) vs {client.weak_model} (weak)")
print()

results = []
for i, item in enumerate(queries):
    q = item["query"] if isinstance(item, dict) else item
    try:
        # Get routing decision
        # The router will decide whether to use strong or weak model
        response = client.chat.completions.create(
            model="router-sw_ranking-0.5",  # 50% threshold
            messages=[{"role": "user", "content": q}]
        )
        
        # Check which model was used
        # RouteLLM adds model info to response
        model_used = response.model
        is_strong = "gpt" in model_used.lower() or "strong" in model_used.lower()
        
        results.append({
            "query_id": item.get("id", i+1) if isinstance(item, dict) else i+1,
            "query": q,
            "model_used": model_used,
            "routed_to": "strong" if is_strong else "weak",
            "response": response.choices[0].message.content[:100] if response.choices[0].message.content else ""
        })
        print(f"Query {i+1}: routed to {model_used}")
        
    except Exception as e:
        print(f"Error on query {i+1}: {e}")
        results.append({
            "query_id": i+1,
            "query": q,
            "error": str(e)
        })

# Save results
output_file = "/home/yliulupo/projects/routesmith/report/routellm_smart_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_file}")

# Calculate routing distribution
strong_count = sum(1 for r in results if r.get("routed_to") == "strong")
weak_count = sum(1 for r in results if r.get("routed_to") == "weak")
print(f"\nRouting distribution: {strong_count} strong, {weak_count} weak")
