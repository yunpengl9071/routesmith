#!/usr/bin/env python3
"""
Simplified RouteLLM baseline experiment for RouteSmith comparison.
"""
import os
import json
import time
from datetime import datetime

# Set the OpenRouter API key
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Model prices (per 1M tokens) from OpenRouter
MODEL_PRICES = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

def calculate_cost(input_tokens, output_tokens, model):
    """Calculate cost in USD."""
    prices = MODEL_PRICES.get(model, {"input": 0, "output": 0})
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost

def run_cheapest_first(queries, max_queries=10):
    """Always use gpt-4o-mini"""
    from litellm import completion
    
    results = []
    total_cost = 0
    
    for i, q in enumerate(queries[:max_queries]):
        query_text = q["query"]
        try:
            response = completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": query_text}],
                temperature=0,
                max_tokens=300
            )
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            cost = calculate_cost(input_tokens, output_tokens, "openai/gpt-4o-mini")
            total_cost += cost
            results.append({"query": query_text, "model": "gpt-4o-mini", "cost": cost})
            print(f"  [{i+1}] cheapest: ${cost:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return {"router": "cheapest-first", "total_cost": total_cost, "results": results}

def run_all_strong(queries, max_queries=10):
    """Always use gpt-4o"""
    from litellm import completion
    
    results = []
    total_cost = 0
    
    for i, q in enumerate(queries[:max_queries]):
        query_text = q["query"]
        try:
            response = completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": query_text}],
                temperature=0,
                max_tokens=300
            )
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            cost = calculate_cost(input_tokens, output_tokens, "openai/gpt-4o")
            total_cost += cost
            results.append({"query": query_text, "model": "gpt-4o", "cost": cost})
            print(f"  [{i+1}] strong: ${cost:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return {"router": "all-strong", "total_cost": total_cost, "results": results}

def run_routellm_random(queries, max_queries=10):
    """Use RouteLLM random router"""
    from routellm.controller import Controller
    
    results = []
    total_cost = 0
    strong_count = 0
    weak_count = 0
    
    try:
        controller = Controller(
            routers=["random"],
            strong_model="openai/gpt-4o",
            weak_model="openai/gpt-4o-mini",
            progress_bar=False
        )
        
        for i, q in enumerate(queries[:max_queries]):
            query_text = q["query"]
            try:
                response = controller.chat.completions.create(
                    model="router-random-0.5",
                    messages=[{"role": "user", "content": query_text}],
                    temperature=0,
                    max_tokens=300
                )
                usage = response.usage
                input_tokens = getattr(usage, 'prompt_tokens', 0)
                output_tokens = getattr(usage, 'completion_tokens', 0)
                model_used = response.model
                cost = calculate_cost(input_tokens, output_tokens, model_used)
                total_cost += cost
                
                if "mini" in model_used:
                    weak_count += 1
                else:
                    strong_count += 1
                    
                results.append({"query": query_text, "model": model_used, "cost": cost})
                print(f"  [{i+1}] routed to {model_used}: ${cost:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
    except Exception as e:
        print(f"Controller error: {e}")
    
    return {"router": "routellm-random", "total_cost": total_cost, "strong_count": strong_count, "weak_count": weak_count, "results": results}

def main():
    # Load queries
    with open('report/real_50_queries/results.json', 'r') as f:
        queries = json.load(f)
    
    print(f"Loaded {len(queries)} queries")
    
    max_queries = 10  # Small number for quick test
    
    # Run experiments
    print("\n=== Cheapest-First (gpt-4o-mini) ===")
    exp1 = run_cheapest_first(queries, max_queries)
    
    print("\n=== All-Strong (gpt-4o) ===")
    exp2 = run_all_strong(queries, max_queries)
    
    print("\n=== RouteLLM Random ===")
    exp3 = run_routellm_random(queries, max_queries)
    
    # Summary
    all_strong_cost = exp2["total_cost"]
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Cheapest-First: ${exp1['total_cost']:.4f}")
    print(f"All-Strong: ${all_strong_cost:.4f}")
    print(f"RouteLLM Random: ${exp3['total_cost']:.4f}")
    
    if all_strong_cost > 0:
        savings_cheapest = ((all_strong_cost - exp1['total_cost']) / all_strong_cost) * 100
        savings_routellm = ((all_strong_cost - exp3['total_cost']) / all_strong_cost) * 100
        print(f"\nSavings vs All-Strong:")
        print(f"  Cheapest-First: {savings_cheapest:.1f}%")
        print(f"  RouteLLM Random: {savings_routellm:.1f}%")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiments": [exp1, exp2, exp3],
        "summary": {
            "cheapest_first_cost": exp1["total_cost"],
            "all_strong_cost": all_strong_cost,
            "routellm_random_cost": exp3["total_cost"],
            "routellm_strong_count": exp3.get("strong_count", 0),
            "routellm_weak_count": exp3.get("weak_count", 0),
            "routeSmith_reference": {
                "cost_savings": "67.2%",
                "quality_retention": "100%"
            }
        }
    }
    
    with open('report/routellm_baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to report/routellm_baseline_results.json")

if __name__ == "__main__":
    main()
