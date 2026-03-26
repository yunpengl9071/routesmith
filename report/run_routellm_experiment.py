#!/usr/bin/env python3
"""
RouteLLM baseline experiment for RouteSmith comparison.
"""
import os
import json
import time
from datetime import datetime

# Set the OpenRouter API key
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from routellm.controller import Controller

# Model prices (per 1M tokens) from OpenRouter
MODEL_PRICES = {
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},  # $2.50/1M input, $10.00/1M output
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},  # $0.15/1M input, $0.60/1M output
}

def calculate_cost(input_tokens, output_tokens, model):
    """Calculate cost in USD."""
    prices = MODEL_PRICES.get(model, {"input": 0, "output": 0})
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost

def run_routellm_experiment(queries, router_name, threshold=0.5, max_queries=20):
    """Run RouteLLM experiment with specified router."""
    print(f"\n=== Running RouteLLM with {router_name} router (threshold={threshold}) ===")
    
    # Initialize controller
    try:
        controller = Controller(
            routers=[router_name],
            strong_model="openai/gpt-4o",
            weak_model="openai/gpt-4o-mini",
            progress_bar=True
        )
    except Exception as e:
        print(f"Error initializing controller: {e}")
        return None
    
    results = []
    total_cost = 0
    strong_count = 0
    weak_count = 0
    
    for i, q in enumerate(queries[:max_queries]):
        query_text = q["query"]
        
        try:
            # Route the query
            start_time = time.time()
            
            response = controller.chat.completions.create(
                model=f"router-{router_name}-{threshold}",
                messages=[{"role": "user", "content": query_text}],
                temperature=0,
                max_tokens=500
            )
            
            elapsed = time.time() - start_time
            
            # Get usage info
            usage = response.usage
            input_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            output_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            
            # Determine which model was used
            model_used = response.model
            
            # Calculate cost
            cost = calculate_cost(input_tokens, output_tokens, model_used)
            total_cost += cost
            
            if "gpt-4o-mini" in model_used:
                weak_count += 1
            else:
                strong_count += 1
            
            results.append({
                "query_id": q.get("query_id", i+1),
                "query": query_text,
                "model_used": model_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "elapsed_time": elapsed,
                "response": response.choices[0].message.content[:200]
            })
            
            print(f"  [{i+1}] {query_text[:40]}... -> {model_used} (${cost:.4f})")
            
        except Exception as e:
            print(f"  Error on query {i+1}: {e}")
            results.append({
                "query_id": q.get("query_id", i+1),
                "query": query_text,
                "error": str(e)
            })
    
    return {
        "router": router_name,
        "threshold": threshold,
        "total_queries": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "total_cost": total_cost,
        "strong_model_count": strong_count,
        "weak_model_count": weak_count,
        "results": results
    }

def run_cheapest_first_baseline(queries, max_queries=20):
    """Simple baseline: always use cheapest model (gpt-4o-mini)."""
    print(f"\n=== Running Cheapest-First Baseline (always gpt-4o-mini) ===")
    
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
                max_tokens=500
            )
            
            usage = response.usage
            input_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            output_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            
            cost = calculate_cost(input_tokens, output_tokens, "openai/gpt-4o-mini")
            total_cost += cost
            
            results.append({
                "query_id": q.get("query_id", i+1),
                "query": query_text,
                "model_used": "openai/gpt-4o-mini",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "response": response.choices[0].message.content[:200]
            })
            
            print(f"  [{i+1}] {query_text[:40]}... -> gpt-4o-mini (${cost:.4f})")
            
        except Exception as e:
            print(f"  Error on query {i+1}: {e}")
            results.append({
                "query_id": q.get("query_id", i+1),
                "query": query_text,
                "error": str(e)
            })
    
    return {
        "router": "cheapest-first",
        "total_queries": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "total_cost": total_cost,
        "strong_model_count": 0,
        "weak_model_count": len(results),
        "results": results
    }

def run_all_strong_baseline(queries, max_queries=20):
    """Baseline: always use strongest model (gpt-4o)."""
    print(f"\n=== Running All-Strong Baseline (always gpt-4o) ===")
    
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
                max_tokens=500
            )
            
            usage = response.usage
            input_tokens = usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0
            output_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
            
            cost = calculate_cost(input_tokens, output_tokens, "openai/gpt-4o")
            total_cost += cost
            
            results.append({
                "query_id": q.get("query_id", i+1),
                "query": query_text,
                "model_used": "openai/gpt-4o",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "response": response.choices[0].message.content[:200]
            })
            
            print(f"  [{i+1}] {query_text[:40]}... -> gpt-4o (${cost:.4f})")
            
        except Exception as e:
            print(f"  Error on query {i+1}: {e}")
            results.append({
                "query_id": q.get("query_id", i+1),
                "query": query_text,
                "error": str(e)
            })
    
    return {
        "router": "all-strong",
        "total_queries": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "total_cost": total_cost,
        "strong_model_count": len(results),
        "weak_model_count": 0,
        "results": results
    }

def main():
    # Load queries
    with open('report/real_50_queries/results.json', 'r') as f:
        queries = json.load(f)
    
    print(f"Loaded {len(queries)} queries")
    
    # Run experiments
    experiments = []
    
    # 1. Cheapest-first baseline
    exp1 = run_cheapest_first_baseline(queries, max_queries=15)
    experiments.append(exp1)
    
    # 2. All-strong baseline
    exp2 = run_all_strong_baseline(queries, max_queries=15)
    experiments.append(exp2)
    
    # 3. RouteLLM with random router
    exp3 = run_routellm_experiment(queries, "random", threshold=0.5, max_queries=15)
    if exp3:
        experiments.append(exp3)
    
    # Calculate savings vs all-strong
    all_strong_cost = experiments[1]["total_cost"] if len(experiments) > 1 else 0
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for exp in experiments:
        if exp is None:
            continue
        savings = 0
        if all_strong_cost > 0:
            savings = ((all_strong_cost - exp["total_cost"]) / all_strong_cost) * 100
        
        print(f"\n{exp['router']}:")
        print(f"  Total Cost: ${exp['total_cost']:.4f}")
        print(f"  Savings vs All-Strong: {savings:.1f}%")
        print(f"  Strong model usage: {exp['strong_model_count']}/{exp['total_queries']}")
        print(f"  Weak model usage: {exp['weak_model_count']}/{exp['total_queries']}")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiments": experiments,
        "comparison": {
            "all_strong_cost": all_strong_cost,
            "cheapest_first_cost": experiments[0]["total_cost"] if experiments else 0,
            "routellm_random_cost": experiments[2]["total_cost"] if len(experiments) > 2 and experiments[2] else 0,
            "routeSmith_baseline": {
                "cost_savings": "67.2%",
                "quality_retention": "100%"
            }
        }
    }
    
    with open('report/routellm_baseline_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to report/routellm_baseline_results.json")

if __name__ == "__main__":
    main()
