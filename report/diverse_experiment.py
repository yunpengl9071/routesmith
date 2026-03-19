#!/usr/bin/env python3
"""
Diverse Model Routing Experiment
Tests 5 models across price-performance spectrum, tracks costs/quality/latency
"""

import os
import json
import time
import asyncio
import aiohttp
from datetime import datetime

API_KEY = "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"

# Verified model IDs from OpenRouter
MODELS = {
    "nemotron": "nvidia/nemotron-3-nano-30b-a3b:free",       # Free
    "phi4": "microsoft/phi-4",                               # Budget ~$0.06/1M
    "gpt4o_mini": "openai/gpt-4o-mini",                      # Standard ~$0.15/1M  
    "deepseek": "deepseek/deepseek-v3.2",                    # Standard ~$0.27/1M
    "gpt4o": "openai/gpt-4o",                                # Premium ~$2.50/1M
}

# Test queries - varied complexity
QUERIES = [
    "What is 15 + 27?",
    "Explain photosynthesis in one sentence.",
    "What are the main differences between Python and JavaScript?",
    "Write a short email apologizing for a delayed delivery.",
    "What is the capital of Australia?",
    "Explain quantum computing to a 10-year-old.",
    "What are 3 tips for better sleep?",
    "Convert 50 miles to kilometers.",
    "What is the meaning of the word 'ubiquitous'?",
    "List 5 healthy breakfast options.",
]

async def call_model(session, model_id, query):
    """Call OpenRouter API and return response with metadata"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; RouteSmith/1.0)",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 200,
    }
    
    start = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            latency = time.time() - start
            data = await resp.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                response = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                # Cost is nested in usage.cost
                cost = usage.get("cost", 0)
                return {
                    "success": True,
                    "response": response[:300],  # Truncate for storage
                    "latency": round(latency, 2),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cost": cost,
                }
            else:
                return {"success": False, "error": data.get("error", data), "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e), "latency": time.time() - start}

async def judge_quality(response, query):
    """Simple LLM-as-judge: rate response quality 1-10"""
    # This is a placeholder - in production, use another LLM to judge
    # For now, we'll do heuristic scoring
    score = 5  # Base score
    
    if response.get("success"):
        resp_text = response.get("response", "").lower()
        query_lower = query.lower()
        
        # Check for relevant content
        if any(q in resp_text for q in query_lower.split()[:3]):
            score += 2
        
        # Check response length (too short = low quality)
        if len(response.get("response", "")) > 50:
            score += 1
        if len(response.get("response", "")) > 100:
            score += 1
            
        # Penalize errors
        if "error" in resp_text:
            score -= 3
            
        # Cap at 1-10
        return max(1, min(10, score))
    return 1

async def run_experiment():
    """Run the full experiment"""
    results = {model: {"queries": [], "total_cost": 0, "total_latency": 0, "quality_scores": []} 
               for model in MODELS}
    
    async with aiohttp.ClientSession() as session:
        for query in QUERIES:
            print(f"\nQuery: {query[:50]}...")
            
            for model_name, model_id in MODELS.items():
                print(f"  Testing {model_name}...", end=" ", flush=True)
                result = await call_model(session, model_id, query)
                
                if result["success"]:
                    quality = await judge_quality(result, query)
                    result["quality"] = quality
                    results[model_name]["queries"].append({"query": query, "result": result})
                    results[model_name]["total_cost"] += result["cost"]
                    results[model_name]["total_latency"] += result["latency"]
                    results[model_name]["quality_scores"].append(quality)
                    print(f"✓ cost=${result['cost']:.4f}, latency={result['latency']}s, quality={quality}/10")
                else:
                    err = str(result.get("error", "failed"))[:30]
                    print(f"✗ {err}")
                    results[model_name]["queries"].append({"query": query, "result": result})
                    results[model_name]["quality_scores"].append(1)
        
    # Calculate statistics
    summary = []
    for model_name, data in results.items():
        avg_latency = data["total_latency"] / len(QUERIES) if QUERIES else 0
        avg_quality = sum(data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0
        summary.append({
            "model": model_name,
            "model_id": MODELS[model_name],
            "total_cost": round(data["total_cost"], 4),
            "avg_latency": round(avg_latency, 2),
            "avg_quality": round(avg_quality, 2),
            "queries": len(QUERIES),
        })
    
    return summary, results

async def main():
    print("=" * 60)
    print("DIVERSE MODEL ROUTING EXPERIMENT")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    summary, raw = await run_experiment()
    
    # Calculate baselines
    total_queries = len(QUERIES)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for s in summary:
        print(f"\n{s['model']}:")
        print(f"  Model ID: {s['model_id']}")
        print(f"  Total Cost: ${s['total_cost']:.4f}")
        print(f"  Avg Latency: {s['avg_latency']}s")
        print(f"  Avg Quality: {s['avg_quality']}/10")
    
    # Calculate baselines
    print("\n" + "=" * 60)
    print("BASELINE COMPARISONS (if all queries went to ONE model)")
    print("=" * 60)
    
    for s in summary:
        baseline_cost = s['total_cost']
        quality = s['avg_quality']
        print(f"{s['model']:15} Cost: ${baseline_cost:.4f}, Quality: {quality}/10")
    
    # Simulate random routing
    import random
    random.seed(42)
    simulated_routing_cost = 0
    simulated_routing_quality = 0
    
    for _ in range(100):  # 100 simulations
        for query in QUERIES:
            model_choice = random.choice(list(MODELS.keys()))
            s = next(x for x in summary if x['model'] == model_choice)
            simulated_routing_cost += s['total_cost'] / total_queries
            simulated_routing_quality += s['avg_quality'] / total_queries
    
    print(f"\nRandom Routing (100 sims): ${simulated_routing_cost:.4f}, Quality: {simulated_routing_quality:.2f}/10")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "queries_tested": total_queries,
        "summary": summary,
        "simulated_routing": {
            "cost": round(simulated_routing_cost, 4),
            "quality": round(simulated_routing_quality, 2)
        }
    }
    
    with open("/home/yliulupo/projects/routesmith/report/experiment_raw.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nRaw results saved to experiment_raw.json")
    return summary, output

if __name__ == "__main__":
    asyncio.run(main())
