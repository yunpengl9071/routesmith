#!/usr/bin/env python3
"""
Diverse Model Routing Experiment
Tests 5 models across price-performance spectrum, tracks costs/quality/latency
"""

import json
import time
import asyncio
import aiohttp
import random
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
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            latency = time.time() - start
            
            if resp.status != 200:
                text = await resp.text()
                return {"success": False, "error": text[:200], "latency": latency}
            
            data = await resp.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0]["message"]
                response = msg.get("content") or msg.get("refusal") or ""
                usage = data.get("usage", {})
                cost = usage.get("cost", 0)
                return {
                    "success": True,
                    "response": response[:300],
                    "latency": round(latency, 2),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cost": cost,
                }
            else:
                return {"success": False, "error": str(data), "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e), "latency": time.time() - start}

def judge_quality(response_text, query):
    """Simple heuristic quality judge"""
    score = 5
    if not response_text:
        return 1
    
    resp_lower = response_text.lower()
    query_words = query.lower().split()
    
    # Check for relevance
    if any(w in resp_lower for w in query_words[:3] if len(w) > 3):
        score += 2
    
    # Check length
    if len(response_text) > 50:
        score += 1
    if len(response_text) > 100:
        score += 1
    
    return max(1, min(10, score))

async def run_experiment():
    """Run the full experiment"""
    results = {model: {"queries": [], "total_cost": 0, "total_latency": 0, "quality_scores": []} 
               for model in MODELS}
    
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for query in QUERIES:
            print(f"\nQuery: {query[:50]}...")
            
            for model_name, model_id in MODELS.items():
                print(f"  {model_name}...", end=" ", flush=True)
                result = await call_model(session, model_id, query)
                
                if result["success"]:
                    quality = judge_quality(result.get("response", ""), query)
                    result["quality"] = quality
                    results[model_name]["queries"].append({"query": query, "result": result})
                    results[model_name]["total_cost"] += result["cost"]
                    results[model_name]["total_latency"] += result["latency"]
                    results[model_name]["quality_scores"].append(quality)
                    print(f"${result['cost']:.4f}, {result['latency']}s, q={quality}/10")
                else:
                    print(f"ERR: {result.get('error', 'failed')[:40]}")
                    results[model_name]["queries"].append({"query": query, "result": result})
                    results[model_name]["quality_scores"].append(1)
        
    return results

async def main():
    print("=" * 60)
    print("DIVERSE MODEL ROUTING EXPERIMENT")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    results = await run_experiment()
    total_queries = len(QUERIES)
    
    # Calculate summary
    summary = []
    for model_name, data in results.items():
        avg_latency = data["total_latency"] / total_queries if total_queries else 0
        avg_quality = sum(data["quality_scores"]) / len(data["quality_scores"]) if data["quality_scores"] else 0
        summary.append({
            "model": model_name,
            "model_id": MODELS[model_name],
            "total_cost": round(data["total_cost"], 6),
            "avg_latency": round(avg_latency, 2),
            "avg_quality": round(avg_quality, 2),
            "queries": total_queries,
        })
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for s in summary:
        print(f"\n{s['model']}:")
        print(f"  Model ID: {s['model_id']}")
        print(f"  Total Cost: ${s['total_cost']:.6f}")
        print(f"  Avg Latency: {s['avg_latency']}s")
        print(f"  Avg Quality: {s['avg_quality']}/10")
    
    # Baselines
    print("\n" + "=" * 60)
    print("BASELINE COMPARISONS (if ALL queries went to ONE model)")
    print("=" * 60)
    
    for s in summary:
        baseline_cost = s['total_cost']
        quality = s['avg_quality']
        print(f"{s['model']:15} Cost: ${baseline_cost:.6f}, Quality: {quality}/10")
    
    # Random routing simulation
    random.seed(42)
    total_runs = 100
    routing_costs = []
    routing_qualities = []
    
    for _ in range(total_runs):
        run_cost = 0
        run_quality = 0
        for query in QUERIES:
            model_choice = random.choice(list(MODELS.keys()))
            s = next(x for x in summary if x['model'] == model_choice)
            run_cost += s['total_cost'] / total_queries
            run_quality += s['avg_quality'] / total_queries
        routing_costs.append(run_cost)
        routing_qualities.append(run_quality)
    
    avg_routing_cost = sum(routing_costs) / len(routing_costs)
    avg_routing_quality = sum(routing_qualities) / len(routing_qualities)
    
    print(f"\nRandom Routing ({total_runs} sims avg): ${avg_routing_cost:.6f}, Quality: {avg_routing_quality:.2f}/10")
    
    # Output
    output = {
        "timestamp": datetime.now().isoformat(),
        "queries_tested": total_queries,
        "summary": summary,
        "simulated_routing": {
            "cost": round(avg_routing_cost, 6),
            "quality": round(avg_routing_quality, 2)
        }
    }
    
    with open("/home/yliulupo/projects/routesmith/report/experiment_raw.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nRaw results saved.")
    return summary, output

if __name__ == "__main__":
    asyncio.run(main())
