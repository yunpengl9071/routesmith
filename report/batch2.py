#!/usr/bin/env python3
"""
Batch 2: RouteSmith Scale-Up (Customer Support Queries 16-30)
Tests account, subscription, and technical support queries across 3 models
"""

import json
import time
import asyncio
import aiohttp

API_KEY = "REDACTED_OPENROUTER_KEY_2"

# Test models
MODELS = {
    "nemotron": "nvidia/nemotron-3-nano-30b-a3b:free",
    "phi4": "microsoft/phi-4",
    "gpt4o_mini": "openai/gpt-4o-mini",
}

# Customer support queries (different from batch 1)
# Mix of account, subscription, and technical issues
QUERIES = [
    # Account issues (5)
    "I can't log into my account - it says my password is incorrect but I'm sure it's right. How can I reset it?",
    "I need to update my email address on my account. Where do I do that?",
    "My account was locked after too many failed login attempts. How do I unlock it?",
    "I accidentally created two accounts with the same email. Can you merge them?",
    "I want to delete my account permanently. What is the process?",
    
    # Subscription issues (5)
    "I was charged twice for my monthly subscription. Can I get a refund?",
    "How do I upgrade from the free plan to the premium plan?",
    "I cancelled my subscription but I'm still being charged. What's happening?",
    "My subscription renewal failed. What should I do to avoid service interruption?",
    "Can I get a pro-rated refund if I downgrade my subscription mid-cycle?",
    
    # Technical issues (5)
    "The mobile app keeps crashing every time I try to upload a file. It's happening on both iPhone and Android.",
    "I'm getting a 500 error when I try to generate a report. This was working yesterday.",
    "The API is returning timeout errors for all my requests. Is there a known issue?",
    "My dashboard is showing stale data - it's not reflecting the changes I made an hour ago.",
    "The export feature is only downloading partial data. It used to work fine with the same settings.",
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
        "max_tokens": 300,
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
                    "response": response,
                    "latency": round(latency, 2),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cost": cost,
                }
            else:
                return {"success": False, "error": str(data), "latency": latency}
    except Exception as e:
        return {"success": False, "error": str(e), "latency": time.time() - start}

async def run_experiment():
    """Run the full experiment"""
    all_results = []
    
    connector = aiohttp.TCPConnector(limit=3)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, query in enumerate(QUERIES):
            print(f"\n[{i+1}/15] Query: {query[:60]}...")
            
            for model_name, model_id in MODELS.items():
                print(f"  -> {model_name}...", end=" ", flush=True)
                result = await call_model(session, model_id, query)
                
                entry = {
                    "query": query,
                    "model": model_name,
                    "response": result.get("response", result.get("error", "")),
                    "cost": result.get("cost", 0),
                    "latency": result.get("latency", 0),
                    "success": result.get("success", False)
                }
                all_results.append(entry)
                
                if result["success"]:
                    print(f"${result['cost']:.6f}, {result['latency']}s")
                else:
                    print(f"ERR: {result.get('error', 'failed')[:40]}")
    
    return all_results

async def main():
    print("=" * 60)
    print("BATCH 2: CUSTOMER SUPPORT QUERIES")
    print("=" * 60)
    
    results = await run_experiment()
    
    # Save to JSON
    output_path = "/home/yliulupo/projects/routesmith/report/batch2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_name in MODELS.keys():
        model_results = [r for r in results if r["model"] == model_name]
        total_cost = sum(r["cost"] for r in model_results)
        success_count = sum(1 for r in model_results if r["success"])
        print(f"{model_name}: {success_count}/15 successful, ${total_cost:.6f} total")
    
    print(f"\nResults saved to: {output_path}")
    return results

if __name__ == "__main__":
    asyncio.run(main())
