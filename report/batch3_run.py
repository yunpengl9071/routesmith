#!/usr/bin/env python3
"""
Batch 3: RouteSmith Scale-Up
15 customer support queries (features, integrations, security)
3 models: nemotron, phi-4, gpt-4o-mini
"""

import os
import json
import time
import asyncio
import aiohttp

API_KEY = os.environ.get("OPENROUTER_API_KEY", "REDACTED_OPENROUTER_KEY_2")

MODELS = {
    "nvidia/nemotron-3-nano-30b-a3b:free": "nvidia/nemotron-3-nano-30b-a3b:free",
    "microsoft/phi-4": "microsoft/phi-4",
    "openai/gpt-4o-mini": "openai/gpt-4o-mini"
}

# 15 new customer support queries focused on features, integrations, security
QUERIES = [
    # Features (5)
    "What advanced analytics features are available in the Enterprise plan?",
    "Does your platform support real-time collaboration features?",
    "Can I customize the workflow automation triggers?",
    "What reporting features come with the Team plan?",
    "Does your API support batch processing for bulk operations?",

    # Integrations (5)
    "How do I integrate RouteSmith with our existing Salesforce CRM?",
    "Can I connect your platform with Zapier for automated workflows?",
    "What webhooks are available for third-party integrations?",
    "Do you support GraphQL API for custom integrations?",
    "How do I set up a custom SSO integration with Okta?",

    # Security (5)
    "What security certifications does your platform have?",
    "How is customer data encrypted at rest and in transit?",
    "Can you provide SOC 2 compliance documentation?",
    "What SSO authentication methods do you support?",
    "How do you handle data retention and deletion requests?"
]

MODEL_IDS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "microsoft/phi-4",
    "openai/gpt-4o-mini"
]

async def call_model(session, model_id, query):
    """Call OpenRouter API"""
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
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data = await resp.json()
            latency = time.time() - start
            
            if resp.status == 200 and "choices" in data:
                response_text = data["choices"][0]["message"]["content"]
                # Extract cost from response usage
                cost = 0
                if "usage" in data:
                    # Estimate cost based on model pricing
                    prompt_tokens = data["usage"].get("prompt_tokens", 0)
                    completion_tokens = data["usage"].get("completion_tokens", 0)
                    # Rough cost estimates
                    if "nemotron" in model_id:
                        cost = (prompt_tokens * 0.0000001 + completion_tokens * 0.0000002)
                    elif "phi-4" in model_id:
                        cost = (prompt_tokens * 0.0000004 + completion_tokens * 0.0000016)
                    elif "gpt-4o-mini" in model_id:
                        cost = (prompt_tokens * 0.00000015 + completion_tokens * 0.0000006)
                
                return {
                    "success": True,
                    "response": response_text,
                    "cost": round(cost, 6),
                    "latency": round(latency, 2)
                }
            else:
                return {
                    "success": False,
                    "error": data.get("error", str(data)),
                    "cost": 0,
                    "latency": round(latency, 2)
                }
    except Exception as e:
        latency = time.time() - start
        return {
            "success": False,
            "error": str(e),
            "cost": 0,
            "latency": round(latency, 2)
        }

async def main():
    results = []
    
    async with aiohttp.ClientSession() as session:
        for query in QUERIES:
            print(f"\nQuery: {query[:60]}...")
            for model_id in MODEL_IDS:
                model_name = model_id.split("/")[-1].replace(":free", "")
                print(f"  Calling {model_name}...")
                result = await call_model(session, model_id, query)
                
                results.append({
                    "query": query,
                    "model": model_id,
                    "response": result.get("response", result.get("error", "No response")),
                    "cost": result.get("cost", 0)
                })
                
                print(f"    Cost: ${result.get('cost', 0):.6f}")
                
                # Small delay between calls
                await asyncio.sleep(0.5)
    
    # Save results
    output_path = "/home/yliulupo/projects/routesmith/report/batch3_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    total_cost = sum(r["cost"] for r in results)
    print(f"\n✅ Complete! Saved {len(results)} results to {output_path}")
    print(f"Total cost: ${total_cost:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
