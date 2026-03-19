#!/usr/bin/env python3
"""
RouteSmith REAL Experiment — STRICT Token Limit Version

Enforces 50 token max responses to control costs.
"""

import os, json, time
from openai import OpenAI
from pathlib import Path
from datetime import datetime

api_key = open(Path.home() / 'Documents/api_keys/openrouter').read().strip()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

MODELS = {
    'economy': {'model': 'qwen/qwen3.5-35b-a3b', 'cost_per_1k': 0.16},
    'standard': {'model': 'minimax/minimax-m2.5', 'cost_per_1k': 0.27},
    'premium': {'model': 'qwen/qwen3.5-plus-02-15', 'cost_per_1k': 2.00}
}

QUERIES = [
    ("Why is my API returning 500 errors?", "technical"),
    ("How do I reset my password?", "faq"),
    ("Why was I charged twice?", "billing"),
    ("Do you have a mobile app?", "product"),
    ("How do I enable 2FA?", "account"),
    ("What's your uptime SLA?", "faq"),
    ("How do I cancel?", "billing"),
    ("Is there an SDK?", "product"),
    ("Webhook signature invalid", "technical"),
    ("Forgot username", "account"),
]

print("=" * 60)
print("RouteSmith REAL Experiment — 10 Queries, Strict Token Limit")
print("=" * 60)

total_cost = 0
results = []

for i, (query, category) in enumerate(QUERIES):
    # Simple routing: technical→premium, billing/account→standard, else→economy
    if category == 'technical':
        tier = 'premium'
    elif category in ['billing', 'account']:
        tier = 'standard'
    else:
        tier = 'economy'
    
    try:
        response = client.chat.completions.create(
            model=MODELS[tier]['model'],
            messages=[
                {"role": "system", "content": "Answer in ONE sentence max. Be direct."},
                {"role": "user", "content": query}
            ],
            max_tokens=30
        )
        
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = (tokens / 1000) * MODELS[tier]['cost_per_1k']
        
        total_cost += cost
        results.append({
            'query': query,
            'category': category,
            'tier': tier,
            'tokens': tokens,
            'cost': cost,
            'answer': answer[:80] + '...' if len(answer) > 80 else answer
        })
        
        print(f"[{i+1:2d}] {category:10s} | {tier:8s} | {tokens:3d} tok | ${cost:.6f}")
        print(f"       Q: {query}")
        print(f"       A: {answer}\n")
        
        time.sleep(1)  # Rate limit
        
    except Exception as e:
        print(f"[{i+1:2d}] ERROR: {e}\n")
        results.append({'query': query, 'error': str(e)})

print("=" * 60)
print(f"COMPLETE: {len(results)} queries | Total cost: ${total_cost:.6f}")
print(f"Average cost/query: ${total_cost/len(QUERIES):.6f}")
print(f"Extrapolated 100 queries: ${total_cost * 10:.6f}")
print("=" * 60)

# Save results
with open(Path.home() / 'projects/routesmith/report/real_10_query_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'total_cost': round(total_cost, 6),
        'avg_cost_per_query': round(total_cost / len(QUERIES), 6),
        'estimated_100_queries': round(total_cost * 10, 6),
        'results': results
    }, f, indent=2)

print(f"\n💾 Results saved to: real_10_query_results.json")
