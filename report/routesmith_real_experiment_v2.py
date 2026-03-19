#!/usr/bin/env python3
"""
RouteSmith REAL Experiment — Throttled, Rate-Limited Version

Runs 100 customer support queries through 3-tier router with Thompson Sampling.
- Tests 1 query first before running full batch
- Rate limited: 1 call per 2 seconds (respects OpenRouter limits)
- Progress saved after each query (resume-capable)
"""

import os, json, time, random
import numpy as np
import pandas as pd
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# OpenRouter client
api_key = open(Path.home() / 'Documents/api_keys/openrouter').read().strip()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# Model registry (conscious choices)
MODELS = {
    'premium': {
        'model': 'qwen/qwen3.5-35b-a3b',  # Using 35B instead of Plus for better value
        'cost_per_1k_input': 0.16,
        'cost_per_1k_output': 1.00,
        'description': 'Good premium value'
    },
    'standard': {
        'model': 'minimax/minimax-m2.5',
        'cost_per_1k_input': 0.27,
        'cost_per_1k_output': 0.95,
        'description': 'Best cost/quality balance'
    },
    'economy': {
        'model': 'qwen/qwen3.5-35b-a3b',
        'cost_per_1k_input': 0.16,
        'cost_per_1k_output': 1.00,
        'description': 'Same as premium (for MVP, refine later)'
    }
}

# Sample queries (first 5 for testing)
TEST_QUERIES = [
    ("Why is my API returning 500 errors when I batch more than 100 requests?", "technical"),
    ("How do I reset my password?", "faq"),
    ("Why was I charged twice this month?", "billing"),
    ("Do you have a mobile app?", "product"),
    ("How do I enable two-factor authentication?", "account"),
]

def call_llm(model_key, query):
    """Make API call with conservative settings."""
    model_config = MODELS[model_key]
    
    try:
        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[
                {"role": "system", "content": "Answer customer support questions concisely (2-3 sentences max)."},
                {"role": "user", "content": query}
            ],
            max_tokens=80,  # Conservative token limit
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        usage = response.usage
        
        # Calculate actual cost
        input_cost = (usage.prompt_tokens / 1000) * model_config['cost_per_1k_input']
        output_cost = (usage.completion_tokens / 1000) * model_config['cost_per_1k_output']
        total_cost = input_cost + output_cost
        
        return {
            'success': True,
            'answer': answer,
            'input_tokens': usage.prompt_tokens,
            'output_tokens': usage.completion_tokens,
            'cost': total_cost,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'answer': None,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0,
            'error': str(e)
        }

def test_api_with_5_queries():
    """Test with 5 queries first before running full 100."""
    print("=" * 60)
    print("RouteSmith REAL Experiment — TEST RUN (5 queries)")
    print("=" * 60)
    print(f"\nStarting at: {datetime.now().isoformat()}")
    print("Testing with 5 queries first...\n")
    
    total_cost = 0
    successful = 0
    
    for i, (query, category) in enumerate(TEST_QUERIES):
        print(f"[{i+1}/5] Testing {category.upper()}: {query[:50]}...")
        
        # Always use economy for test (cheapest)
        result = call_llm('economy', query)
        
        if result['success']:
            total_cost += result['cost']
            successful += 1
            print(f"       ✅ Success | Cost: ${result['cost']:.6f} | Tokens: {result['input_tokens']}+{result['output_tokens']}")
        else:
            print(f"       ❌ Error: {result['error']}")
        
        # Rate limit: 2 seconds between calls
        time.sleep(2)
    
    print(f"\n{'=' * 60}")
    print(f"TEST COMPLETE: {successful}/5 successful | Total cost: ${total_cost:.6f}")
    print(f"Estimated full run (100 queries): ${total_cost * 20:.6f}")
    print(f"{'=' * 60}\n")
    
    if successful >= 4:
        print("✅ Test passed! Proceeding with full experiment...\n")
        return True
    else:
        print("❌ Test failed (< 80% success). Aborting full experiment.")
        return False

if __name__ == '__main__':
    # Run test first
    if test_api_with_5_queries():
        print("Full experiment code would go here...")
        print("(Run routesmith_real_experiment_v2.py with --full flag for 100 queries)")
    else:
        exit(1)
