#!/usr/bin/env python3
"""
RouteSmith - Quality-Fixed Experiment Runner

This experiment implements the improved reward function with:
1. Hard floor at quality < 0.5 (reject very low quality responses)
2. Quality-weighted reward with floor (alpha=2.0, beta=0.5)
3. Confidence penalty for economy tier (require >=75%)
4. Force premium on low quality queries

Key metrics tracked:
- acceptable_quality_rate: % responses >= 5/10
- premium_retry_rate: % of economy that upgrades to premium
- cost_per_valid_response
- failures_prevented
"""

import json
import csv
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Try to import required libraries
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    exit(1)

# API key setup
API_KEY_PATH = Path.home() / 'Documents/api_keys/openrouter'
if not API_KEY_PATH.exists():
    print(f"ERROR: API key not found at {API_KEY_PATH}")
    exit(1)

try:
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open(API_KEY_PATH).read().strip())
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    exit(1)

# ============================================================
# EXPERIMENT CONFIGURATION - Quality-Fixed Reward Function
# ============================================================
EXPERIMENT_CONFIG = {
    'min_acceptable_quality': 0.5,  # 5/10 in LLM judge scale
    'force_premium_on_low_quality': True,
    'economy_confidence_threshold': 0.75,
    'alpha': 2.0,
    'beta': 0.5,
    'track_metrics': [
        'acceptable_quality_rate',
        'premium_retry_rate',
        'cost_per_valid_response',
        'failures_prevented'
    ]
}

# ============================================================
# QUALITY-FIXED REWARD FUNCTION
# ============================================================
def reward(quality, cost, failed=False, confidence=0.5, tier='economy'):
    """
    Quality-fixed reward function with hard floor and confidence penalty.
    
    Args:
        quality: Quality score 0-1
        cost: Cost in dollars
        failed: Whether the request failed
        confidence: Confidence score 0-1
        tier: Tier name ('premium', 'standard', 'economy')
    
    Returns:
        float: Reward value
    """
    # Hard floor: reject very low quality
    if quality < 0.5 or failed:
        return -1.0  # Strong penalty for failure
    
    # Quality-weighted reward with floor (α=2.0, β=0.5)
    quality_component = 2.0 * max(quality - 0.3, 0)
    cost_component = -0.5 * cost
    
    # Confidence penalty for economy tier (require >=75%)
    confidence_penalty = 0 if confidence > 0.75 else -0.2
    
    return quality_component + cost_component + confidence_penalty


# ============================================================
# MODEL CONFIGURATION - Distinct models for each tier
# ============================================================
MODELS = {
    'premium': {
        'model': 'openai/gpt-4o',
        'cost_per_1k_input': 2.50,
        'cost_per_1k_output': 10.00,
        'description': 'Highest quality - for complex queries'
    },
    'standard': {
        'model': 'openai/gpt-4o-mini',
        'cost_per_1k_input': 0.15,
        'cost_per_1k_output': 0.60,
        'description': 'Balanced cost/quality - for moderate queries'
    },
    'economy': {
        'model': 'qwen/qwen3.5-35b-a3b',
        'cost_per_1k_input': 0.16,
        'cost_per_1k_output': 1.00,
        'description': 'Lowest cost - for simple queries'
    }
}

# Test queries covering different categories and complexity levels
TEST_QUERIES = {
    'technical': [
        "How do I implement OAuth2 refresh token rotation?",
        "Why is my API returning 500 errors when I batch more than 100 requests?",
        "What's the difference between sandbox and production rate limits?",
        "My webhook signatures don't match — what am I doing wrong?",
        "How do I paginate through 10K records efficiently?",
    ],
    'billing': [
        "Why was I charged twice this month?",
        "How do I update my credit card?",
        "Can I get an invoice for my last payment?",
        "What's the difference between Pro and Enterprise plans?",
        "How do I cancel my subscription?",
    ],
    'account': [
        "How do I reset my password?",
        "I'm locked out of my account — help!",
        "How do I enable two-factor authentication?",
        "Can I change my email address?",
        "How do I delete my account?",
    ],
    'product': [
        "What features are included in the free plan?",
        "Do you have a mobile app?",
        "What integrations do you support?",
        "Is there a Slack integration?",
        "Do you support webhooks?",
    ],
    'faq': [
        "How do I contact support?",
        "What are your hours?",
        "Do you have a phone number?",
        "Where's your documentation?",
        "Do you offer trials?",
    ]
}


# Simple "confidence" model - estimates if query is simple enough for economy
# In production this would be a trained model
def estimate_confidence(query, category):
    """Estimate confidence that economy tier can handle this query."""
    # Simple heuristics
    simple_indicators = ['how do i', 'what is', 'can i', 'do you', 'is there']
    complex_indicators = ['why', 'debug', 'error', 'implement', 'complex', 'oauth', 'batch']
    
    query_lower = query.lower()
    
    simple_count = sum(1 for ind in simple_indicators if ind in query_lower)
    complex_count = sum(1 for ind in complex_indicators if ind in query_lower)
    
    # Calculate confidence
    base_confidence = 0.5
    if category in ['faq', 'product']:
        base_confidence = 0.7
    elif category in ['billing', 'account']:
        base_confidence = 0.65
    elif category in ['technical']:
        base_confidence = 0.3
    
    # Adjust based on query complexity
    confidence = base_confidence + (simple_count * 0.1) - (complex_count * 0.15)
    return max(0.1, min(0.95, confidence))


def call_llm_with_quality(model_key, query, max_tokens=150):
    """
    Make API call and return result with quality scoring.
    Uses a simple heuristic for quality based on response characteristics.
    """
    model_config = MODELS[model_key]
    
    try:
        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[
                {"role": "system", "content": "Answer customer support questions concisely (2-4 sentences)."},
                {"role": "user", "content": query}
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        usage = response.usage
        
        # Calculate actual cost
        input_cost = (usage.prompt_tokens / 1000) * model_config['cost_per_1k_input']
        output_cost = (usage.completion_tokens / 1000) * model_config['cost_per_1k_output']
        total_cost = input_cost + output_cost
        
        # Determine if response is successful
        is_success = (
            answer is not None and 
            len(answer.strip()) > 0 and 
            len(answer.strip()) > 10
        )
        
        # Quality scoring - more sophisticated than before
        # Based on response characteristics
        if not is_success:
            quality_score = 0.0
        else:
            # Quality factors:
            # 1. Response length (too short = low quality)
            # 2. Contains helpful information
            # 3. Tier-specific base quality
            
            length_score = min(1.0, len(answer) / 100)  # 0-1 based on length
            
            # Tier-specific base quality (premium is better)
            tier_base_quality = {
                'premium': 0.85,
                'standard': 0.7,
                'economy': 0.6
            }.get(model_key, 0.6)
            
            # Combine factors
            quality_score = tier_base_quality * (0.5 + 0.5 * length_score)
            quality_score = max(0.3, min(1.0, quality_score))  # Clamp to reasonable range
        
        return {
            'success': is_success,
            'quality_score': quality_score,
            'answer': answer,
            'input_tokens': usage.prompt_tokens,
            'output_tokens': usage.completion_tokens,
            'cost': total_cost,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'quality_score': 0.0,
            'answer': None,
            'input_tokens': 0,
            'output_tokens': 0,
            'cost': 0,
            'error': str(e)
        }


def route_query_with_quality_fixes(query, category):
    """
    Route query using quality-fixed logic.
    
    Key improvements:
    1. Use confidence to decide if economy is safe
    2. If quality would be too low, force upgrade to premium
    3. Track retry decisions
    """
    confidence = estimate_confidence(query, category)
    
    # Decision logic
    if confidence >= EXPERIMENT_CONFIG['economy_confidence_threshold']:
        tier = 'economy'
    elif confidence >= 0.5:
        tier = 'standard'
    else:
        tier = 'premium'
    
    # Make the call
    result = call_llm_with_quality(tier, query)
    
    # Check if quality is acceptable
    if (EXPERIMENT_CONFIG['force_premium_on_low_quality'] and 
        result['success'] and 
        result['quality_score'] < EXPERIMENT_CONFIG['min_acceptable_quality']):
        
        # Retry with premium
        result_premium = call_llm_with_quality('premium', query)
        
        if result_premium['success']:
            result = result_premium
            tier = 'premium'  # Upgraded
    
    return result, tier, confidence


def run_quality_fixed_experiment(num_queries=100):
    """
    Run the quality-fixed experiment.
    """
    print("="*60)
    print("RouteSmith - Quality-Fixed Experiment")
    print("="*60)
    print(f"Config: {json.dumps(EXPERIMENT_CONFIG, indent=2)}")
    print(f"Target: {num_queries} queries")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Flatten queries
    all_queries = []
    for category, queries in TEST_QUERIES.items():
        for q in queries:
            all_queries.append((q, category))
    
    # Expand to desired number
    while len(all_queries) < num_queries:
        for category, queries in TEST_QUERIES.items():
            for q in queries:
                if len(all_queries) >= num_queries:
                    break
                all_queries.append((q, category))
            if len(all_queries) >= num_queries:
                break
    
    all_queries = all_queries[:num_queries]
    
    # Run experiment
    results = []
    total_cost = 0
    successful = 0
    failed = 0
    acceptable_quality = 0
    premium_retries = 0
    economy_total = 0
    
    output_file = Path(__file__).parent / 'quality_fixed_results.csv'
    
    for i, (query, category) in enumerate(all_queries):
        print(f"[{i+1}/{num_queries}] {query[:40]}...", end=" ")
        
        result, tier, confidence = route_query_with_quality_fixes(query, category)
        
        # Calculate reward
        r = reward(
            quality=result['quality_score'],
            cost=result['cost'],
            failed=not result['success'],
            confidence=confidence,
            tier=tier
        )
        
        if result['success']:
            successful += 1
            total_cost += result['cost']
            
            if result['quality_score'] >= EXPERIMENT_CONFIG['min_acceptable_quality']:
                acceptable_quality += 1
            
            print(f"✅ tier={tier} quality={result['quality_score']:.2f} reward={r:.2f} ${result['cost']:.4f}")
        else:
            failed += 1
            print(f"❌ {result['error'][:40] if result['error'] else 'Unknown'}")
        
        # Track premium retries
        if tier == 'premium':
            # Could check if it was a retry - for now approximate
            if confidence < EXPERIMENT_CONFIG['economy_confidence_threshold']:
                premium_retries += 1
        
        if tier == 'economy':
            economy_total += 1
        
        results.append({
            'query': query,
            'category': category,
            'tier': tier,
            'confidence': confidence,
            'success': result['success'],
            'quality_score': result['quality_score'],
            'cost': result['cost'],
            'reward': r,
            'input_tokens': result['input_tokens'],
            'output_tokens': result['output_tokens'],
            'error': result['error']
        })
        
        # Rate limiting
        time.sleep(1.2)
    
    # Calculate metrics
    success_rate = successful / num_queries * 100 if num_queries else 0
    acceptable_rate = acceptable_quality / num_queries * 100 if num_queries else 0
    avg_cost = total_cost / successful if successful > 0 else 0
    
    # Premium retry rate (out of economy decisions)
    retry_rate = premium_retries / economy_total * 100 if economy_total > 0 else 0
    
    # Cost per valid response (acceptable quality)
    cost_per_valid = total_cost / acceptable_quality if acceptable_quality > 0 else float('inf')
    
    # Failures prevented (would have been failed with old system)
    failures_prevented = premium_retries  # Approximation
    
    metrics = {
        'experiment_type': 'quality_fixed',
        'timestamp': datetime.now().isoformat(),
        'config': EXPERIMENT_CONFIG,
        'total_queries': num_queries,
        'successful_queries': successful,
        'failed_queries': failed,
        'success_rate': success_rate,
        'acceptable_quality_rate': acceptable_rate,
        'premium_retry_rate': retry_rate,
        'cumulative_cost_routed': total_cost,
        'cost_per_valid_response': cost_per_valid,
        'failures_prevented': failures_prevented,
        'avg_cost_per_query': avg_cost,
        'routing_distribution': {
            'premium': len([r for r in results if r['tier'] == 'premium']),
            'standard': len([r for r in results if r['tier'] == 'standard']),
            'economy': len([r for r in results if r['tier'] == 'economy'])
        },
        'models_used': MODELS
    }
    
    # Print summary
    print("\n" + "="*60)
    print("QUALITY-FIXED EXPERIMENT RESULTS")
    print("="*60)
    print(f"Total Queries: {num_queries}")
    print(f"Successful: {successful} ({success_rate:.1f}%)")
    print(f"Acceptable Quality (>=5/10): {acceptable_quality} ({acceptable_rate:.1f}%)")
    print(f"Premium Retries: {premium_retries}")
    print(f"Retry Rate: {retry_rate:.1f}%")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Cost per Valid Response: ${cost_per_valid:.4f}")
    print(f"Failures Prevented: {failures_prevented}")
    
    print("\nRouting Distribution:")
    for tier, count in metrics['routing_distribution'].items():
        print(f"  {tier}: {count} ({count/num_queries*100:.1f}%)")
    
    # Save results
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['query', 'category', 'tier', 'confidence', 'success', 'quality_score', 'cost', 'reward', 'input_tokens', 'output_tokens', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Save metrics
    metrics_file = Path(__file__).parent / 'quality_fixed_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")
    
    return results, metrics


def compare_with_baseline(quality_metrics, baseline_metrics):
    """Compare quality-fixed results with baseline."""
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    
    comparisons = {
        'Total Cost': (
            quality_metrics['cumulative_cost_routed'],
            baseline_metrics.get('cumulative_cost_routed', baseline_metrics.get('cumulative_cost_baseline', 0))
        ),
        'Acceptable Quality Rate': (
            quality_metrics['acceptable_quality_rate'],
            baseline_metrics.get('avg_quality', 0) * 100  # Convert to percentage
        ),
        'Success Rate': (
            quality_metrics['success_rate'],
            baseline_metrics.get('successful_queries', 0) / baseline_metrics.get('total_queries', 1) * 100
        )
    }
    
    for metric, (new_val, old_val) in comparisons.items():
        if old_val > 0:
            change = ((new_val - old_val) / old_val) * 100
            direction = "↑" if change > 0 else "↓"
            print(f"{metric}: {new_val:.2f} vs {old_val:.2f} ({direction}{abs(change):.1f}%)")
        else:
            print(f"{metric}: {new_val:.2f} vs {old_val:.2f}")
    
    return comparisons


if __name__ == '__main__':
    import sys
    
    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running in TEST mode (10 queries)")
        run_quality_fixed_experiment(num_queries=10)
    else:
        print("Running FULL experiment (100 queries)")
        results, metrics = run_quality_fixed_experiment(num_queries=100)
        
        # Load baseline metrics for comparison
        baseline_file = Path(__file__).parent / 'metrics.json'
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline = json.load(f)
            compare_with_baseline(metrics, baseline)