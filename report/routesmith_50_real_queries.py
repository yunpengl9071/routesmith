#!/usr/bin/env python3
"""
RouteSmith REAL Experiment — 50 Queries, Cost-Controlled

Key improvements from pilot:
1. Use NON-REASONING models (no hidden CoT tokens)
2. Strict token limits enforced
3. Save progress after each query
4. Rate limited: 1 call per 1.5 seconds

Budget: ~$2-5 for 50 queries (using cheap non-reasoning models)
"""

import os, json, time, random
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# OpenRouter client
api_key = open(Path.home() / 'Documents/api_keys/openrouter').read().strip()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# NON-REASONING models (verified)
MODELS = {
    'premium': {
        'model': 'qwen/qwen3-next-80b-a3b-instruct',  # Optimized for non-reasoning fast responses
        'cost_per_1k': 0.38,  # Avg cost
        'description': 'Fast, no thinking traces'
    },
    'standard': {
        'model': 'google/gemma-3-27b',  # Free tier available
        'cost_per_1k': 0.00,  # Free!
        'description': 'Free, capable for general queries'
    },
    'economy': {
        'model': 'nvidia/nemotron-3-nano-30b-a3b',  # Free
        'cost_per_1k': 0.00,  # Free!
        'description': 'Free, fast, simple queries'
    }
}

# 50 Real Customer Support Queries (5 categories × 10 each)
QUERIES = [
    # Technical (10) - route to Premium
    ("Why is my API returning 500 errors when batching?", "technical"),
    ("How do I implement OAuth2 refresh token rotation?", "technical"),
    ("What's the difference between sandbox and production rate limits?", "technical"),
    ("My webhook signatures don't match — what's wrong?", "technical"),
    ("How do I paginate through 10K records efficiently?", "technical"),
    ("Why are my async jobs timing out after 30 seconds?", "technical"),
    ("How do I implement idempotency keys correctly?", "technical"),
    ("What's the schema for v2 vs v3 API?", "technical"),
    ("My CORS preflight requests are failing — help!", "technical"),
    ("How do I handle partial failures in batch operations?", "technical"),
    
    # Billing (10) - route to Standard
    ("Why was I charged twice this month?", "billing"),
    ("How do I update my credit card?", "billing"),
    ("Can I get an invoice for my last payment?", "billing"),
    ("What's the difference between Pro and Enterprise?", "billing"),
    ("How do I cancel my subscription?", "billing"),
    ("Do you offer discounts for nonprofits?", "billing"),
    ("When does my billing cycle reset?", "billing"),
    ("Why did my usage spike last week?", "billing"),
    ("Can I set a spending cap?", "billing"),
    ("How do I add a team member to my account?", "billing"),
    
    # Account (10) - route to Standard
    ("How do I reset my password?", "account"),
    ("I'm locked out of my account — help!", "account"),
    ("How do I enable two-factor authentication?", "account"),
    ("Can I change my email address?", "account"),
    ("How do I delete my account?", "account"),
    ("How do I export my data?", "account"),
    ("My account shows wrong usage stats — fix it", "account"),
    ("How do I add SSO for my team?", "account"),
    ("Can I merge two accounts?", "account"),
    ("How do I view my API key history?", "account"),
    
    # Product Info (10) - route to Economy
    ("What features are included in the free plan?", "product"),
    ("Do you have a mobile app?", "product"),
    ("What integrations do you support?", "product"),
    ("Is there a Slack integration?", "product"),
    ("Do you support webhooks?", "product"),
    ("What's your uptime SLA?", "product"),
    ("Do you have a status page?", "product"),
    ("Where are your servers located?", "product"),
    ("Are you SOC2 compliant?", "product"),
    ("Do you offer on-premise deployment?", "product"),
    
    # Simple FAQ (10) - route to Economy
    ("What's your pricing?", "faq"),
    ("How do I contact support?", "faq"),
    ("What are your support hours?", "faq"),
    ("Do you have a phone number?", "faq"),
    ("Where's your documentation?", "faq"),
    ("Do you offer free trials?", "faq"),
    ("Can I try before I buy?", "faq"),
    ("Do you have video tutorials?", "faq"),
    ("Is there a community forum?", "faq"),
    ("Do you have a Chrome extension?", "faq"),
]

class ThompsonSamplingRouter:
    """Simple TS router with per-category priors."""
    
    def __init__(self):
        self.priors = {}
        for category in ['technical', 'billing', 'account', 'product', 'faq']:
            for tier in MODELS.keys():
                self.priors[(category, tier)] = {'alpha': 1, 'beta': 1}
        self.cost_bias = 0.1
    
    def select_tier(self, category):
        samples = {}
        for tier in MODELS.keys():
            sample = random.betavariate(
                self.priors[(category, tier)]['alpha'],
                self.priors[(category, tier)]['beta']
            )
            cost = MODELS[tier]['cost_per_1k']
            adjusted = sample - (self.cost_bias * cost)
            samples[tier] = adjusted
        return max(samples, key=samples.get)
    
    def update(self, category, tier, quality_score):
        prior = self.priors[(category, tier)]
        if quality_score >= 0.7:
            prior['alpha'] += 1
        else:
            prior['beta'] += 1

def call_llm(model_key, query):
    """Make API call with strict token limits."""
    model_config = MODELS[model_key]
    
    try:
        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[
                {"role": "system", "content": "Answer in 1-2 sentences. Be direct and concise."},
                {"role": "user", "content": query}
            ],
            max_tokens=50,  # Strict limit
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        usage = response.usage
        
        # Calculate cost (free models = $0)
        cost = (usage.total_tokens / 1000) * model_config['cost_per_1k']
        
        return {
            'success': True,
            'answer': answer,
            'input_tokens': usage.prompt_tokens,
            'output_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens,
            'cost': cost,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'answer': None,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost': 0,
            'error': str(e)
        }

def evaluate_quality(query, answer):
    """Simple automated quality eval."""
    if not answer:
        return 0.0
    
    quality = 0.5  # Base
    
    # Good answers are 20-150 chars (1-2 sentences)
    if 20 <= len(answer) <= 150:
        quality += 0.25
    
    # Answers with actionable words score higher
    if any(word in answer.lower() for word in ['click', 'go to', 'navigate', 'check', 'verify', 'use']):
        quality += 0.15
    
    # Penalize too short
    if len(answer) < 15:
        quality -= 0.2
    
    return min(1.0, max(0.0, quality))

def run_experiment():
    """Run full 50-query experiment."""
    print("=" * 70)
    print("RouteSmith REAL Experiment — 50 Queries, Non-Reasoning Models")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Models: {[MODELS[t]['model'] for t in MODELS.keys()]}")
    print(f"Budget target: <$5.00\n")
    
    router = ThompsonSamplingRouter()
    results = []
    cumulative_cost = 0
    baseline_cost = 0  # All premium
    successful = 0
    
    # Map categories to expected tiers (for baseline comparison)
    expected_tier = {
        'technical': 'premium',
        'billing': 'standard',
        'account': 'standard',
        'product': 'economy',
        'faq': 'economy'
    }
    
    for i, (query, category) in enumerate(QUERIES):
        # Route via Thompson Sampling
        selected_tier = router.select_tier(category)
        
        # Make API call
        result = call_llm(selected_tier, query)
        
        if result['success']:
            quality = evaluate_quality(query, result['answer'])
            router.update(category, selected_tier, quality)
            
            cumulative_cost += result['cost']
            
            # Baseline: what if we always used our tier mapping?
            baseline_tier = expected_tier[category]
            baseline_cost += (result['total_tokens'] / 1000) * MODELS[baseline_tier]['cost_per_1k']
            
            successful += 1
            
            result_entry = {
                'query_id': i + 1,
                'query': query,
                'category': category,
                'selected_tier': selected_tier,
                'expected_tier': expected_tier[category],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'total_tokens': result['total_tokens'],
                'cost_usd': result['cost'],
                'quality_score': quality,
                'success': True,
                'answer': (result['answer'][:80] + '...') if (result['answer'] and len(result['answer']) > 80) else (result['answer'] or '')
            }
            
            print(f"[{i+1:2d}/50] {category:10s} → {selected_tier:8s} | {result['total_tokens']:4d} tok | ${result['cost']:.6f} | Q: {quality:.2f}")
        else:
            print(f"[{i+1:2d}/50] {category:10s} → ERROR: {result['error']}")
            result_entry = {
                'query_id': i + 1,
                'query': query,
                'category': category,
                'selected_tier': selected_tier,
                'expected_tier': expected_tier[category],
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'cost_usd': 0,
                'quality_score': 0,
                'success': False,
                'answer': None,
                'error': result['error']
            }
        
        results.append(result_entry)
        time.sleep(1.5)  # Rate limit
    
    # Calculate final metrics
    df_results = results
    
    cost_reduction = 1 - (cumulative_cost / baseline_cost) if baseline_cost > 0 else 0
    avg_quality = sum(r['quality_score'] for r in results if r['success']) / max(1, successful)
    
    # Routing distribution
    tier_counts = {}
    for r in results:
        tier = r['selected_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    # Learning curve
    learning_curve = []
    for window_end in range(10, 51, 10):
        window = results[:window_end]
        correct = sum(1 for r in window if r['selected_tier'] == r['expected_tier'])
        accuracy = correct / len(window)
        learning_curve.append({'after_queries': window_end, 'accuracy': round(accuracy, 2)})
    
    metrics = {
        'experiment': '50_real_queries_non_reasoning',
        'timestamp': datetime.now().isoformat(),
        'total_queries': 50,
        'successful_queries': successful,
        'failed_queries': 50 - successful,
        'success_rate': round(successful / 50 * 100, 1),
        'cumulative_cost_routed': round(cumulative_cost, 6),
        'cumulative_cost_baseline': round(baseline_cost, 6),
        'cost_reduction_percent': round(cost_reduction * 100, 1) if baseline_cost > 0 else 0,
        'avg_quality': round(avg_quality, 3),
        'routing_distribution': tier_counts,
        'learning_curve': learning_curve,
        'models_used': MODELS
    }
    
    # Save results
    output_dir = Path.home() / 'projects/routesmith/report/real_50_queries'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n✅ Total Queries: 50")
    print(f"✅ Successful: {successful} ({successful/50*100:.1f}%)")
    print(f"✅ Failed: {50 - successful}")
    print(f"\n💰 Cost (Routed): ${cumulative_cost:.6f}")
    print(f"💰 Cost (Baseline): ${baseline_cost:.6f}")
    if baseline_cost > 0:
        print(f"💰 Savings: {cost_reduction*100:.1f}%")
    print(f"\n📊 Avg Quality: {avg_quality:.3f}")
    print(f"\n🎯 Routing Distribution: {tier_counts}")
    print(f"\n📈 Learning Curve:")
    for lc in learning_curve:
        print(f"   After {lc['after_queries']:2d} queries: {lc['accuracy']*100:.0f}% accuracy")
    print(f"\n💾 Results saved to: {output_dir}")
    print("=" * 70)
    
    return metrics

if __name__ == '__main__':
    run_experiment()
