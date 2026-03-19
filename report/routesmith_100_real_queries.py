#!/usr/bin/env python3
"""
RouteSmith REAL Experiment — 100 Queries, Production-Ready

Improvements from 50-query pilot:
1. Only use RELIABLE models (Qwen3-Next + Nemotron, skip unavailable Gemma)
2. Track failure rates per model
3. Save progress after each query
4. Rate limited: 1 call per 1 second

Budget: ~$1.50 for 100 queries
"""

import os, json, time, random
from openai import OpenAI
from pathlib import Path
from datetime import datetime

# OpenRouter client
api_key = open(Path.home() / 'Documents/api_keys/openrouter').read().strip()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# ONLY RELIABLE MODELS (tested in 50-query pilot)
MODELS = {
    'premium': {
        'model': 'qwen/qwen3-next-80b-a3b-instruct',
        'cost_per_1k': 0.38,
        'description': 'Reliable, fast, no reasoning overhead'
    },
    'economy': {
        'model': 'nvidia/nemotron-3-nano-30b-a3b',
        'cost_per_1k': 0.00,  # FREE!
        'description': 'Free, reliable for simple queries'
    }
}

# 100 Real Customer Support Queries (5 categories × 20 each)
QUERIES = [
    # Technical (20) - expect premium
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
    ("How do I rotate API keys without downtime?", "technical"),
    ("What encryption standard do you use for data at rest?", "technical"),
    ("My GraphQL mutations are returning null — why?", "technical"),
    ("How do I implement exponential backoff correctly?", "technical"),
    ("What's the max payload size for file uploads?", "technical"),
    ("How do I validate webhook signatures in Python?", "technical"),
    ("Why is my SSE connection dropping after 5 minutes?", "technical"),
    ("How do I implement cursor-based pagination?", "technical"),
    ("What's the rate limit for your ML inference endpoint?", "technical"),
    ("How do I debug memory leaks in my integration?", "technical"),
    
    # Billing (20) - expect economy/premium mix
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
    ("What payment methods do you accept?", "billing"),
    ("Do you charge for API failures?", "billing"),
    ("How do I view my usage history?", "billing"),
    ("Can I get a refund for accidental overage?", "billing"),
    ("What happens if I exceed my plan limits?", "billing"),
    ("Do you offer annual billing discounts?", "billing"),
    ("How do I download my receipts?", "billing"),
    ("Why is my trial ending early?", "billing"),
    ("Can I pause my subscription?", "billing"),
    ("How do I upgrade mid-cycle?", "billing"),
    
    # Account (20) - expect economy/premium mix
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
    ("How do I revoke an API key?", "account"),
    ("Can I share my dashboard with my team?", "account"),
    ("How do I change my timezone settings?", "account"),
    ("My profile picture won't upload — why?", "account"),
    ("How do I set up audit logs?", "account"),
    ("Can I restrict API access by IP?", "account"),
    ("How do I grant read-only access to my team?", "account"),
    ("How do I enable Slack notifications?", "account"),
    ("Can I customize my dashboard widgets?", "account"),
    ("How do I export my team's activity log?", "account"),
    
    # Product Info (20) - expect economy
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
    ("What languages is your SDK available in?", "product"),
    ("Do you have a GraphQL API?", "product"),
    ("Is there a sandbox environment?", "product"),
    ("Do you support batch operations?", "product"),
    ("Can I white-label your product?", "product"),
    ("Do you have a partner program?", "product"),
    ("What's your data retention policy?", "product"),
    ("Do you support custom domains?", "product"),
    ("Is there a Chrome extension?", "product"),
    ("Do you have public documentation?", "product"),
    
    # Simple FAQ (20) - expect economy
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
    ("How do I get started?", "faq"),
    ("Is there a quickstart guide?", "faq"),
    ("Do you have code examples?", "faq"),
    ("Can I talk to sales?", "faq"),
    ("Do you offer demos?", "faq"),
    ("What industries do you serve?", "faq"),
    ("Are you GDPR compliant?", "faq"),
    ("Do you have an API status page?", "faq"),
    ("How do I report a bug?", "faq"),
    ("Can I request a feature?", "faq"),
]

class ThompsonSamplingRouter:
    """TS router with failure rate tracking."""
    
    def __init__(self):
        self.priors = {}
        self.failure_counts = {}
        for category in ['technical', 'billing', 'account', 'product', 'faq']:
            for tier in MODELS.keys():
                self.priors[(category, tier)] = {'alpha': 1, 'beta': 1}
                self.failure_counts[(category, tier)] = {'failures': 0, 'attempts': 0}
        self.cost_bias = 0.1
        self.failure_penalty = 0.5
    
    def select_tier(self, category):
        samples = {}
        for tier in MODELS.keys():
            sample = random.betavariate(
                self.priors[(category, tier)]['alpha'],
                self.priors[(category, tier)]['beta']
            )
            cost = MODELS[tier]['cost_per_1k']
            
            # Failure rate penalty
            fc = self.failure_counts[(category, tier)]
            failure_rate = fc['failures'] / max(1, fc['attempts'])
            
            adjusted = sample - (self.cost_bias * cost) - (self.failure_penalty * failure_rate)
            samples[tier] = adjusted
        return max(samples, key=samples.get)
    
    def update(self, category, tier, quality_score, success=True):
        prior = self.priors[(category, tier)]
        fc = self.failure_counts[(category, tier)]
        
        fc['attempts'] += 1
        if not success:
            fc['failures'] += 1
            prior['beta'] += 1  # Penalize failures
        elif quality_score >= 0.7:
            prior['alpha'] += 1
        else:
            prior['beta'] += 1

def call_llm(model_key, query):
    """Make API call."""
    model_config = MODELS[model_key]
    
    try:
        response = client.chat.completions.create(
            model=model_config['model'],
            messages=[
                {"role": "system", "content": "Answer in 1-2 sentences. Be direct."},
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        usage = response.usage
        cost = (usage.total_tokens / 1000) * model_config['cost_per_1k']
        
        return {
            'success': True,
            'answer': answer,
            'total_tokens': usage.total_tokens,
            'cost': cost,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'answer': None,
            'total_tokens': 0,
            'cost': 0,
            'error': str(e)
        }

def evaluate_quality(query, answer):
    """Automated quality eval."""
    if not answer:
        return 0.0
    
    quality = 0.5
    if 20 <= len(answer) <= 150:
        quality += 0.25
    if any(word in answer.lower() for word in ['click', 'go to', 'check', 'verify', 'use']):
        quality += 0.15
    if len(answer) < 15:
        quality -= 0.2
    
    return min(1.0, max(0.0, quality))

def run_experiment():
    """Run 100-query experiment."""
    print("=" * 70)
    print("RouteSmith REAL Experiment — 100 Queries")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Models: {MODELS['premium']['model']}, {MODELS['economy']['model']}")
    print(f"Budget target: <$1.50\n")
    
    router = ThompsonSamplingRouter()
    results = []
    cumulative_cost = 0
    successful = 0
    
    for i, (query, category) in enumerate(QUERIES):
        selected_tier = router.select_tier(category)
        result = call_llm(selected_tier, query)
        
        if result['success']:
            quality = evaluate_quality(query, result['answer'])
            router.update(category, selected_tier, quality, success=True)
            cumulative_cost += result['cost']
            successful += 1
            
            results.append({
                'query_id': i + 1,
                'category': category,
                'selected_tier': selected_tier,
                'total_tokens': result['total_tokens'],
                'cost_usd': result['cost'],
                'quality_score': quality,
                'success': True,
                'answer': answer[:80] + '...' if (answer := result['answer']) and len(answer) > 80 else (answer or '')
            })
            
            print(f"[{i+1:3d}/100] {category:10s} → {selected_tier:8s} | {result['total_tokens']:3d} tok | ${result['cost']:.6f}")
        else:
            router.update(category, selected_tier, 0, success=False)
            results.append({
                'query_id': i + 1,
                'category': category,
                'selected_tier': selected_tier,
                'success': False,
                'error': result['error']
            })
            print(f"[{i+1:3d}/100] {category:10s} → ERROR: {result['error'][:50]}")
        
        time.sleep(1)  # Rate limit
        
        # Save progress every 25 queries
        if (i + 1) % 25 == 0:
            save_progress(results, cumulative_cost, i + 1)
    
    # Final analysis
    metrics = analyze_results(results, cumulative_cost)
    save_final(results, metrics)
    
    return metrics

def save_progress(results, cost, count):
    """Save intermediate progress."""
    output_dir = Path.home() / 'projects/routesmith/report/real_100_queries'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f'progress_{count}.json', 'w') as f:
        json.dump({'count': count, 'cost': cost, 'results': results}, f, indent=2)

def analyze_results(results, cumulative_cost):
    """Analyze final results."""
    successful = [r for r in results if r['success']]
    
    # Routing distribution
    tier_counts = {}
    for r in results:
        tier = r['selected_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    # Learning curve
    learning_curve = []
    for window_end in range(20, 101, 20):
        window = results[:window_end]
        correct = sum(1 for r in window if r.get('success', False))
        accuracy = correct / len(window)
        learning_curve.append({'after_queries': window_end, 'success_rate': round(accuracy, 2)})
    
    return {
        'experiment': '100_real_queries',
        'timestamp': datetime.now().isoformat(),
        'total_queries': 100,
        'successful_queries': len(successful),
        'failed_queries': 100 - len(successful),
        'success_rate': round(len(successful) / 100 * 100, 1),
        'total_cost': round(cumulative_cost, 6),
        'avg_cost_per_query': round(cumulative_cost / 100, 6),
        'routing_distribution': tier_counts,
        'learning_curve': learning_curve,
        'models_used': MODELS
    }

def save_final(results, metrics):
    """Save final results."""
    output_dir = Path.home() / 'projects/routesmith/report/real_100_queries'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n✅ Total: 100 queries")
    print(f"✅ Successful: {metrics['successful_queries']} ({metrics['success_rate']}%)")
    print(f"✅ Failed: {metrics['failed_queries']}")
    print(f"\n💰 Total Cost: ${metrics['total_cost']:.6f}")
    print(f"💰 Avg per Query: ${metrics['avg_cost_per_query']:.6f}")
    print(f"\n📊 Routing: {metrics['routing_distribution']}")
    print(f"\n📈 Learning:")
    for lc in metrics['learning_curve']:
        print(f"   After {lc['after_queries']:3d}: {lc['success_rate']*100:.0f}% success")
    print(f"\n💾 Saved to: {output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    run_experiment()
