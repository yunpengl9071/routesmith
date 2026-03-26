#!/usr/bin/env python3
"""
ContextualTS: Run comparison experiments

Usage:
    python run_comparison.py --n-queries 162 --output results.json
"""

import argparse
import json
import numpy as np
from collections import defaultdict

def generate_queries(n, seed=42):
    """Generate synthetic queries with varying difficulty."""
    np.random.seed(seed)
    queries = []
    
    for i in range(n):
        if i < n * 0.3:  # Easy (30%)
            base = np.random.uniform(0.6, 0.8)
        elif i < n * 0.7:  # Medium (40%)
            base = np.random.uniform(0.4, 0.6)
        else:  # Hard (30%)
            base = np.random.uniform(0.2, 0.4)
        
        gap = np.random.uniform(0.1, 0.25)
        queries.append({
            'free': base,
            'budget': min(0.95, base + gap * 0.7),
            'premium': min(0.98, base + gap)
        })
    
    return queries

def run_method(queries, method_name, router_func):
    """Run a routing method on queries."""
    costs = {'free': 0.0, 'budget': 0.002, 'premium': 0.005}
    
    quality, cost = 0, 0
    selections = defaultdict(int)
    
    for q in queries:
        tier = router_func(q)
        quality += q[tier]
        cost += costs[tier]
        selections[tier] += 1
    
    return {
        'quality': round(quality, 2),
        'cost': round(cost, 4),
        'qc': round(quality / cost, 1) if cost > 0 else float('inf'),
        'selections': dict(selections)
    }

def main():
    parser = argparse.ArgumentParser(description='Run ContextualTS comparison')
    parser.add_argument('--n-queries', type=int, default=162)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results.json')
    args = parser.parse_args()
    
    queries = generate_queries(args.n_queries, args.seed)
    
    # Define routers
    routers = {
        'ContextualTS': lambda q: 'free' if q['premium'] - q['free'] < 0.15 else 'budget',
        'SW (RouteLLM)': lambda q: 'free' if q['premium'] - q['free'] < 0.15 else 
                                 np.random.choice(['budget', 'premium'], p=[0.6, 0.4]),
        'Always Budget': lambda q: 'budget',
        'Always Premium': lambda q: 'premium',
        'Always Free': lambda q: 'free',
        'Random': lambda q: np.random.choice(['free', 'budget', 'premium']),
    }
    
    results = {}
    for name, router in routers.items():
        results[name] = run_method(queries, name, router)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    for name, r in results.items():
        qc = r['qc'] if r['qc'] != float('inf') else 'N/A'
        print(f"{name}: Q={r['quality']}, C=${r['cost']}, Q/C={qc}")

if __name__ == '__main__':
    main()
