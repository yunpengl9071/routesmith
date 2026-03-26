"""
ContextualTS: Adaptive Lambda Thompson Sampling for LLM Routing

This implementation uses adaptive lambda in Thompson Sampling based on 
query difficulty estimation.

Reference: Liu-Lupo (2026) - "Contextual Thompson Sampling: Adaptive Lambda for Efficient LLM Routing"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class ContextualTS:
    """
    Thompson Sampling with adaptive lambda based on query difficulty.
    
    For easy queries (small quality gap between tiers), use high lambda
    to favor cheap tiers. For hard queries, use low lambda to ensure quality.
    """
    
    def __init__(self, lambda_high: float = 50, lambda_low: float = 10, 
                 threshold: float = 0.10):
        """
        Args:
            lambda_high: Cost penalty for easy queries
            lambda_low: Cost penalty for hard queries  
            threshold: Quality gap threshold for easy/hard classification
        """
        self.lambda_high = lambda_high
        self.lambda_low = lambda_low
        self.threshold = threshold
        
        # Beta posteriors for each tier
        self.alpha = {"free": 1, "budget": 1, "premium": 1}
        self.beta = {"free": 1, "budget": 1, "premium": 1}
        
    def reset(self):
        """Reset posteriors to uniform."""
        self.alpha = {"free": 1, "budget": 1, "premium": 1}
        self.beta = {"free": 1, "budget": 1, "premium": 1}
        
    def select_tier(self, quality_estimates: Dict[str, float], 
                   costs: Dict[str, float]) -> str:
        """
        Select tier using Thompson Sampling with adaptive lambda.
        
        Args:
            quality_estimates: Dict of estimated quality per tier
            costs: Dict of cost per tier
            
        Returns:
            Selected tier name
        """
        # Calculate quality gap
        gap = quality_estimates.get("premium", 0) - quality_estimates.get("free", 0)
        
        # Set lambda based on difficulty
        if gap < self.threshold:
            lam = self.lambda_high  # Easy query - favor cost
        else:
            lam = self.lambda_low   # Hard query - favor quality
            
        # Sample from posteriors and compute expected reward
        scores = {}
        for tier in ["free", "budget", "premium"]:
            posterior_sample = np.random.beta(self.alpha[tier], self.beta[tier])
            reward = quality_estimates[tier] - lam * costs[tier]
            scores[tier] = posterior_sample * (1 + reward)
            
        # Select best tier
        selected = max(scores, key=scores.get)
        
        # Update posteriors based on observed quality (assumes we observe outcome)
        # This is simplified - real implementation would observe actual quality
        
        return selected
    
    def update_posterior(self, tier: str, quality: float):
        """Update posterior after observing quality for selected tier."""
        self.alpha[tier] += quality
        self.beta[tier] += (1 - quality)


def run_contextual_ts(quality_matrix: List[Dict], n_seeds: int = 5,
                      costs: Dict[str, float] = None) -> Dict:
    """
    Run ContextualTS on quality matrix.
    
    Args:
        quality_matrix: List of dicts with quality scores per tier
        n_seeds: Number of random seeds to run
        costs: Cost per tier (default: free=0, budget=0.002, premium=0.005)
        
    Returns:
        Results dict with quality, cost, Q/C ratio
    """
    if costs is None:
        costs = {"free": 0.0, "budget": 0.002, "premium": 0.005}
    
    results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        cts = ContextualTS()
        
        total_quality = 0
        total_cost = 0
        
        for query in quality_matrix:
            quality_est = {
                "free": query.get("free_correct", 0),
                "budget": query.get("budget_correct", 0), 
                "premium": query.get("premium_correct", 0)
            }
            tier = cts.select_tier(quality_est, costs)
            quality = quality_est[tier]
            cost = costs[tier]
            
            total_quality += quality
            total_cost += cost
            
            # Update posterior
            cts.update_posterior(tier, quality)
            
        qc = total_quality / total_cost if total_cost > 0 else float('inf')
        results.append({
            "quality": total_quality,
            "cost": total_cost,
            "qc_ratio": qc
        })
        
    # Compute statistics
    qualities = [r["quality"] for r in results]
    costs_list = [r["cost"] for r in results]
    qcs = [r["qc_ratio"] for r in results if r["qc_ratio"] != float('inf')]
    
    return {
        "mean_quality": np.mean(qualities),
        "std_quality": np.std(qualities),
        "mean_cost": np.mean(costs_list),
        "std_cost": np.std(costs_list),
        "mean_qc": np.mean(qcs) if qcs else float('inf'),
        "std_qc": np.std(qcs) if qcs else 0
    }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        results = run_contextual_ts(data["queries"])
        print(json.dumps(results, indent=2))
