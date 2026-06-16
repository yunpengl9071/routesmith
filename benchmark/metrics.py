"""Metric computation for benchmark experiments."""
from __future__ import annotations

import statistics


def accuracy(results: list[dict]) -> float:
    """Fraction of queries answered correctly."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("correct", False)) / len(results)


def total_cost(results: list[dict]) -> float:
    """Total cost in USD across all queries."""
    return sum(r.get("cost_usd", 0.0) for r in results)


def performance_gap_recovery(
    router_acc: float, weak_acc: float, strong_acc: float
) -> float:
    """
    PGR = (router_acc - weak_acc) / (strong_acc - weak_acc)

    Measures fraction of quality gap between weak and strong that the router recovers.
    0.0 = same as weak model; 1.0 = same as strong model.
    Returns 0.0 when strong_acc == weak_acc (degenerate case).
    """
    gap = strong_acc - weak_acc
    if abs(gap) < 1e-9:
        return 0.0
    return (router_acc - weak_acc) / gap


def cumulative_regret(
    oracle_rewards: list[float], actual_rewards: list[float]
) -> list[float]:
    """
    Cumulative regret at each step t: sum_{i=1}^{t} (oracle_i - actual_i).
    Returns a list of length len(oracle_rewards).
    """
    regret = 0.0
    result = []
    for o, a in zip(oracle_rewards, actual_rewards):
        regret += o - a
        result.append(regret)
    return result


def convergence_query(
    rewards: list[float],
    oracle_reward: float,
    window: int = 100,
    threshold: float = 0.02,
) -> int | None:
    """
    First query t (1-indexed) where the rolling mean over the last `window` queries
    stays within `threshold` of oracle_reward. Returns None if never converged.
    """
    for t in range(window, len(rewards) + 1):
        window_mean = statistics.mean(rewards[t - window:t])
        if abs(window_mean - oracle_reward) <= threshold:
            return t
    return None


def strong_usage_pct(results: list[dict], strong_model: str) -> float:
    """Fraction of queries routed to the strong model."""
    if not results:
        return 0.0
    return sum(
        1 for r in results
        if r.get("final_model", r.get("model", "")) == strong_model
    ) / len(results)


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval."""
    import random
    means = [
        statistics.mean(random.choices(values, k=len(values)))
        for _ in range(n_bootstrap)
    ]
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap)
    return means[lo_idx], means[hi_idx]


def summarize(
    results: list[dict], name: str, strong_model: str = "openai/gpt-4o"
) -> dict:
    """Return a summary dict for one strategy's results."""
    n = len(results)
    return {
        "name": name,
        "n": n,
        "accuracy": accuracy(results),
        "total_cost_usd": total_cost(results),
        "cost_per_query": total_cost(results) / max(n, 1),
        "strong_pct": strong_usage_pct(results, strong_model),
    }
