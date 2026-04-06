# benchmark/harness.py
"""Orchestrates running any strategy on any dataset."""
from __future__ import annotations

from pathlib import Path

from benchmark.config import RESULTS_DIR
from benchmark.strategies.base import BaseStrategy


def run_experiment(
    strategy: BaseStrategy,
    queries: list[dict],
    tag: str = "",
) -> tuple[list[dict], Path]:
    """
    Run strategy on queries. tag is appended to the filename for namespacing.
    Returns (results_list, results_path).
    """
    fname = strategy.name
    if tag:
        fname += f"_{tag}"
    fname += "_results.json"
    results_path = RESULTS_DIR / fname

    n = len(queries)
    print(f"\n=== {strategy.name} (tag={tag or 'none'}, {n} queries) ===")
    results = strategy.run(queries, results_path)

    acc = sum(r["correct"] for r in results) / len(results) if results else 0
    cost = sum(r["cost_usd"] for r in results)
    print(f"  DONE: n={len(results)}, acc={acc:.1%}, cost=${cost:.4f}")

    return results, results_path
