#!/usr/bin/env python3
"""Run ProductRouterStrategy on MMLU-600 and print APGR.

Usage:
    uv run python -m benchmark.run_product_bench

This script:
  1. Loads MMLU-600 queries (or 6 for dry-run with --dry).
  2. Runs ProductRouterStrategy via run_experiment.
  3. Loads baseline results (static-strong, static-weak) if available.
  4. Prints summary table with accuracy, PGR, cost, strong usage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark.config import MMLU_CACHE, RESULTS_DIR, STRONG_MODEL
from benchmark.dataset import load_mmlu_sample
from benchmark.harness import run_experiment
from benchmark.metrics import accuracy, performance_gap_recovery, strong_usage_pct, total_cost
from benchmark.strategies.product_router import ProductRouterStrategy


def load_results(fname: str) -> list[dict] | None:
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main(dry_run: bool = False) -> None:
    tag = "mmlu-600-dry" if dry_run else "mmlu-600"

    print("Loading MMLU-600...")
    mmlu = load_mmlu_sample(cache_path=str(MMLU_CACHE))
    queries = mmlu[:6] if dry_run else mmlu[:600]
    print(f"  {len(queries)} queries")

    # Run product router
    strategy = ProductRouterStrategy()
    results, path = run_experiment(strategy, queries, tag=tag)

    # Load baselines for PGR computation
    strong_r = load_results(f"static_strong_{tag}_results.json")
    weak_r = load_results(f"static_weak_{tag}_results.json")

    if not strong_r or not weak_r:
        print("\nNo baseline results found — skipping PGR computation.")
        print("Run `make exp1` first to generate static-strong and static-weak baselines.")
        return

    strong_acc = accuracy(strong_r)
    weak_acc = accuracy(weak_r)
    router_acc = accuracy(results)

    pgr = performance_gap_recovery(router_acc, weak_acc, strong_acc)
    cost = total_cost(results)
    spct = strong_usage_pct(results, STRONG_MODEL)

    print(f"\n{'Strategy':<45} {'Acc':>7} {'PGR':>7} {'Cost':>12} {'Strong%':>9}")
    print("-" * 84)
    print(f"{'Static-Strong (GPT-4o)':<45} {strong_acc:>6.1%} {'—':>7} ${'—':>11} {100.0:>8.1%}")
    print(f"{'Static-Weak (GPT-4o-mini)':<45} {weak_acc:>6.1%} {'—':>7} ${'—':>11} {0.0:>8.1%}")
    print(f"{strategy.name:<45} {router_acc:>6.1%} {pgr:>7.3f} ${cost:>11.4f} {spct:>8.1%}")


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    main(dry_run=dry)
