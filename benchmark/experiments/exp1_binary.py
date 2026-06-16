# benchmark/experiments/exp1_binary.py
"""Experiment 1: Binary routing comparison on MMLU + GSM8K.

Runs all routing strategies in phases on both datasets.
Each phase can run independently — results save incrementally.
"""
from __future__ import annotations

import json
from benchmark.config import (
    MMLU_CACHE, GSM8K_CACHE, RESULTS_DIR, SEEDS, STRONG_MODEL
)
from benchmark.dataset import load_mmlu_sample, load_gsm8k_sample
from benchmark.harness import run_experiment
from benchmark.metrics import accuracy, total_cost, performance_gap_recovery, strong_usage_pct
from benchmark.strategies.static import StaticStrongStrategy, StaticWeakStrategy, RandomRouterStrategy
from benchmark.strategies.routellm_sw import RouteLLMSWStrategy
from benchmark.strategies.ts_cat import TSCatStrategy
from benchmark.strategies.linucb import LinUCBStrategy
from benchmark.strategies.lints import LinTSStrategy

THRESHOLDS_SW = [0.3, 0.5, 0.7]

# Run phases in this order (cheapest to most expensive)
PHASE_ORDER = ["static", "routellm", "ts_cat", "linucb", "lints"]


def run_phase(phase: str, queries: list[dict], dataset_tag: str) -> None:
    """Run one phase of strategies on a dataset."""
    if phase == "static":
        run_experiment(StaticStrongStrategy(), queries, tag=dataset_tag)
        run_experiment(StaticWeakStrategy(), queries, tag=dataset_tag)
        run_experiment(RandomRouterStrategy(seed=42), queries, tag=dataset_tag)

    elif phase == "routellm":
        for t in THRESHOLDS_SW:
            run_experiment(RouteLLMSWStrategy(threshold=t), queries, tag=dataset_tag)

    elif phase == "ts_cat":
        for seed in SEEDS:
            run_experiment(TSCatStrategy(seed=seed), queries, tag=dataset_tag)

    elif phase == "linucb":
        for seed in SEEDS:
            run_experiment(LinUCBStrategy(alpha=1.5, seed=seed), queries, tag=dataset_tag)

    elif phase == "lints":
        for seed in SEEDS:
            run_experiment(LinTSStrategy(v_sq=1.0, seed=seed), queries, tag=dataset_tag)


def _load_results(fname: str) -> list[dict] | None:
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_summary(dataset_tag: str) -> None:
    """Print comparison table for all strategies on this dataset."""
    strong_r = _load_results(f"static_strong_{dataset_tag}_results.json")
    weak_r = _load_results(f"static_weak_{dataset_tag}_results.json")
    if not strong_r or not weak_r:
        print(f"No baselines yet for {dataset_tag}")
        return

    strong_acc = accuracy(strong_r)
    weak_acc = accuracy(weak_r)

    print(f"\n{'Strategy':<45} {'Acc':>7} {'PGR':>7} {'Cost':>12} {'Strong%':>9}")
    print("-" * 84)

    def show(name: str, results: list[dict]) -> None:
        if not results:
            return
        acc = accuracy(results)
        pgr = performance_gap_recovery(acc, weak_acc, strong_acc)
        cost = total_cost(results)
        spct = strong_usage_pct(results, STRONG_MODEL)
        print(f"{name:<45} {acc:>6.1%} {pgr:>7.3f} ${cost:>11.4f} {spct:>8.1%}")

    show("Static-Strong (GPT-4o)", strong_r)
    show("Static-Weak (GPT-4o-mini)", weak_r)

    for t in ["0.30", "0.50", "0.70"]:
        r = _load_results(f"routellm_sw_t{t}_{dataset_tag}_results.json")
        show(f"RouteLLM-SW (t={t})", r or [])

    for label, prefix in [("TS-Cat", "ts_cat"), ("LinUCB-27d", "linucb_27d_alpha1.5"), ("LinTS-27d", "lints_27d_vsq1.0")]:
        all_r = []
        for seed in SEEDS:
            r = _load_results(f"{prefix}_seed{seed}_{dataset_tag}_results.json")
            if r:
                all_r.extend(r)
        show(f"{label} (5 seeds avg)", all_r)


def main(phase: str | None = None) -> None:
    """Run experiments. If phase is given, run only that phase."""
    print("Loading datasets...")
    mmlu = load_mmlu_sample(cache_path=str(MMLU_CACHE))
    gsm8k = load_gsm8k_sample(cache_path=str(GSM8K_CACHE))
    print(f"MMLU: {len(mmlu)} queries, GSM8K: {len(gsm8k)} queries")

    phases = [phase] if phase else PHASE_ORDER

    for p in phases:
        print(f"\n{'='*60}\nPHASE: {p.upper()}\n{'='*60}")
        print(f"\n--- MMLU ({len(mmlu)} queries) ---")
        run_phase(p, mmlu, "mmlu")
        print(f"\n--- GSM8K ({len(gsm8k)} queries) ---")
        run_phase(p, gsm8k, "gsm8k")

    print("\n\n=== EXPERIMENT 1 SUMMARY ===")
    print("\n--- MMLU ---")
    print_summary("mmlu")
    print("\n--- GSM8K ---")
    print_summary("gsm8k")


if __name__ == "__main__":
    import sys
    phase_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(phase=phase_arg)
