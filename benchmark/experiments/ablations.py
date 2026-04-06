# benchmark/experiments/ablations.py
"""Ablation experiments for the RouteSmith paper.

1. Feature dimensionality: 11d vs 17d vs 27d
2. LinUCB beta sensitivity: alpha in {0.5, 1.5, 3.0} vs LinTS (no beta)
3. Warm-start labels: 0, 100, 500 labels before online learning
"""
from __future__ import annotations

import numpy as np

from benchmark.config import MMLU_CACHE, SEEDS
from benchmark.dataset import load_mmlu_sample
from benchmark.harness import run_experiment
from benchmark.strategies.linucb import LinUCBStrategy, _build_feature_vector
from benchmark.strategies.lints import LinTSStrategy


def _truncated_lints(d_actual: int, seed: int) -> LinTSStrategy:
    """LinTS that zeros out features beyond index d_actual."""
    strat = LinTSStrategy(seed=seed)

    orig_route = strat.route.__func__

    def patched_route(self, query: dict) -> dict:
        x = _build_feature_vector(query)
        x_zeroed = x.copy()
        x_zeroed[d_actual:] = 0.0

        # Swap feature in router calls temporarily
        orig_select = self._router.select
        orig_update = self._router.update

        self._router.select = lambda _x: orig_select(x_zeroed)
        self._router.update = lambda arm, _x, reward: orig_update(arm, x_zeroed, reward)

        result = orig_route(self, query)
        result["strategy"] = f"lints_{d_actual}d_seed{seed}"

        self._router.select = orig_select
        self._router.update = orig_update

        return result

    # Bind patched_route
    import types
    strat.route = types.MethodType(patched_route, strat)

    # Override name property
    strat._name_override = f"lints_{d_actual}d_seed{seed}"
    strat.__class__ = type(
        f"LinTS{d_actual}d",
        (LinTSStrategy,),
        {"name": property(lambda s: s._name_override)},
    )
    return strat


def ablation_feature_dims(queries: list[dict], seed: int = 42) -> None:
    """Compare 11d, 17d, 27d feature vectors with LinTS."""
    print("\n=== Ablation: Feature Dimensionality ===")
    for d in [11, 17, 27]:
        print(f"\n--- LinTS {d}d features ---")
        strat = _truncated_lints(d_actual=d, seed=seed)
        run_experiment(strat, queries, tag=f"dim{d}_ablation")


def ablation_beta_sensitivity(queries: list[dict], seed: int = 42) -> None:
    """Compare LinUCB at different alpha (beta) vs LinTS (no beta)."""
    print("\n=== Ablation: LinUCB Beta Sensitivity vs LinTS ===")
    for alpha in [0.5, 1.5, 3.0]:
        print(f"\n--- LinUCB alpha={alpha} ---")
        strat = LinUCBStrategy(alpha=alpha, seed=seed)
        run_experiment(strat, queries, tag="beta_ablation")
    print(f"\n--- LinTS (no beta) ---")
    run_experiment(LinTSStrategy(seed=seed), queries, tag="beta_ablation")


def ablation_warm_start(queries: list[dict], seed: int = 42) -> None:
    """Test LinTS with 0, 100, 500 warm-start labels (oracle positive labels)."""
    print("\n=== Ablation: Warm-Start Labels ===")
    for n_labels in [0, 100, 500]:
        print(f"\n--- LinTS warm-start={n_labels} labels ---")
        strat = LinTSStrategy(seed=seed)
        strat._name_override = f"lints_warmstart{n_labels}_seed{seed}"
        strat.__class__ = type(
            f"LinTSWarm{n_labels}",
            (LinTSStrategy,),
            {"name": property(lambda s: s._name_override)},
        )

        # Pre-train on first n_labels queries with oracle labels
        if n_labels > 0:
            warm_queries = queries[:n_labels]
            for q in warm_queries:
                x = _build_feature_vector(q)
                # Warm-start: assume strong model is correct (optimistic initialization)
                strat._router.update(arm=1, x=x, reward=1.0)

        eval_queries = queries[n_labels:]
        run_experiment(strat, eval_queries, tag=f"warmstart{n_labels}_ablation")


def main() -> None:
    print("Loading MMLU dataset...")
    queries = load_mmlu_sample(cache_path=str(MMLU_CACHE))

    ablation_feature_dims(queries)
    ablation_beta_sensitivity(queries)
    ablation_warm_start(queries)

    print("\nAll ablations complete. Run `python3 -m benchmark.resume` for status.")


if __name__ == "__main__":
    main()
