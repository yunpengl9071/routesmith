# benchmark/experiments/exp2_multimodel.py
"""Experiment 2: 5-arm multi-model quality routing on MMLU.

Uses LinTS-27d with 5 model arms. Quality-dominant reward.
RouteLLM-SW is excluded (binary-only architecture).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from benchmark.config import (
    MULTI_ARMS, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
    RESULTS_DIR, RATE_LIMIT_S, MAX_TOKENS_MCQ, cost_usd, MMLU_CACHE
)
from benchmark.dataset import load_mmlu_sample, extract_answer
from benchmark.strategies.linucb import _build_feature_vector

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))
from routesmith.predictor.lints import LinTSRouter

N_ARMS = len(MULTI_ARMS)
ARM_IDS = [a["id"] for a in MULTI_ARMS]
ARM_NAMES = [a["name"] for a in MULTI_ARMS]
QUALITY_WEIGHT = 0.85
COST_WEIGHT = 0.15
C_MAX = max(cost_usd(a["id"], 300, 100) for a in MULTI_ARMS)
SEEDS_EXP2 = [42, 43, 44]   # 3 seeds for cost control


def _call_model(model_id: str, prompt: str) -> tuple[str, int, int]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS_MCQ,
        temperature=0.0,
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _reward(correct: bool, c: float) -> float:
    return QUALITY_WEIGHT * float(correct) - COST_WEIGHT * (c / max(C_MAX, 1e-9))


def run_lints_5arm(queries: list[dict], seed: int, results_path: Path) -> list[dict]:
    """Run LinTS-5arm on queries with resume support."""
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        done_ids = {r["query_id"] for r in results}
        router = LinTSRouter(n_arms=N_ARMS, d=27, v_sq=1.0, seed=seed)
        if results:
            last_state = results[-1].get("router_state")
            if last_state:
                router.load_state(last_state)
        print(f"  Resuming seed={seed}: {len(done_ids)}/{len(queries)} done, t={router._t}")
    else:
        results = []
        done_ids = set()
        router = LinTSRouter(n_arms=N_ARMS, d=27, v_sq=1.0, seed=seed)

    for q in queries:
        if q["query_id"] in done_ids:
            continue

        x = _build_feature_vector(q)
        arm = router.select(x)
        model_id = ARM_IDS[arm]

        resp, pt, ct = _call_model(model_id, q["prompt"])
        correct = extract_answer(resp) == q.get("answer_letter")
        c = cost_usd(model_id, pt, ct)
        reward = _reward(correct, c)
        router.update(arm=arm, x=x, reward=reward)
        time.sleep(RATE_LIMIT_S)

        result = {
            "query_id": q["query_id"],
            "category": q.get("category", ""),
            "strategy": f"lints_5arm_seed{seed}",
            "arm_chosen": arm,
            "model_name": ARM_NAMES[arm],
            "model_id": model_id,
            "final_model": model_id,
            "correct": correct,
            "cost_usd": c,
            "reward": reward,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "router_state": router.get_state(),
        }
        results.append(result)
        done_ids.add(q["query_id"])

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        n = len(results)
        acc = sum(r["correct"] for r in results) / n
        print(f"  [{n:4d}/{len(queries)}] seed={seed} arm={arm}({ARM_NAMES[arm][:8]}) | "
              f"{'✓' if correct else '✗'} | acc={acc:.1%}")

    return results


def main() -> None:
    print("=== Experiment 2: 5-Arm Multi-Model Quality Routing ===")
    queries = load_mmlu_sample(cache_path=str(MMLU_CACHE))
    print(f"Using {len(queries)} MMLU queries, {N_ARMS} arms, {len(SEEDS_EXP2)} seeds")

    for seed in SEEDS_EXP2:
        print(f"\n--- LinTS-5arm seed={seed} ---")
        path = RESULTS_DIR / f"exp2_lints_5arm_seed{seed}_results.json"
        results = run_lints_5arm(queries, seed=seed, results_path=path)
        acc = sum(r["correct"] for r in results) / len(results)
        cost = sum(r["cost_usd"] for r in results)
        print(f"  DONE: acc={acc:.1%}, cost=${cost:.4f}")

    # Routing distribution summary
    print("\n--- Routing Distribution ---")
    for seed in SEEDS_EXP2:
        path = RESULTS_DIR / f"exp2_lints_5arm_seed{seed}_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f)
        counts: dict[str, int] = {}
        for r in results:
            n = r["model_name"]
            counts[n] = counts.get(n, 0) + 1
        total = len(results)
        print(f"  seed={seed}: " + ", ".join(f"{k}={v/total:.0%}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
