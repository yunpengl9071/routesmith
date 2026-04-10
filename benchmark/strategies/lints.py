# benchmark/strategies/lints.py
"""LinTS-27d benchmark harness adapter.

Wraps LinTSRouter from src/routesmith/predictor/lints.py.
Reuses _build_feature_vector from linucb.py for consistent features.
Two arms: arm 0 = weak model, arm 1 = strong model.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

from benchmark.config import STRONG_MODEL, WEAK_MODEL, cost_usd
from benchmark.strategies.base import BaseStrategy, call_llm
from benchmark.strategies.linucb import (
    _build_feature_vector,
    _correct,
    _max_tokens,
)
from routesmith.predictor.lints import LinTSRouter


class LinTSStrategy(BaseStrategy):
    """
    LinTS-27d binary routing strategy.

    Arm 0 = weak model (WEAK_MODEL)
    Arm 1 = strong model (STRONG_MODEL)
    No beta hyperparameter (unlike LinUCB) — only v_sq (default 1.0).

    Always calls weak first (for reward signal update).
    Restores router state from last result on resume.
    """

    def __init__(self, v_sq: float = 1.0, seed: int = 42) -> None:
        self._v_sq = v_sq
        self._seed = seed
        self._router = LinTSRouter(n_arms=2, d=27, v_sq=v_sq, seed=seed)
        self._c_max = cost_usd(STRONG_MODEL, 300, 50)

    @property
    def name(self) -> str:
        return f"lints_27d_vsq{self._v_sq:.1f}_seed{self._seed}"

    def route(self, query: dict) -> dict:
        x = _build_feature_vector(query)
        arm_chosen = self._router.select(x)
        routing_decision = "strong" if arm_chosen == 1 else "weak"
        max_tok = _max_tokens(query.get("dataset", "mmlu"))

        # Always call weak
        weak_resp, weak_pt, weak_ct = call_llm(
            WEAK_MODEL, query["prompt"], max_tokens=max_tok
        )
        weak_correct = _correct(query, weak_resp)
        weak_cost = cost_usd(WEAK_MODEL, weak_pt, weak_ct)

        if routing_decision == "strong":
            strong_resp, strong_pt, strong_ct = call_llm(
                STRONG_MODEL, query["prompt"], max_tokens=max_tok
            )
            strong_correct = _correct(query, strong_resp)
            strong_cost = cost_usd(STRONG_MODEL, strong_pt, strong_ct)
            final_correct = strong_correct
            final_model = STRONG_MODEL
            total_cost = weak_cost + strong_cost
        else:
            final_correct = weak_correct
            final_model = WEAK_MODEL
            total_cost = weak_cost
            strong_cost = 0.0

        reward = float(final_correct) - 0.15 * (total_cost / max(self._c_max, 1e-9))
        self._router.update(arm=arm_chosen, x=x, reward=reward)

        return {
            "query_id": query["query_id"],
            "dataset": query.get("dataset", ""),
            "category": query.get("category", ""),
            "strategy": self.name,
            "arm_chosen": arm_chosen,
            "routing_decision": routing_decision,
            "model": final_model,
            "final_model": final_model,
            "weak_correct": weak_correct,
            "correct": final_correct,
            "cost_usd": total_cost,
            "weak_cost_usd": weak_cost,
            "strong_cost_usd": strong_cost,
            "prompt_tokens": weak_pt,
            "completion_tokens": weak_ct,
            "router_state": self._router.get_state(),
        }

    def _on_resume(self, existing_results: list[dict]) -> None:
        """Restore LinTS state from last saved result's router_state."""
        if existing_results:
            last_state = existing_results[-1].get("router_state")
            if last_state:
                self._router.load_state(last_state)
                print(f"  Restored LinTS state at t={self._router._t}")
