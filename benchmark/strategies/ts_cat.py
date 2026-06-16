# benchmark/strategies/ts_cat.py
"""Context-free Thompson Sampling with per-category Beta priors (TS-Cat).

Maintains a Beta(α, β) posterior per query category, capturing
category-level routing preferences without per-query feature vectors.
This is the context-free bandit baseline for Experiment 1.
"""
from __future__ import annotations

import random

from benchmark.config import STRONG_MODEL, WEAK_MODEL, cost_usd, MAX_TOKENS_MCQ, MAX_TOKENS_MATH
from benchmark.dataset import extract_answer, extract_gsm8k_answer, evaluate_mbpp_code
from benchmark.strategies.base import BaseStrategy, call_llm


def _max_tokens(dataset: str) -> int:
    """Return appropriate max_tokens for the dataset type."""
    if dataset == "mbpp":
        return 400
    if dataset == "gsm8k":
        return MAX_TOKENS_MATH
    return MAX_TOKENS_MCQ  # mmlu default


def _correct(query: dict, response: str) -> bool:
    """Check if the model response is correct for the given query."""
    dataset = query.get("dataset", "mmlu")
    if dataset == "mmlu":
        return extract_answer(response) == query.get("answer_letter")
    if dataset == "gsm8k":
        pred = extract_gsm8k_answer(response)
        return pred == query.get("answer") if pred else False
    if dataset == "mbpp":
        return evaluate_mbpp_code(response, query.get("test_list", []))
    return False


class BetaPrior:
    """Beta(α, β) posterior for P(strong model needed in this category)."""

    def __init__(self) -> None:
        self.alpha = 1.0  # pseudocount for "strong model needed"
        self.beta = 1.0   # pseudocount for "weak model sufficient"

    def sample(self, rng: random.Random) -> float:
        """Sample from Beta(α, β)."""
        return rng.betavariate(self.alpha, self.beta)

    def update(self, strong_needed: bool) -> None:
        """Update posterior based on whether strong model was needed."""
        if strong_needed:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def to_dict(self) -> dict:
        """Serialize prior state."""
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict) -> "BetaPrior":
        """Deserialize prior state."""
        p = cls()
        p.alpha = d["alpha"]
        p.beta = d["beta"]
        return p


class TSCatStrategy(BaseStrategy):
    """
    Context-free Thompson Sampling router (binary: strong vs weak).

    Per-category Beta priors learn P(strong model needed | category).
    Always calls weak first (needed to update the prior).
    Routes to strong when Beta sample > 0.5.

    seed : int — for Beta sampling reproducibility across runs.
    categories : list[str] | None — known categories; "unknown" used as fallback.
    """

    def __init__(self, seed: int = 42, categories: list[str] | None = None) -> None:
        self._rng = random.Random(seed)
        self._seed = seed
        _cats = categories or ["STEM", "Medicine", "Humanities", "Social", "Common", "unknown"]
        self._priors: dict[str, BetaPrior] = {c: BetaPrior() for c in _cats}

    @property
    def name(self) -> str:
        return f"ts_cat_seed{self._seed}"

    def route(self, query: dict) -> dict:
        category = query.get("category", "unknown")
        prior = self._priors.setdefault(category, BetaPrior())
        p_strong = prior.sample(self._rng)
        routing_decision = "strong" if p_strong > 0.5 else "weak"
        dataset = query.get("dataset", "mmlu")
        max_tok = _max_tokens(dataset)

        # Always call weak (for prior update — production equivalent)
        weak_resp, weak_pt, weak_ct = call_llm(
            WEAK_MODEL, query["prompt"],
            system="You are a helpful assistant.",
            max_tokens=max_tok,
        )
        weak_correct = _correct(query, weak_resp)
        weak_cost = cost_usd(WEAK_MODEL, weak_pt, weak_ct)

        if routing_decision == "strong":
            strong_resp, strong_pt, strong_ct = call_llm(
                STRONG_MODEL, query["prompt"],
                system="You are a helpful assistant.",
                max_tokens=max_tok,
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

        # Update prior: weak wrong → strong was needed
        prior.update(strong_needed=not weak_correct)

        return {
            "query_id": query["query_id"],
            "dataset": dataset,
            "category": category,
            "strategy": self.name,
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
            "prior_state": {k: v.to_dict() for k, v in self._priors.items()},
        }

    def _on_resume(self, existing_results: list[dict]) -> None:
        """Reconstruct prior state from existing results."""
        for r in existing_results:
            cat = r.get("category", "unknown")
            prior = self._priors.setdefault(cat, BetaPrior())
            # weak_correct=True → weak was sufficient → beta++; False → strong needed → alpha++
            prior.update(strong_needed=not r.get("weak_correct", True))
