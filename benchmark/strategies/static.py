# benchmark/strategies/static.py
"""Static routing baselines: always-strong, always-weak, random."""
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


class StaticStrongStrategy(BaseStrategy):
    """Always routes to the strong model (GPT-4o). Upper bound baseline."""

    @property
    def name(self) -> str:
        return "static_strong"

    def route(self, query: dict) -> dict:
        model = STRONG_MODEL
        resp, pt, ct = call_llm(
            model, query["prompt"],
            system="You are a helpful assistant.",
            max_tokens=_max_tokens(query.get("dataset", "mmlu")),
        )
        return {
            "query_id": query["query_id"],
            "dataset": query.get("dataset", ""),
            "category": query.get("category", ""),
            "strategy": self.name,
            "model": model,
            "final_model": model,
            "raw_response": resp,
            "correct": _correct(query, resp),
            "cost_usd": cost_usd(model, pt, ct),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }


class StaticWeakStrategy(BaseStrategy):
    """Always routes to the weak model (GPT-4o-mini). Lower bound baseline."""

    @property
    def name(self) -> str:
        return "static_weak"

    def route(self, query: dict) -> dict:
        model = WEAK_MODEL
        resp, pt, ct = call_llm(
            model, query["prompt"],
            system="You are a helpful assistant.",
            max_tokens=_max_tokens(query.get("dataset", "mmlu")),
        )
        return {
            "query_id": query["query_id"],
            "dataset": query.get("dataset", ""),
            "category": query.get("category", ""),
            "strategy": self.name,
            "model": model,
            "final_model": model,
            "raw_response": resp,
            "correct": _correct(query, resp),
            "cost_usd": cost_usd(model, pt, ct),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }


class RandomRouterStrategy(BaseStrategy):
    """Randomly routes to strong or weak with equal probability."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._seed = seed

    @property
    def name(self) -> str:
        return f"random_router_seed{self._seed}"

    def route(self, query: dict) -> dict:
        model = self._rng.choice([STRONG_MODEL, WEAK_MODEL])
        resp, pt, ct = call_llm(
            model, query["prompt"],
            system="You are a helpful assistant.",
            max_tokens=_max_tokens(query.get("dataset", "mmlu")),
        )
        return {
            "query_id": query["query_id"],
            "dataset": query.get("dataset", ""),
            "category": query.get("category", ""),
            "strategy": self.name,
            "model": model,
            "final_model": model,
            "raw_response": resp,
            "correct": _correct(query, resp),
            "cost_usd": cost_usd(model, pt, ct),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }
