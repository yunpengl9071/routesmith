# benchmark/strategies/linucb.py
"""LinUCB-27d benchmark harness adapter.

Self-contained 27-dim contextual bandit using LinUCB (disjoint arms).
Does not depend on ModelRegistry — feature extraction is standalone.
Two arms: arm 0 = weak model (GPT-4o-mini), arm 1 = strong model (GPT-4o).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from benchmark.config import STRONG_MODEL, WEAK_MODEL, cost_usd, MAX_TOKENS_MCQ, MAX_TOKENS_MATH
from benchmark.dataset import extract_answer, extract_gsm8k_answer, evaluate_mbpp_code
from benchmark.strategies.base import BaseStrategy, call_llm

_MATH_WORDS = frozenset([
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "sum", "product", "factorial", "probability", "percent", "ratio",
    "algebra", "geometry", "trigonometry", "calculus", "matrix", "vector",
    "theorem", "proof", "polynomial", "logarithm", "exponent",
])
_REASONING_WORDS = frozenset([
    "why", "because", "therefore", "however", "although", "explain",
    "reason", "logic", "argument", "conclude", "imply", "infer",
    "deduce", "analyze", "evaluate", "compare", "contrast", "consider",
])
_CODE_WORDS = frozenset([
    "function", "class", "variable", "loop", "array", "string", "integer",
    "implement", "algorithm", "code", "program", "debug", "compile",
    "syntax", "api", "database", "sql", "python", "javascript", "rust",
])
_CREATIVE_WORDS = frozenset([
    "write", "story", "poem", "creative", "imagine", "fiction",
    "character", "narrative", "dialogue", "scene", "metaphor", "lyric",
])


def _build_feature_vector(query: dict) -> np.ndarray:
    """Build a 27-dim feature vector from a query dict (no ModelRegistry)."""
    text = query.get("prompt", query.get("question", ""))
    words = text.lower().split()
    word_set = set(words)

    # 11 original message features (indices 0-10)
    f0 = math.log1p(len(text))
    f1 = math.log1p(len(words))
    sentences = re.split(r"[.!?]+", text)
    f2 = math.log1p(len([s for s in sentences if s.strip()]))
    f3 = float(len(re.findall(r"\?", text)))
    f4 = float(len(re.findall(r"\d+", text)))
    f5 = 1.0 if len(words) > 50 else 0.0
    f6 = float(len(re.findall(r"```", text))) / 2
    f7 = 1.0 if "?" in text else 0.0
    f8 = 1.0 if any(c.isupper() for c in text) else 0.0
    vocab = Counter(words)
    f9 = len(vocab) / max(len(words), 1)
    f10 = min(1.0, len(words) / 100)

    # 6 extended features (indices 11-16)
    def kw_score(target: frozenset) -> float:
        if not word_set:
            return 0.0
        return min(1.0, len(word_set & target) / len(word_set) * 10)

    f11 = kw_score(_MATH_WORDS)
    f12 = kw_score(_REASONING_WORDS)
    f13 = kw_score(_CODE_WORDS)
    f14 = kw_score(_CREATIVE_WORDS)
    f15 = min(1.0, (f11 + f12) / 2 + f3 * 0.1)  # difficulty estimate
    f16 = f9                                        # vocab richness alias

    # 10 zero-padded model features (indices 17-26) — arm distinction via arm index
    model_feats = [0.0] * 10

    vec = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
           f11, f12, f13, f14, f15, f16] + model_feats
    return np.array(vec, dtype=float)


def _max_tokens(dataset: str) -> int:
    if dataset == "mbpp":
        return 400
    if dataset == "gsm8k":
        return MAX_TOKENS_MATH
    return MAX_TOKENS_MCQ


def _correct(query: dict, response: str) -> bool:
    dataset = query.get("dataset", "mmlu")
    if dataset == "mmlu":
        return extract_answer(response) == query.get("answer_letter")
    if dataset == "gsm8k":
        pred = extract_gsm8k_answer(response)
        return pred == query.get("answer") if pred else False
    if dataset == "mbpp":
        return evaluate_mbpp_code(response, query.get("test_list", []))
    return False


class LinUCBStrategy(BaseStrategy):
    """
    LinUCB-27d binary routing strategy (disjoint, 2 arms).

    Arm 0 = weak model (WEAK_MODEL)
    Arm 1 = strong model (STRONG_MODEL)

    UCB score: θ_a^T x + alpha * sqrt(x^T A_a^{-1} x)
    Reward:    acc - 0.15 * cost/c_max

    Always calls weak first (for reward signal update).
    alpha: exploration parameter (default 1.5).
    """

    def __init__(self, alpha: float = 1.5, seed: int = 42) -> None:
        self._alpha = alpha
        self._seed = seed
        self._d = 27
        self._A = [np.eye(self._d) for _ in range(2)]
        self._b = [np.zeros(self._d) for _ in range(2)]
        # Pre-compute c_max for reward normalization
        self._c_max = cost_usd(STRONG_MODEL, 300, 50)

    @property
    def name(self) -> str:
        return f"linucb_27d_alpha{self._alpha:.1f}_seed{self._seed}"

    def _ucb_score(self, arm: int, x_norm: np.ndarray) -> float:
        A_inv = np.linalg.inv(self._A[arm])
        theta = A_inv @ self._b[arm]
        return float(theta @ x_norm + self._alpha * np.sqrt(x_norm @ A_inv @ x_norm))

    def _update(self, arm: int, x_norm: np.ndarray, reward: float) -> None:
        self._A[arm] += np.outer(x_norm, x_norm)
        self._b[arm] += reward * x_norm

    def route(self, query: dict) -> dict:
        x = _build_feature_vector(query)
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        scores = [self._ucb_score(arm, x_norm) for arm in range(2)]
        arm_chosen = int(np.argmax(scores))
        routing_decision = "strong" if arm_chosen == 1 else "weak"
        max_tok = _max_tokens(query.get("dataset", "mmlu"))

        # Always call weak (for reward signal)
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
        self._update(arm_chosen, x_norm, reward)

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
            "ucb_scores": scores,
        }
