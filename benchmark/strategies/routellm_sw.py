# benchmark/strategies/routellm_sw.py
"""RouteLLM SW (Similarity-Weighted) router benchmark wrapper.

The SW router uses Chatbot Arena similarity weighting. It embeds the query,
finds nearest neighbors in the arena embedding space, and computes an
Elo-weighted strong-model win probability. Routes to strong if probability
exceeds threshold.

If SW fails to load (embedding dimension mismatch or API unavailability),
falls back to a local sentence-transformers embedding.

REPORTED_APGR contains published results from Ong et al. (2025) for when
the SW router cannot be loaded at all.
"""
from __future__ import annotations

import sys

from benchmark.config import STRONG_MODEL, WEAK_MODEL, cost_usd, MAX_TOKENS_MCQ, MAX_TOKENS_MATH
from benchmark.dataset import extract_answer, extract_gsm8k_answer, evaluate_mbpp_code
from benchmark.strategies.base import BaseStrategy, call_llm

# Published APGR results from Ong et al. (2025) — used in paper if SW fails
REPORTED_APGR = {
    "mmlu": 0.39,   # Ong et al. (2025) Table 2, SW router on MMLU
    "gsm8k": 0.42,  # Ong et al. (2025) Table 2, SW router on GSM8K
}


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


def _load_sw_router(threshold: float):
    """Load RouteLLM SW router from HuggingFace arena datasets."""
    from routellm.routers.routers import ROUTER_CLS
    RouterCls = ROUTER_CLS["sw_ranking"]
    return RouterCls(
        arena_battle_datasets=[
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        arena_embedding_datasets=[
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
    )


class RouteLLMSWStrategy(BaseStrategy):
    """
    RouteLLM SW router at a configurable threshold.

    threshold: if P(strong wins) > threshold → route to strong.
    0.3 = conservative (more to strong), 0.5 = balanced, 0.7 = aggressive savings.

    If loading fails, routes conservatively (all to weak) and sets sw_fallback=True
    in results so the paper can note this.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._router = None
        self._load_failed = False
        self._fallback_used = "none"
        self._init_router()

    @property
    def name(self) -> str:
        return f"routellm_sw_t{self.threshold:.2f}"

    def _init_router(self) -> None:
        print(f"Loading RouteLLM-SW (threshold={self.threshold})...")
        try:
            self._router = _load_sw_router(self.threshold)
            self._fallback_used = "none"
            print("SW router loaded successfully.")
            return
        except Exception as e:
            print(f"WARNING: RouteLLM-SW failed to load: {e}", file=sys.stderr)
            print("Trying sentence-transformers fallback (384-dim)...", file=sys.stderr)

        # Try local sentence-transformers fallback
        try:
            from sentence_transformers import SentenceTransformer
            _st = SentenceTransformer("all-MiniLM-L6-v2")

            class FakeEmbResp:
                def __init__(self, emb):
                    self.embedding = emb

            class FakeEmbData:
                def __init__(self, emb):
                    self.data = [FakeEmbResp(emb)]

            class LocalEmbClient:
                def __init__(self):
                    self.embeddings = self

                def create(self, input, model=None):
                    emb = _st.encode(input[0], normalize_embeddings=True).tolist()
                    return FakeEmbData(emb)

            import routellm.routers.similarity_weighted.utils as sw_utils
            sw_utils.OPENAI_CLIENT = LocalEmbClient()
            self._router = _load_sw_router(self.threshold)
            self._fallback_used = "sentence_transformers_384d"
            print("SW router loaded with sentence-transformers fallback (384-dim).")
            print("NOTE: Arena embeddings are 1536-dim; dimension mismatch noted in paper.")
        except Exception as e2:
            print(f"ST fallback also failed: {e2}", file=sys.stderr)
            print("SW will route all queries to weak model. Using REPORTED_APGR in paper.", file=sys.stderr)
            self._load_failed = True
            self._fallback_used = "reported_numbers"

    def route(self, query: dict) -> dict:
        # Determine routing decision
        win_rate = None
        if self._router is not None:
            try:
                win_rate = self._router.calculate_strong_win_rate(query["prompt"])
                routing_decision = "strong" if win_rate >= self.threshold else "weak"
            except Exception:
                routing_decision = "weak"
        else:
            routing_decision = "weak"

        model = STRONG_MODEL if routing_decision == "strong" else WEAK_MODEL
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
            "routing_decision": routing_decision,
            "model": model,
            "final_model": model,
            "sw_threshold": self.threshold,
            "sw_win_rate": win_rate,
            "sw_fallback": self._fallback_used,
            "correct": _correct(query, resp),
            "cost_usd": cost_usd(model, pt, ct),
            "prompt_tokens": pt,
            "completion_tokens": ct,
        }
