"""
Feature-based Similarity-Weighted (SW) Router.

Implements the core RouteLLM SW algorithm (Ong et al., 2024) but using
the RouteSmith 27-dim feature vectors in place of text embeddings.

RouteLLM SW uses:
  1. Cosine similarity between the current query embedding and training queries
  2. Exponential weighting: w_i = exp(gamma * sim(q, q_i))
  3. Weighted Elo/Bradley-Terry to estimate win probability for strong model
  4. Route to strong if P(strong wins) > threshold

Our feature-based SW:
  - Same algorithm, but uses the 27-dim feature vector for similarity
  - No embedding API or GPU required → direct apples-to-apples comparison
  - Shows the tradeoff: semantic embeddings vs. lightweight feature similarity

This allows an honest comparison:
  * SW-features: fast, no GPU, trains on same 500 labels as our LinUCB
  * RouteLLM-SW: uses full text embeddings + 55K+ Chatbot Arena labels
"""

import math
import sys
import os

# Allow standalone and package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.feature_utils import extract_features_27d, l2_norm


def _cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two feature vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class SWRouter:
    """
    Similarity-Weighted router using feature vectors.

    Parameters
    ----------
    gamma : float
        Exponential weighting factor. Higher = more local (only very similar
        training examples matter). Default 5.0 matches RouteLLM hyperparameter.
    threshold : float
        Routing threshold: route to strong model if weighted win rate > threshold.
        At threshold=0.5 (equal cost), always routes to the better model.
        Increase to trade quality for cost savings.
    strong_model_id : str
        Identifier for the strong (expensive) model.
    weak_model_id : str
        Identifier for the weak (cheap) model.
    """

    def __init__(
        self,
        gamma: float = 5.0,
        threshold: float = 0.5,
        strong_model_id: str = "gpt4",
        weak_model_id: str = "mixtral",
    ):
        self.gamma = gamma
        self.threshold = threshold
        self.strong_model_id = strong_model_id
        self.weak_model_id = weak_model_id

        # Training memory: list of (feature_vec, strong_win_label)
        # strong_win_label = 1 if strong was better, 0 if weak was better/equal
        self._memory: list = []

    def warm_start(self, data: list) -> None:
        """
        Pre-train on labeled examples.

        Parameters
        ----------
        data : list of (prompt, strong_correct, weak_correct)
            Labels from dataset. strong_win = 1 if strong_correct > weak_correct.
        """
        for prompt, s_ok, w_ok in data:
            feat = extract_features_27d(prompt, self.strong_model_id)
            # Bradley-Terry label: strong wins if it was correct when weak wasn't
            # (or both correct → treat as strong win since strong is preferred when equal)
            strong_win = 1 if s_ok >= w_ok else 0
            self._memory.append((feat, strong_win))

    def _predict_strong_prob(self, query_feat: list) -> float:
        """
        Predict P(strong model wins) for the given query features.
        Returns 0.5 if no training data available.
        """
        if not self._memory:
            return 0.5

        # Compute exponentially weighted average of strong-win labels
        total_weight = 0.0
        weighted_wins = 0.0

        for mem_feat, strong_win in self._memory:
            sim = _cosine_sim(query_feat, mem_feat)
            # Clip similarity to [-1, 1] for numerical safety
            sim = max(-1.0, min(1.0, sim))
            w = math.exp(self.gamma * sim)
            total_weight += w
            weighted_wins += w * strong_win

        if total_weight < 1e-12:
            return 0.5
        return weighted_wins / total_weight

    def route(self, prompt: str) -> str:
        """
        Route a query. Returns model_id ("gpt4" or "mixtral").
        """
        feat = extract_features_27d(prompt, self.strong_model_id)
        p_strong = self._predict_strong_prob(feat)
        return self.strong_model_id if p_strong >= self.threshold else self.weak_model_id

    def score_for_ranking(self, prompt: str) -> float:
        """
        Return a scalar score for APGR computation (higher = more likely to send to strong).
        This is P(strong wins) in [-inf, +inf] logit space for better ranking resolution.
        """
        feat = extract_features_27d(prompt, self.strong_model_id)
        p = self._predict_strong_prob(feat)
        p = max(1e-9, min(1 - 1e-9, p))
        return math.log(p / (1 - p))  # logit

    def update(self, prompt: str, strong_correct: int, weak_correct: int) -> None:
        """Online update: add a new observation to memory."""
        feat = extract_features_27d(prompt, self.strong_model_id)
        strong_win = 1 if strong_correct >= weak_correct else 0
        self._memory.append((feat, strong_win))

    @property
    def n_training_examples(self) -> int:
        return len(self._memory)
