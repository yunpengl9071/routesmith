"""NeuralUCB contextual bandit predictor for LLM routing.

Implements NeuralUCB (Zhou et al., 2020) adapted for LLM model selection.
A 2-layer MLP learns a nonlinear reward function from context features,
while UCB exploration is performed on the learned hidden-layer representation.

Architecture per arm:
    MLP: context (d_in) -> hidden (64, ReLU) -> scalar reward prediction
    UCB: Z_inv (h x h) tracks uncertainty in the hidden feature space

Score_a(x) = MLP_a(x) + alpha * sqrt(z^T Z_a^{-1} z)
    where z = last hidden activation of MLP_a

This captures nonlinear feature interactions that LinUCB misses, while
maintaining principled UCB exploration on the learned representation.

The reward signal combines quality, cost, and latency:
    reward = quality - cost_lambda * norm_cost - latency_lambda * norm_latency
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np

from routesmith.predictor.base import BasePredictor, PredictionResult
from routesmith.predictor.features import FeatureExtractor

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


class _MLP:
    """Lightweight 2-layer MLP: input -> hidden (ReLU) -> scalar.

    Supports forward, backward, and SGD updates. No external ML framework.
    """

    def __init__(self, d_in: int, d_hidden: int = 64, seed: int = 0) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        rng = np.random.RandomState(seed)

        # He initialization
        self.W1 = rng.randn(d_hidden, d_in).astype(np.float64) * math.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden, dtype=np.float64)
        self.W2 = rng.randn(d_hidden).astype(np.float64) * math.sqrt(2.0 / d_hidden)
        self.b2 = 0.0

        # Cache for backward pass
        self._x = np.zeros(d_in)
        self._h_pre = np.zeros(d_hidden)
        self._h = np.zeros(d_hidden)

    def forward(self, x: np.ndarray) -> float:
        self._x = x
        self._h_pre = self.W1 @ x + self.b1
        self._h = np.maximum(0.0, self._h_pre)  # ReLU
        return float(self.W2 @ self._h + self.b2)

    def get_last_hidden(self) -> np.ndarray:
        return self._h.copy()

    def backward(self, dloss_dout: float) -> dict[str, Any]:
        dW2 = dloss_dout * self._h
        db2 = dloss_dout
        dh = dloss_dout * self.W2
        dh_pre = dh * (self._h_pre > 0).astype(np.float64)
        dW1 = np.outer(dh_pre, self._x)
        db1 = dh_pre
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def apply_gradients(self, grads: dict[str, Any], lr: float) -> None:
        self.W1 += lr * grads["dW1"]
        self.b1 += lr * grads["db1"]
        self.W2 += lr * grads["dW2"]
        self.b2 += lr * grads["db2"]


class NeuralUCBPredictor(BasePredictor):
    """NeuralUCB predictor with MLP reward model and UCB on hidden features.

    Each arm has:
    - An MLP that maps context -> predicted reward (scalar)
    - A Z_inv matrix (h x h) tracking uncertainty in the hidden representation

    The UCB score is: MLP(x) + alpha * sqrt(z^T Z_inv z)
    where z is the post-ReLU hidden activation.

    Parameters
    ----------
    registry : ModelRegistry
        For feature extraction and cost lookups.
    alpha : float
        UCB exploration parameter on hidden features. Default 0.5.
    cost_lambda : float
        Cost penalty weight. Default 0.3.
    latency_lambda : float
        Latency penalty weight. Default 0.1.
    learning_rate : float
        SGD learning rate for MLP. Default 0.005.
    hidden_dim : int
        MLP hidden layer size. Default 64.
    warmup_rounds : int
        Min observations per arm before UCB activates. Default 2.
    replay_size : int
        Max replay buffer size for experience replay. Default 2000.
    """

    def __init__(
        self,
        registry: "ModelRegistry",
        alpha: float = 0.5,
        cost_lambda: float = 0.3,
        latency_lambda: float = 0.1,
        learning_rate: float = 0.005,
        hidden_dim: int = 64,
        warmup_rounds: int = 2,
        replay_size: int = 2000,
    ) -> None:
        self._registry = registry
        self._alpha = alpha
        self._cost_lambda = cost_lambda
        self._latency_lambda = latency_lambda
        self._lr = learning_rate
        self._hidden_dim = hidden_dim
        self._warmup_rounds = warmup_rounds
        self._replay_size = replay_size

        self._extractor = FeatureExtractor(registry)
        self._d: int | None = None

        # Per-arm state
        self._nets: dict[str, _MLP] = {}
        self._Z_inv: dict[str, np.ndarray] = {}
        self._arm_counts: dict[str, int] = {}

        # Replay buffer for mini-batch training stability
        self._replay: list[tuple[str, np.ndarray, float]] = []

        self._total_updates = 0

        # Cost/latency normalization
        models = registry.list_models()
        costs = [m.cost_per_1k_total for m in models]
        lats = [m.latency_p50_ms for m in models]
        self._max_cost = max(costs) if costs else 1.0
        self._max_latency = max(lats) if lats else 1.0

    def _ensure_arm(self, model_id: str, d: int) -> None:
        if model_id not in self._nets:
            seed = hash(model_id) % (2**31)
            self._nets[model_id] = _MLP(d, self._hidden_dim, seed=seed)
            self._Z_inv[model_id] = np.eye(self._hidden_dim, dtype=np.float64)
            self._arm_counts[model_id] = 0

    def _get_context(self, messages: list[dict[str, str]], model_id: str) -> np.ndarray:
        fv = self._extractor.extract(messages, model_id)
        x = np.array(fv.features, dtype=np.float64)
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        return x

    def _compute_reward(self, model_id: str, quality: float) -> float:
        model = self._registry.get(model_id)
        norm_cost = 0.0
        norm_latency = 0.0
        if model is not None:
            if self._max_cost > 0:
                norm_cost = model.cost_per_1k_total / self._max_cost
            if self._max_latency > 0:
                norm_latency = model.latency_p50_ms / self._max_latency
        return quality - self._cost_lambda * norm_cost - self._latency_lambda * norm_latency

    def predict(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> list[PredictionResult]:
        results: list[PredictionResult] = []

        for model_id in model_ids:
            x = self._get_context(messages, model_id)
            d = len(x)
            if self._d is None:
                self._d = d
            self._ensure_arm(model_id, d)

            count = self._arm_counts[model_id]

            if count < self._warmup_rounds:
                ucb_score = 2.0 + (self._warmup_rounds - count)
                mean_pred = 0.0
                confidence_width = float("inf")
                phase = "warmup"
            else:
                mean_pred = self._nets[model_id].forward(x)
                z = self._nets[model_id].get_last_hidden()
                Zz = self._Z_inv[model_id] @ z
                var = max(float(z @ Zz), 0.0)
                confidence_width = math.sqrt(var)
                ucb_score = mean_pred + self._alpha * confidence_width
                phase = "learned"

            display_quality = max(0.0, min(2.0, ucb_score))
            pred_confidence = min(1.0, count / 20.0) if count > 0 else 0.1

            results.append(PredictionResult(
                model_id=model_id,
                predicted_quality=display_quality,
                confidence=pred_confidence,
                metadata={
                    "neural_ucb_mean": round(mean_pred, 4),
                    "neural_ucb_confidence_width": round(confidence_width, 4),
                    "neural_ucb_score": round(ucb_score, 4),
                    "arm_count": count,
                    "phase": phase,
                },
            ))

        return sorted(results, key=lambda r: r.predicted_quality, reverse=True)

    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
        x = self._get_context(messages, model_id)
        d = len(x)
        if self._d is None:
            self._d = d
        self._ensure_arm(model_id, d)

        reward = self._compute_reward(model_id, actual_quality)

        # Forward pass
        pred = self._nets[model_id].forward(x)
        z = self._nets[model_id].get_last_hidden()

        # Update Z_inv via Sherman-Morrison on hidden features
        Zz = self._Z_inv[model_id] @ z
        denom = 1.0 + float(z @ Zz)
        self._Z_inv[model_id] -= np.outer(Zz, Zz) / denom

        # SGD on MSE loss
        error = pred - reward
        grads = self._nets[model_id].backward(-2.0 * error)
        self._nets[model_id].apply_gradients(grads, self._lr)

        self._arm_counts[model_id] += 1
        self._total_updates += 1

        # Experience replay for stability
        self._replay.append((model_id, x, reward))
        if len(self._replay) > self._replay_size:
            self._replay = self._replay[-self._replay_size:]

        if len(self._replay) > 10:
            for _ in range(8):
                rmid, rx, rrew = random.choice(self._replay)
                self._ensure_arm(rmid, len(rx))
                rpred = self._nets[rmid].forward(rx)
                rerr = rpred - rrew
                rgrads = self._nets[rmid].backward(-2.0 * rerr)
                self._nets[rmid].apply_gradients(rgrads, self._lr * 0.5)

    def warm_start(
        self,
        examples: list[tuple[list[dict[str, str]], str, float]],
        epochs: int = 3,
    ) -> None:
        """Pre-train the MLPs on labeled examples.

        examples: list of (messages, model_id, quality_score) tuples.
        """
        for _ in range(epochs):
            random.shuffle(examples)
            for messages, model_id, quality in examples:
                if self._registry.get(model_id) is None:
                    continue
                x = self._get_context(messages, model_id)
                d = len(x)
                if self._d is None:
                    self._d = d
                self._ensure_arm(model_id, d)

                reward = self._compute_reward(model_id, quality)
                pred = self._nets[model_id].forward(x)
                error = pred - reward
                grads = self._nets[model_id].backward(-2.0 * error)
                self._nets[model_id].apply_gradients(grads, self._lr)

                z = self._nets[model_id].get_last_hidden()
                Zz = self._Z_inv[model_id] @ z
                denom = 1.0 + float(z @ Zz)
                self._Z_inv[model_id] -= np.outer(Zz, Zz) / denom
                self._arm_counts[model_id] += 1

    def diagnostics(self) -> dict[str, Any]:
        arm_stats = {}
        for model_id in self._arm_counts:
            arm_stats[model_id] = {
                "count": self._arm_counts[model_id],
                "mlp_params": self._nets[model_id].d_in * self._nets[model_id].d_hidden
                              + self._nets[model_id].d_hidden + self._nets[model_id].d_hidden + 1
                              if model_id in self._nets else 0,
            }
        return {
            "type": "neural_ucb",
            "alpha": self._alpha,
            "cost_lambda": self._cost_lambda,
            "latency_lambda": self._latency_lambda,
            "learning_rate": self._lr,
            "hidden_dim": self._hidden_dim,
            "feature_dim": self._d,
            "total_updates": self._total_updates,
            "replay_buffer_size": len(self._replay),
            "arms": arm_stats,
        }
