"""LinUCB contextual bandit predictor for cost-aware model routing.

Implements the LinUCB algorithm (Li et al., 2010) adapted for LLM routing.
Each model (arm) maintains a linear ridge regression model over the shared
context features. The exploration bonus is derived from the uncertainty
in the linear prediction, scaled by alpha.

Key difference from the existing UCBLearner: this predictor uses the full
19-dim feature vector (message stats + model metadata) to make context-dependent
routing decisions, rather than treating all queries identically.

The predicted quality incorporates a cost penalty so that the bandit jointly
optimizes quality and cost:

    reward = quality - cost_lambda * normalized_cost
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from routesmith.predictor.base import BasePredictor, PredictionResult
from routesmith.predictor.features import FeatureExtractor

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


class LinUCBPredictor(BasePredictor):
    """Contextual bandit predictor using the LinUCB (disjoint) algorithm.

    Each registered model is an arm. The context vector is the 19-dim
    feature vector from FeatureExtractor (11 message features + 8 model
    features). For each arm we maintain:

        A_a  : (d x d) matrix  = D_a^T D_a + I   (design matrix + ridge)
        b_a  : (d,) vector     = D_a^T r_a        (reward-weighted features)

    At prediction time:
        theta_a = A_a^{-1} b_a                    (ridge estimate)
        p_a     = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)   (UCB score)

    The UCB score is used as predicted_quality so the existing router
    picks the cheapest model whose UCB score exceeds min_quality.

    Parameters
    ----------
    registry : ModelRegistry
        Used for feature extraction and cost lookups.
    alpha : float
        Exploration parameter. Controls width of confidence interval.
        Higher alpha = more exploration. Typical range: 0.5 - 3.0.
    cost_lambda : float
        Weight for cost penalty in the reward signal.
        reward = quality - cost_lambda * (cost / max_cost_in_registry).
        Set to 0.0 to optimize quality only.
    warmup_rounds : int
        Number of initial rounds where each arm is forced (round-robin)
        to collect at least one observation per model before UCB kicks in.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        alpha: float = 1.5,
        cost_lambda: float = 0.3,
        warmup_rounds: int = 1,
    ) -> None:
        self._registry = registry
        self._alpha = alpha
        self._cost_lambda = cost_lambda
        self._warmup_rounds = warmup_rounds

        self._extractor = FeatureExtractor(registry)
        self._d: int | None = None  # feature dimension, set on first call

        # Per-arm state: model_id -> (A, b, count)
        self._arms: dict[str, dict[str, Any]] = {}

        # Global counters
        self._total_updates = 0

        # Cost normalization: max cost across registered models
        costs = [m.cost_per_1k_total for m in registry.list_models()]
        self._max_cost = max(costs) if costs else 1.0

    def _ensure_arm(self, model_id: str, d: int) -> dict[str, Any]:
        """Lazily initialize per-arm matrices."""
        if model_id not in self._arms:
            self._arms[model_id] = {
                "A": np.eye(d, dtype=np.float64),
                "b": np.zeros(d, dtype=np.float64),
                "A_inv": np.eye(d, dtype=np.float64),
                "count": 0,
            }
        return self._arms[model_id]

    def _get_context(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        context=None,
    ) -> np.ndarray:
        """Extract and normalize feature vector as context."""
        fv = self._extractor.extract(messages, model_id, context=context)
        x = np.array(fv.features, dtype=np.float64)

        # L2 normalize to stabilize ridge regression
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm

        return x

    def predict(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
        context=None,
    ) -> list[PredictionResult]:
        """Predict quality for each model using LinUCB scores.

        During warmup (fewer than warmup_rounds updates per arm),
        arms with insufficient data get a boosted exploration score
        to ensure all arms are tried.
        """
        results: list[PredictionResult] = []

        for model_id in model_ids:
            x = self._get_context(messages, model_id, context=context)
            d = len(x)

            if self._d is None:
                self._d = d

            arm = self._ensure_arm(model_id, d)

            # Compute LinUCB score
            A_inv = arm["A_inv"]  # noqa: N806
            theta = A_inv @ arm["b"]

            # Predicted reward (mean estimate)
            mean_pred = float(theta @ x)

            # Confidence width
            confidence = float(np.sqrt(x @ A_inv @ x))

            # UCB score
            ucb_score = mean_pred + self._alpha * confidence

            # During warmup, give unexplored arms a large bonus
            if arm["count"] < self._warmup_rounds:
                ucb_score = 2.0 + (self._warmup_rounds - arm["count"])

            # Clamp to reasonable range for compatibility with router
            # (router expects quality in ~[0, 1] but UCB can exceed 1)
            display_quality = max(0.0, min(2.0, ucb_score))

            # Confidence for the router: higher when we have more data
            pred_confidence = min(1.0, arm["count"] / 20.0) if arm["count"] > 0 else 0.1

            results.append(
                PredictionResult(
                    model_id=model_id,
                    predicted_quality=display_quality,
                    confidence=pred_confidence,
                    metadata={
                        "linucb_mean": round(mean_pred, 4),
                        "linucb_confidence_width": round(confidence, 4),
                        "linucb_ucb_score": round(ucb_score, 4),
                        "linucb_arm_count": arm["count"],
                        "phase": "warmup" if arm["count"] < self._warmup_rounds else "learned",
                    },
                )
            )

        return sorted(results, key=lambda r: r.predicted_quality, reverse=True)

    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
        context=None,
    ) -> None:
        """Update the arm's ridge regression with observed reward.

        The reward incorporates a cost penalty:
            reward = quality - cost_lambda * (cost / max_cost)

        This encourages the bandit to prefer cheaper models when quality
        is similar, while still routing to expensive models when they
        provide meaningfully higher quality.
        """
        x = self._get_context(messages, model_id, context=context)
        d = len(x)

        if self._d is None:
            self._d = d

        arm = self._ensure_arm(model_id, d)

        # Compute cost-penalized reward
        model_config = self._registry.get(model_id)
        if model_config is not None and self._max_cost > 0:
            normalized_cost = model_config.cost_per_1k_total / self._max_cost
        else:
            normalized_cost = 0.0

        if reward_override is not None:
            reward = reward_override
        else:
            reward = actual_quality - self._cost_lambda * normalized_cost

        # Sherman-Morrison rank-1 update for A_inv
        # A_new = A + x x^T
        # A_inv_new = A_inv - (A_inv x x^T A_inv) / (1 + x^T A_inv x)
        A_inv = arm["A_inv"]  # noqa: N806
        A_inv_x = A_inv @ x  # noqa: N806
        denom = 1.0 + float(x @ A_inv_x)
        arm["A_inv"] = A_inv - np.outer(A_inv_x, A_inv_x) / denom

        # Update A and b
        arm["A"] = arm["A"] + np.outer(x, x)
        arm["b"] = arm["b"] + reward * x
        arm["count"] += 1

        self._total_updates += 1

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic info about the predictor state."""
        arm_stats = {}
        for model_id, arm in self._arms.items():
            theta = arm["A_inv"] @ arm["b"]
            arm_stats[model_id] = {
                "count": arm["count"],
                "theta_norm": round(float(np.linalg.norm(theta)), 4),
                "theta_mean": round(float(np.mean(theta)), 4),
            }

        return {
            "type": "linucb",
            "alpha": self._alpha,
            "cost_lambda": self._cost_lambda,
            "warmup_rounds": self._warmup_rounds,
            "feature_dim": self._d,
            "total_updates": self._total_updates,
            "max_cost": self._max_cost,
            "arms": arm_stats,
        }

    def get_arm_weights(self, model_id: str) -> dict[str, float] | None:
        """Get the learned feature weights for a specific arm.

        Useful for interpreting what features drive routing to this model.
        Returns dict mapping feature_name -> weight, or None if arm unknown.
        """
        arm = self._arms.get(model_id)
        if arm is None:
            return None

        theta = arm["A_inv"] @ arm["b"]
        names = self._extractor.ALL_FEATURE_NAMES
        if len(names) != len(theta):
            return {f"f{i}": float(theta[i]) for i in range(len(theta))}
        return {name: round(float(theta[i]), 4) for i, name in enumerate(names)}

    def add_arm(self, model_id: str) -> None:
        """Signal intent to add a new arm.

        LinUCBPredictor uses dict-based lazy arm init — the arm is
        initialized automatically by _ensure_arm() on first predict/update.
        This method is a no-op if the arm already has state.
        """
        pass  # arm auto-initialized on first predict via _ensure_arm

    def remove_arm(self, model_id: str) -> None:
        """Remove arm state for a deregistered model."""
        self._arms.pop(model_id, None)

    def serialize_state(self) -> bytes:
        """Serialize arm states to JSON bytes (no pickle)."""
        import json
        state = {
            "total_updates": self._total_updates,
            "d": self._d,
            "arms": {
                model_id: {
                    "A": arm["A"].tolist(),
                    "A_inv": arm["A_inv"].tolist(),
                    "b": arm["b"].tolist(),
                    "count": arm["count"],
                }
                for model_id, arm in self._arms.items()
            },
        }
        return json.dumps(state).encode()

    def load_state(self, blob: bytes) -> None:
        """Load arm states from JSON bytes.

        Skips load if stored feature dimension differs (cold start on mismatch).
        """
        import json

        import numpy as np
        state = json.loads(blob.decode())
        stored_d = state.get("d")
        if stored_d is not None and self._d is not None and stored_d != self._d:
            return  # dimension mismatch — cold start
        self._total_updates = state.get("total_updates", 0)
        if stored_d is not None:
            self._d = stored_d
        for model_id, arm_data in state.get("arms", {}).items():
            A = np.array(arm_data["A"], dtype=np.float64)  # noqa: N806
            # Prefer the stored A_inv for numerical identity; fall back to
            # recomputing it for blobs serialized before this fix was applied.
            if "A_inv" in arm_data:
                A_inv = np.array(arm_data["A_inv"], dtype=np.float64)  # noqa: N806
            else:
                A_inv = np.linalg.inv(A)  # noqa: N806
            self._arms[model_id] = {
                "A": A,
                "b": np.array(arm_data["b"], dtype=np.float64),
                "A_inv": A_inv,
                "count": arm_data["count"],
            }
