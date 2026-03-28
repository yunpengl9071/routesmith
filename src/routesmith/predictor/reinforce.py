"""REINFORCE policy gradient predictor for cost-aware model routing.

Implements the REINFORCE algorithm (Williams, 1992) with a learned value
baseline for variance reduction. The policy is a softmax linear model
over the shared context features, and the baseline is a separate linear
regression estimating expected reward.

This is a genuine policy gradient RL method:
- Policy: pi(a|x) = softmax(W_a^T x + bias_a) over arms (models)
- Baseline: V(x) = v^T x + v_bias  (linear value estimate)
- Update: theta_a += lr * (R - V(x)) * grad log pi(a|x)
- Baseline update: v += lr_v * (R - V(x)) * x

The reward signal combines quality and cost:
    R = quality - cost_lambda * normalized_cost
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from routesmith.predictor.base import BasePredictor, PredictionResult
from routesmith.predictor.features import FeatureExtractor

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


class ReinforcePredictor(BasePredictor):
    """Policy gradient predictor using REINFORCE with learned baseline.

    The policy is a softmax linear model: for each arm (model) a,
    we maintain a weight vector w_a and bias b_a. Given context x,
    the probability of selecting arm a is:

        pi(a|x) = exp(w_a^T x + b_a) / sum_k exp(w_k^T x + b_k)

    The value baseline is a separate linear model V(x) = v^T x + v_bias
    that estimates the expected reward for any context x. This reduces
    variance in the policy gradient estimate.

    The policy gradient update (REINFORCE with baseline) is:

        For the selected arm a with observed reward R:
        advantage = R - V(x)
        w_a += lr * advantage * (x - sum_k pi(k|x) * x)  [simplified]
        v   += lr_baseline * advantage * x

    Parameters
    ----------
    registry : ModelRegistry
        Used for feature extraction and cost lookups.
    learning_rate : float
        Policy learning rate. Default 0.01.
    baseline_lr : float
        Value baseline learning rate. Default 0.05.
    cost_lambda : float
        Cost penalty weight. reward = quality - cost_lambda * normalized_cost.
    temperature : float
        Softmax temperature. Lower = more greedy, higher = more exploratory.
    entropy_bonus : float
        Coefficient for entropy regularization. Encourages exploration by
        penalizing low-entropy (overly confident) policies.
    """

    def __init__(
        self,
        registry: "ModelRegistry",
        learning_rate: float = 0.01,
        baseline_lr: float = 0.05,
        cost_lambda: float = 0.3,
        temperature: float = 1.0,
        entropy_bonus: float = 0.01,
    ) -> None:
        self._registry = registry
        self._lr = learning_rate
        self._baseline_lr = baseline_lr
        self._cost_lambda = cost_lambda
        self._temperature = temperature
        self._entropy_bonus = entropy_bonus

        self._extractor = FeatureExtractor(registry)
        self._d: int | None = None  # feature dimension

        # Policy parameters: model_id -> (weights, bias)
        self._policy_w: dict[str, np.ndarray] = {}
        self._policy_b: dict[str, float] = {}

        # Value baseline parameters
        self._value_w: np.ndarray | None = None
        self._value_bias: float = 0.0

        # Tracking
        self._total_updates = 0
        self._episode_rewards: list[float] = []  # rolling window for stats

        # Cost normalization
        costs = [m.cost_per_1k_total for m in registry.list_models()]
        self._max_cost = max(costs) if costs else 1.0

        # Cache last policy probabilities for the update step
        self._last_context: np.ndarray | None = None
        self._last_probs: dict[str, float] | None = None

    def _ensure_arm(self, model_id: str, d: int) -> None:
        """Initialize policy parameters for an arm if needed."""
        if model_id not in self._policy_w:
            # Small random initialization to break symmetry
            rng = np.random.RandomState(hash(model_id) % (2**31))
            self._policy_w[model_id] = rng.randn(d) * 0.01
            self._policy_b[model_id] = 0.0

    def _ensure_baseline(self, d: int) -> None:
        """Initialize value baseline if needed."""
        if self._value_w is None:
            self._value_w = np.zeros(d, dtype=np.float64)
            self._value_bias = 0.0

    def _get_context(
        self,
        messages: list[dict[str, str]],
        model_id: str,
    ) -> np.ndarray:
        """Extract and normalize feature vector."""
        fv = self._extractor.extract(messages, model_id)
        x = np.array(fv.features, dtype=np.float64)
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        return x

    def _compute_logits_and_probs(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
        """Compute softmax policy probabilities for all arms.

        Returns (logits_dict, probs_dict, shared_context).
        We use the message features only (first 11 dims) as shared context
        since model features differ per arm. The per-arm weights learn to
        map this shared context to arm-specific preferences.
        """
        # Use first model to get context dimension
        x_first = self._get_context(messages, model_ids[0])
        d = len(x_first)

        if self._d is None:
            self._d = d

        self._ensure_baseline(d)

        # Compute logits for each arm using its own context vector
        logits: dict[str, float] = {}
        contexts: dict[str, np.ndarray] = {}
        for model_id in model_ids:
            x = self._get_context(messages, model_id)
            self._ensure_arm(model_id, d)
            logit = float(self._policy_w[model_id] @ x + self._policy_b[model_id])
            logits[model_id] = logit / self._temperature
            contexts[model_id] = x

        # Numerically stable softmax
        max_logit = max(logits.values())
        exp_logits = {m: math.exp(logits[m] - max_logit) for m in model_ids}
        Z = sum(exp_logits.values())
        probs = {m: exp_logits[m] / Z for m in model_ids}

        return logits, probs, x_first

    def predict(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> list[PredictionResult]:
        """Predict quality using policy probabilities.

        The predicted_quality is set to the policy probability, which the
        router uses to rank models. Higher probability = policy prefers
        this arm for the given context.

        We also cache the probabilities and context so the subsequent
        update() call can compute the correct policy gradient.
        """
        logits, probs, context = self._compute_logits_and_probs(messages, model_ids)

        # Cache for update
        self._last_context = context
        self._last_probs = probs

        # Value baseline estimate
        if self._value_w is not None:
            baseline_val = float(self._value_w @ context + self._value_bias)
        else:
            baseline_val = 0.0

        # Entropy for diagnostics
        entropy = -sum(p * math.log(p + 1e-10) for p in probs.values())

        results = []
        for model_id in model_ids:
            prob = probs[model_id]
            results.append(
                PredictionResult(
                    model_id=model_id,
                    predicted_quality=prob,
                    confidence=min(1.0, self._total_updates / 50.0),
                    metadata={
                        "policy_prob": round(prob, 4),
                        "logit": round(logits[model_id], 4),
                        "baseline_value": round(baseline_val, 4),
                        "entropy": round(entropy, 4),
                        "phase": "warmup" if self._total_updates < 10 else "learned",
                    },
                )
            )

        return sorted(results, key=lambda r: r.predicted_quality, reverse=True)

    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
        """Update policy via REINFORCE with baseline.

        This is the core RL training step:
        1. Compute reward R = quality - cost_lambda * normalized_cost
        2. Compute advantage A = R - V(x) using learned baseline
        3. Update policy: w_a += lr * A * grad_log_pi(a|x)
        4. Update baseline: v += lr_v * A * x
        """
        x = self._get_context(messages, model_id)
        d = len(x)

        if self._d is None:
            self._d = d

        self._ensure_arm(model_id, d)
        self._ensure_baseline(d)

        # Step 1: Compute cost-penalized reward
        model_config = self._registry.get(model_id)
        if model_config is not None and self._max_cost > 0:
            normalized_cost = model_config.cost_per_1k_total / self._max_cost
        else:
            normalized_cost = 0.0

        reward = actual_quality - self._cost_lambda * normalized_cost

        # Step 2: Compute advantage using value baseline
        baseline_val = float(self._value_w @ x + self._value_bias)
        advantage = reward - baseline_val

        # Step 3: Policy gradient update (REINFORCE)
        # grad log pi(a|x) = x * (1 - pi(a|x))  for selected arm a
        # grad log pi(a|x) = x * (- pi(a|x))     for non-selected arm k != a
        #
        # We need current probabilities. If cached from predict(), use those.
        # Otherwise recompute.
        all_model_ids = list(self._policy_w.keys())
        if not all_model_ids:
            all_model_ids = [model_id]

        # Recompute probabilities for all known arms
        logits_raw: dict[str, float] = {}
        for mid in all_model_ids:
            self._ensure_arm(mid, d)
            x_mid = self._get_context(messages, mid) if mid != model_id else x
            logits_raw[mid] = float(
                self._policy_w[mid] @ x_mid + self._policy_b[mid]
            ) / self._temperature

        max_l = max(logits_raw.values())
        exp_l = {m: math.exp(logits_raw[m] - max_l) for m in all_model_ids}
        Z = sum(exp_l.values())
        probs = {m: exp_l[m] / Z for m in all_model_ids}

        # Update selected arm: w_a += lr * advantage * x * (1 - pi(a))
        pi_a = probs.get(model_id, 1.0 / len(all_model_ids))
        grad_selected = x * (1.0 - pi_a)
        self._policy_w[model_id] += self._lr * advantage * grad_selected
        self._policy_b[model_id] += self._lr * advantage * (1.0 - pi_a)

        # Update non-selected arms: w_k -= lr * advantage * x * pi(k)
        for mid in all_model_ids:
            if mid == model_id:
                continue
            pi_k = probs.get(mid, 0.0)
            x_mid = self._get_context(messages, mid)
            self._policy_w[mid] -= self._lr * advantage * x_mid * pi_k
            self._policy_b[mid] -= self._lr * advantage * pi_k

        # Entropy bonus: encourage exploration by pulling logits toward uniform
        if self._entropy_bonus > 0:
            uniform_prob = 1.0 / len(all_model_ids)
            for mid in all_model_ids:
                pi_k = probs.get(mid, uniform_prob)
                # Gradient of entropy w.r.t. logits: -(log(pi) + 1) * pi * (1-pi)
                # Simplified: push toward uniform
                entropy_grad = (uniform_prob - pi_k)
                self._policy_b[mid] += self._entropy_bonus * entropy_grad

        # Step 4: Update value baseline via MSE gradient
        # dL/dv = -2 * (R - V(x)) * x = -2 * advantage * x
        self._value_w += self._baseline_lr * advantage * x
        self._value_bias += self._baseline_lr * advantage

        # Tracking
        self._total_updates += 1
        self._episode_rewards.append(reward)
        if len(self._episode_rewards) > 200:
            self._episode_rewards = self._episode_rewards[-200:]

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic info about the predictor state."""
        arm_stats = {}
        for model_id in self._policy_w:
            w = self._policy_w[model_id]
            arm_stats[model_id] = {
                "weight_norm": round(float(np.linalg.norm(w)), 4),
                "bias": round(self._policy_b[model_id], 4),
            }

        avg_reward = (
            sum(self._episode_rewards) / len(self._episode_rewards)
            if self._episode_rewards
            else 0.0
        )

        diag: dict[str, Any] = {
            "type": "reinforce",
            "learning_rate": self._lr,
            "baseline_lr": self._baseline_lr,
            "cost_lambda": self._cost_lambda,
            "temperature": self._temperature,
            "entropy_bonus": self._entropy_bonus,
            "total_updates": self._total_updates,
            "avg_reward_last_200": round(avg_reward, 4),
            "max_cost": self._max_cost,
            "arms": arm_stats,
        }

        if self._value_w is not None:
            diag["baseline_weight_norm"] = round(float(np.linalg.norm(self._value_w)), 4)
            diag["baseline_bias"] = round(self._value_bias, 4)

        return diag

    def get_policy_probs(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> dict[str, float]:
        """Get the current policy's action probabilities for a given context.

        Useful for inspecting what the policy has learned.
        """
        _, probs, _ = self._compute_logits_and_probs(messages, model_ids)
        return {m: round(p, 4) for m, p in probs.items()}

    def get_arm_weights(self, model_id: str) -> dict[str, float] | None:
        """Get the learned policy weights for a specific arm.

        Returns dict mapping feature_name -> weight, or None if arm unknown.
        """
        w = self._policy_w.get(model_id)
        if w is None:
            return None

        names = self._extractor.ALL_FEATURE_NAMES
        if len(names) != len(w):
            return {f"f{i}": float(w[i]) for i in range(len(w))}
        return {name: round(float(w[i]), 4) for i, name in enumerate(names)}
