# src/routesmith/predictor/lints.py
"""LinTS-27d: Linear Thompson Sampling for contextual bandit LLM routing.

Maintains a Gaussian posterior per arm a:
    p(θ_a | D_a) = N(μ_a, v² Σ_a)
where Σ_a = A_a⁻¹, μ_a = Σ_a b_a.

At each step:
    θ̃_a ~ N(μ_a, v² Σ_a) for each arm a
    a*   = argmax_a φ(x_t)ᵀ θ̃_a

Update on (a, x, r):
    A_a ← A_a + φφᵀ
    b_a ← b_a + r · φ

Regret bound: O(d√(T log T)) with high probability (Abeille & Lazaric, 2017).
No β hyperparameter required (unlike LinUCB).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LinTSArm:
    """Per-arm Gaussian posterior for LinTS."""

    d: int
    A: np.ndarray = field(init=False)  # (d, d) — design matrix + ridge
    b: np.ndarray = field(init=False)  # (d,)   — reward-weighted features

    def __post_init__(self) -> None:
        self.A = np.eye(self.d)
        self.b = np.zeros(self.d)

    @property
    def mu(self) -> np.ndarray:
        """Posterior mean: A⁻¹ b."""
        return np.linalg.solve(self.A, self.b)

    @property
    def sigma(self) -> np.ndarray:
        """Posterior covariance: A⁻¹."""
        return np.linalg.inv(self.A)

    def sample(self, rng: np.random.Generator, v_sq: float = 1.0) -> np.ndarray:
        """Sample θ̃ ~ N(μ, v² Σ)."""
        return rng.multivariate_normal(self.mu, v_sq * self.sigma)

    def update(self, x: np.ndarray, reward: float) -> None:
        """Rank-1 update: A += xxᵀ, b += r·x."""
        self.A += np.outer(x, x)
        self.b += reward * x

    def to_dict(self) -> dict[str, Any]:
        return {"A": self.A.tolist(), "b": self.b.tolist(), "d": self.d}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinTSArm:
        arm = cls(d=data["d"])
        arm.A = np.array(data["A"])
        arm.b = np.array(data["b"])
        return arm


class LinTSRouter:
    """
    Multi-arm Linear Thompson Sampling router.

    Parameters
    ----------
    n_arms : int
        Number of model arms.
    d : int
        Feature dimension (35 for RouteSmith's full feature space).
    v_sq : float
        Noise variance scaling for posterior samples. Default 1.0.
        Unlike LinUCB's alpha, this rarely needs tuning.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_arms: int,
        d: int,
        v_sq: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_arms = n_arms
        self.d = d
        self.v_sq = v_sq
        self.arms: list[LinTSArm] = [LinTSArm(d=d) for _ in range(n_arms)]
        self._rng = np.random.default_rng(seed)
        self._t = 0

    def select(self, x: np.ndarray) -> int:
        """Sample from each arm's posterior and return arm with highest score."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        scores = [
            float(arm.sample(self._rng, self.v_sq) @ x_norm)
            for arm in self.arms
        ]
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """Update the selected arm's posterior."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        self.arms[arm].update(x_norm, reward)
        self._t += 1

    def get_state(self) -> dict[str, Any]:
        return {
            "n_arms": self.n_arms,
            "d": self.d,
            "v_sq": self.v_sq,
            "t": self._t,
            "arms": [arm.to_dict() for arm in self.arms],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.n_arms = state["n_arms"]
        self.d = state["d"]
        self.v_sq = state["v_sq"]
        self._t = state["t"]
        self.arms = [LinTSArm.from_dict(a) for a in state["arms"]]


class LinTSPredictor:
    """BasePredictor-compatible wrapper around LinTSRouter.

    Maps model IDs to arm indices (registration order from registry).
    Uses the package's FeatureExtractor for 35-dim feature vectors.

    Parameters
    ----------
    registry : ModelRegistry
        Registered models become arms (indexed by registration order).
    v_sq : float
        Posterior variance scaling. Default 1.0 rarely needs tuning.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        registry: Any,
        v_sq: float = 1.0,
        seed: int = 42,
    ) -> None:
        from routesmith.predictor.base import PredictionResult
        from routesmith.predictor.features import FeatureExtractor

        self._registry = registry
        self._extractor = FeatureExtractor(registry)
        self._PredictionResult = PredictionResult

        models = registry.list_models()
        self._arm_index: dict[str, int] = {m.model_id: i for i, m in enumerate(models)}
        self._arm_names: list[str] = [m.model_id for m in models]
        n_arms = len(models)
        d = 35  # matches FeatureExtractor full output (27 base + 8 context)

        self._router = LinTSRouter(n_arms=n_arms, d=d, v_sq=v_sq, seed=seed)
        self._total_updates = 0

    def _features(self, messages: list[dict], model_id: str, context=None) -> np.ndarray:
        fv = self._extractor.extract(messages, model_id, context=context)
        x = np.array(fv.features[:35], dtype=np.float64)
        if len(x) < 35:
            x = np.pad(x, (0, 35 - len(x)))
        return x  # LinTSRouter.select/update handles normalization internally

    def predict(
        self,
        messages: list[dict],
        model_ids: list[str],
        context=None,
    ) -> list:
        """Sample from each arm's posterior and return ranked predictions."""
        results = []
        for model_id in model_ids:
            arm_idx = self._arm_index.get(model_id)
            if arm_idx is None:
                results.append(self._PredictionResult(
                    model_id=model_id,
                    predicted_quality=0.0,
                    confidence=0.0,
                    metadata={"lints": "unregistered_model"},
                ))
                continue

            x = self._features(messages, model_id, context=context)
            x_norm = x / (np.linalg.norm(x) + 1e-8)
            arm = self._router.arms[arm_idx]
            theta_sample = arm.sample(self._router._rng, self._router.v_sq)
            score = float(theta_sample @ x_norm)

            update_count = int(np.sum(arm.A) - arm.A.shape[0])  # rank-1 updates applied
            results.append(self._PredictionResult(
                model_id=model_id,
                predicted_quality=score,
                confidence=min(1.0, update_count / 20.0),
                metadata={
                    "lints_arm": arm_idx,
                    "lints_score": round(score, 4),
                    "lints_t": self._router._t,
                },
            ))

        return sorted(results, key=lambda r: r.predicted_quality, reverse=True)

    def update(
        self,
        messages: list[dict],
        model_id: str,
        actual_quality: float,
        reward_override: float | None = None,
        context=None,
    ) -> None:
        """Update the arm's Gaussian posterior with observed quality."""
        arm_idx = self._arm_index.get(model_id)
        if arm_idx is None:
            return
        x = self._features(messages, model_id, context=context)
        reward = reward_override if reward_override is not None else actual_quality
        self._router.update(arm=arm_idx, x=x, reward=reward)
        self._total_updates += 1

    def add_arm(self, model_id: str) -> None:
        """Add a new arm with optimistic initialization (A=I, b=0)."""
        if model_id in self._arm_index:
            return
        new_idx = len(self._arm_names)
        self._arm_names.append(model_id)
        self._arm_index[model_id] = new_idx
        self._router.arms.append(LinTSArm(d=self._router.d))
        self._router.n_arms += 1

    def remove_arm(self, model_id: str) -> None:
        """Remove an arm and reindex remaining arms."""
        if model_id not in self._arm_index:
            return
        idx = self._arm_index.pop(model_id)
        self._arm_names.pop(idx)
        self._router.arms.pop(idx)
        self._router.n_arms -= 1
        # Reindex arms that shifted down
        for name in self._arm_names[idx:]:
            self._arm_index[name] -= 1

    def serialize_state(self) -> bytes:
        """Serialize predictor state to JSON bytes."""
        import json
        state = {
            "router_state": self._router.get_state(),
            "arm_names": self._arm_names,
            "arm_index": self._arm_index,
            "total_updates": self._total_updates,
        }
        return json.dumps(state).encode()

    def load_state(self, blob: bytes) -> None:
        """Load predictor state from JSON bytes.

        If stored feature dimension differs from current, skips load (cold start).
        """
        import json
        state = json.loads(blob.decode())
        router_state = state["router_state"]
        stored_d = router_state.get("d", self._router.d)
        if stored_d != self._router.d:
            return  # dimension mismatch — cold start
        self._router.load_state(router_state)
        self._arm_names = state["arm_names"]
        self._arm_index = state["arm_index"]
        self._total_updates = state.get("total_updates", 0)

    def get_state(self) -> dict[str, Any]:
        return {"arm_index": self._arm_index, **self._router.get_state()}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "type": "lints",
            "v_sq": self._router.v_sq,
            "n_arms": self._router.n_arms,
            "d": self._router.d,
            "t": self._router._t,
            "total_updates": self._total_updates,
            "arms": self._arm_names,
        }
