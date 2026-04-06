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
    def from_dict(cls, data: dict[str, Any]) -> "LinTSArm":
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
        Feature dimension (27 for RouteSmith's full feature space).
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
