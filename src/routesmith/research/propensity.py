"""Propensity logging + exploration floor for off-policy evaluation (WU-0.2).

Wraps a ``LinTSRouter`` so every routing decision is:

* **propensity-logged** — the decision-time probability ``p(a | x)`` of *every* arm,
  under the exact Thompson-sampling policy plus an enforced exploration floor;
* **exactly replayable** — reproducible from the logged posterior state + a per-decision
  seed, which the study's G1 gate requires for shadow replay.

Thompson-sampling propensities have no closed form, so ``p_TS(a | x)`` is estimated by
Monte Carlo. The key simplification: for a linear-TS arm the *score*
``θ̃_a · x̂`` (with ``θ̃_a ~ N(μ_a, v²Σ_a)``) is **univariate Gaussian**,
``N(μ_a·x̂, v²·x̂ᵀΣ_a x̂)`` — so we never draw full ``d``-dimensional samples; we draw one
scalar per arm per MC replicate and take the argmax. This reproduces ``LinTSRouter.select``
exactly and runs in well under a millisecond at 10⁴ replicates.

The logged policy is the ε-floored mixture

    p(a | x) = (1 − ε) · p_TS(a | x) + ε / K     with per-arm floor ε/K > 0,

which guarantees positivity (support for IPS/DR) and — because the chosen arm is sampled
*from this same logged vector* — makes the logged propensity exactly the sampling
probability (calibrated by construction, up to MC error in ``p_TS``).

Nothing here modifies ``LinTSRouter``; it is a read-only instrument built on top.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from routesmith.predictor.lints import LinTSRouter

# Schema/version tag stamped on every decision so logs remain interpretable across
# instrument revisions. Bump on any breaking change to RoutingDecision fields.
PROPENSITY_SCHEMA_VERSION = "v2.1"
DEFAULT_MC_SAMPLES = 10_000


@dataclass(frozen=True)
class ExplorationFloor:
    """Enforced ε-exploration floor over ``k`` arms.

    Guarantees every arm keeps probability ≥ ``epsilon / k`` at decision time, so logged
    bandit feedback satisfies positivity and off-policy estimators have support.
    """

    epsilon: float = 0.10
    k: int = 5

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")

    @property
    def per_arm_floor(self) -> float:
        """Minimum decision-time probability of any arm: ε/K."""
        return self.epsilon / self.k

    def apply(self, p_ts: np.ndarray) -> np.ndarray:
        """Floor a Thompson-sampling distribution into the logged mixture."""
        if p_ts.shape[0] != self.k:
            raise ValueError(f"expected {self.k} arms, got {p_ts.shape[0]}")
        return (1.0 - self.epsilon) * p_ts + self.epsilon / self.k


@dataclass
class RoutingDecision:
    """A single propensity-logged routing decision (one logged bandit-feedback context).

    Mirrors the study logging schema (episode_v2_1): the reward ``r`` is attached later,
    when the routed episode's outcome is known.
    """

    chosen_arm: int
    propensity: float                 # p(chosen | x) — the logged mixture probability
    propensity_vector: list[float]    # p(a | x) for all arms; sums to 1
    exploration_flag: bool            # True iff the ε-floor (uniform component) fired
    n_arms: int
    epsilon: float
    policy_version: str
    posterior_seed: int               # per-decision seed → exact shadow replay
    posterior_state_hash: str         # sha256 of the router posterior (provenance)
    mc_samples: int
    ts_propensity_vector: list[float]  # p_TS(a | x) before flooring (diagnostic)
    schema_version: str = PROPENSITY_SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)
    arm_names: list[str] | None = None

    def to_schema_dict(self) -> dict[str, Any]:
        """Row shaped for the study's episode schema / DecisionLog persistence."""
        return asdict(self)


class PropensityLogger:
    """Turns a ``LinTSRouter`` into a propensity-logging, exactly-replayable policy.

    Parameters
    ----------
    router:
        The live ``LinTSRouter`` (its posterior is read, never mutated).
    floor:
        The exploration floor to enforce.
    policy_version:
        Identifier logged with every decision (e.g. a checkpoint hash / epoch).
    mc_samples:
        Monte-Carlo replicates for the propensity estimate (default 10⁴; MC s.e. ≈
        ``sqrt(p(1-p)/N)`` ≈ 0.005 at p=0.5).
    """

    def __init__(
        self,
        router: LinTSRouter,
        floor: ExplorationFloor,
        policy_version: str = "",
        mc_samples: int = DEFAULT_MC_SAMPLES,
        arm_names: list[str] | None = None,
    ) -> None:
        if router.n_arms != floor.k:
            raise ValueError(
                f"router has {router.n_arms} arms but floor.k={floor.k}; they must match"
            )
        self.router = router
        self.floor = floor
        self.policy_version = policy_version
        self.mc_samples = mc_samples
        self.arm_names = arm_names

    # ---- construction from the production predictor --------------------------
    @classmethod
    def from_predictor(
        cls,
        predictor: Any,
        floor: ExplorationFloor,
        policy_version: str = "",
        mc_samples: int = DEFAULT_MC_SAMPLES,
    ) -> "PropensityLogger":
        """Build from a ``LinTSPredictor`` (uses its underlying router + arm names)."""
        return cls(
            router=predictor._router,
            floor=floor,
            policy_version=policy_version,
            mc_samples=mc_samples,
            arm_names=list(getattr(predictor, "_arm_names", []) or []),
        )

    # ---- posterior score moments (the univariate reduction) -----------------
    def _score_moments(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-arm mean/std of the TS score ``θ̃_a · x̂``.

        ``θ̃_a ~ N(μ_a, v²Σ_a)`` ⇒ score ~ ``N(μ_a·x̂, v²·x̂ᵀΣ_a x̂)``.
        Reproduces ``LinTSRouter.select`` (same normalization) without d-dim draws.
        """
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        means = np.empty(self.router.n_arms)
        stds = np.empty(self.router.n_arms)
        v_sq = self.router.v_sq
        for i, arm in enumerate(self.router.arms):
            mu = arm.mu
            sigma = arm.sigma
            means[i] = float(mu @ x_norm)
            var = float(v_sq * (x_norm @ sigma @ x_norm))
            stds[i] = float(np.sqrt(max(var, 0.0)))
        return means, stds

    def ts_propensities(self, x: np.ndarray, seed: Any) -> np.ndarray:
        """Monte-Carlo estimate of ``p_TS(a | x)`` = P(arm a has the top TS score)."""
        means, stds = self._score_moments(x)
        rng = np.random.default_rng(seed)
        # (mc_samples, K) scores = mean + std * Z  ;  argmax per replicate
        z = rng.standard_normal((self.mc_samples, self.router.n_arms))
        scores = means[None, :] + stds[None, :] * z
        winners = np.argmax(scores, axis=1)
        counts = np.bincount(winners, minlength=self.router.n_arms)
        return counts / self.mc_samples

    # ---- provenance ---------------------------------------------------------
    def posterior_state_hash(self) -> str:
        """Stable sha256 over the router's posterior (A, b per arm) + config."""
        h = hashlib.sha256()
        h.update(json.dumps({
            "v_sq": self.router.v_sq, "d": self.router.d, "t": self.router._t,
            "arms": [{"A": arm.A.tolist(), "b": arm.b.tolist()} for arm in self.router.arms],
        }, sort_keys=True).encode())
        return h.hexdigest()

    # ---- the decision -------------------------------------------------------
    def decide(self, x: np.ndarray, decision_seed: int) -> RoutingDecision:
        """Make and fully log one routing decision.

        Two independent RNG streams are derived from ``decision_seed`` (via SeedSequence),
        one for the MC propensity estimate and one for the action draw, so the *entire*
        decision — propensity vector and chosen arm — is bit-exactly reproducible from
        ``(posterior state, decision_seed)``.
        """
        mc_seed, act_seed = np.random.SeedSequence(decision_seed).spawn(2)
        p_ts = self.ts_propensities(x, mc_seed)
        p = self.floor.apply(p_ts)
        p = p / p.sum()  # guard against MC rounding; keeps it a proper distribution

        act_rng = np.random.default_rng(act_seed)
        if act_rng.random() < self.floor.epsilon:
            exploration_flag = True
            chosen = int(act_rng.integers(self.router.n_arms))  # uniform floor draw
        else:
            exploration_flag = False
            chosen = int(act_rng.choice(self.router.n_arms, p=p_ts))  # TS component
        # By construction P(chosen=a) = ε/K + (1-ε)·p_TS(a) = p(a): logged prob is exact.

        return RoutingDecision(
            chosen_arm=chosen,
            propensity=float(p[chosen]),
            propensity_vector=[float(v) for v in p],
            exploration_flag=exploration_flag,
            n_arms=int(self.router.n_arms),
            epsilon=float(self.floor.epsilon),
            policy_version=self.policy_version,
            posterior_seed=int(decision_seed),
            posterior_state_hash=self.posterior_state_hash(),
            mc_samples=int(self.mc_samples),
            ts_propensity_vector=[float(v) for v in p_ts],
            arm_names=self.arm_names,
        )


def shadow_replay(
    router_state: dict[str, Any],
    x: np.ndarray,
    decision_seed: int,
    floor: ExplorationFloor,
    mc_samples: int = DEFAULT_MC_SAMPLES,
    policy_version: str = "",
) -> RoutingDecision:
    """Reconstruct a decision from a logged posterior state + seed.

    Rebuilds a ``LinTSRouter`` from ``router_state`` (as produced by ``get_state``) and
    re-runs ``decide`` with the same seed. The result must match the original decision
    bit-for-bit — the exact-replay guarantee the G1 logger gate checks.
    """
    router = LinTSRouter(
        n_arms=router_state["n_arms"], d=router_state["d"],
        v_sq=router_state["v_sq"],
    )
    router.load_state(router_state)
    logger = PropensityLogger(router, floor, policy_version=policy_version,
                              mc_samples=mc_samples)
    return logger.decide(x, decision_seed)


class DecisionLog:
    """SQLite sink for routing decisions (its own table; never touches core storage).

    Point it at the feedback DB file to co-locate, or a separate file. Rewards are
    attached later by ``attach_reward`` when the routed episode's outcome is verified.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS routing_decisions (
            decision_id      TEXT PRIMARY KEY,
            request_id       TEXT,
            task_id          TEXT,
            chosen_arm       INTEGER NOT NULL,
            propensity       REAL NOT NULL,
            propensity_vector_json TEXT NOT NULL,
            ts_propensity_vector_json TEXT NOT NULL,
            exploration_flag INTEGER NOT NULL,
            n_arms           INTEGER NOT NULL,
            epsilon          REAL NOT NULL,
            policy_version   TEXT,
            posterior_seed   INTEGER NOT NULL,
            posterior_state_hash TEXT NOT NULL,
            mc_samples       INTEGER NOT NULL,
            schema_version   TEXT NOT NULL,
            reward           REAL,
            outcome          TEXT,
            created_at       REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rd_task ON routing_decisions(task_id);
        CREATE INDEX IF NOT EXISTS idx_rd_policy ON routing_decisions(policy_version);
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def log(
        self,
        decision: RoutingDecision,
        decision_id: str,
        task_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO routing_decisions
               (decision_id, request_id, task_id, chosen_arm, propensity,
                propensity_vector_json, ts_propensity_vector_json, exploration_flag,
                n_arms, epsilon, policy_version, posterior_seed, posterior_state_hash,
                mc_samples, schema_version, reward, outcome, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (decision_id, request_id, task_id, decision.chosen_arm, decision.propensity,
             json.dumps(decision.propensity_vector),
             json.dumps(decision.ts_propensity_vector),
             int(decision.exploration_flag), decision.n_arms, decision.epsilon,
             decision.policy_version, decision.posterior_seed,
             decision.posterior_state_hash, decision.mc_samples, decision.schema_version,
             None, None, decision.timestamp),
        )
        self._conn.commit()

    def attach_reward(self, decision_id: str, reward: float, outcome: str) -> None:
        """Attach the verified episode outcome once it is known."""
        self._conn.execute(
            "UPDATE routing_decisions SET reward=?, outcome=? WHERE decision_id=?",
            (reward, outcome, decision_id),
        )
        self._conn.commit()

    def export_episodes(self) -> list[dict[str, Any]]:
        """Return all logged decisions as (x, a, p, r)-style rows for OPE."""
        rows = self._conn.execute("SELECT * FROM routing_decisions").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["propensity_vector"] = json.loads(d.pop("propensity_vector_json"))
            d["ts_propensity_vector"] = json.loads(d.pop("ts_propensity_vector_json"))
            d["exploration_flag"] = bool(d["exploration_flag"])
            out.append(d)
        return out

    def close(self) -> None:
        self._conn.close()
