# tests/test_propensity.py
"""Unit tests for the propensity-logging instrument (WU-0.2) — no API calls.

Proves the properties the study's G1 logger gate depends on:
  * propensities are a valid distribution and respect the ε/K floor;
  * the ε-floor mixture is exactly what is logged (calibration by construction);
  * decisions replay bit-exactly from logged posterior state + seed;
  * logged propensities match realized action frequencies (empirical calibration);
  * MC propensities reproduce the router's own select() frequencies.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from routesmith.predictor.lints import LinTSRouter
from routesmith.research.propensity import (
    DecisionLog,
    ExplorationFloor,
    PropensityLogger,
    RoutingDecision,
    shadow_replay,
)

K = 5
D = 8


def _trained_router(seed: int = 0, updates: int = 40) -> LinTSRouter:
    """A router with a non-uniform posterior so arms are genuinely distinguishable."""
    r = LinTSRouter(n_arms=K, d=D, v_sq=1.0, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(updates):
        x = rng.normal(size=D)
        arm = int(rng.integers(K))
        reward = 1.0 if arm in (0, 1) else 0.2  # make arms 0,1 look good
        r.update(arm, x, reward)
    return r


def _logger(router=None, epsilon=0.10, mc=20000):
    router = router or _trained_router()
    return PropensityLogger(router, ExplorationFloor(epsilon=epsilon, k=K),
                            policy_version="test-v1", mc_samples=mc)


class TestExplorationFloor:
    def test_per_arm_floor(self):
        assert ExplorationFloor(epsilon=0.10, k=5).per_arm_floor == pytest.approx(0.02)

    def test_apply_is_a_distribution_with_floor(self):
        floor = ExplorationFloor(epsilon=0.10, k=K)
        p_ts = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
        p = floor.apply(p_ts)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= floor.per_arm_floor - 1e-12).all()

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ExplorationFloor(epsilon=1.5, k=5)
        with pytest.raises(ValueError):
            ExplorationFloor(epsilon=0.1, k=0)


class TestPropensities:
    def test_vector_is_valid_distribution(self):
        d = _logger().decide(np.ones(D), decision_seed=7)
        v = np.array(d.propensity_vector)
        assert v.sum() == pytest.approx(1.0, abs=1e-9)
        assert (v > 0).all()  # positivity from the floor

    def test_respects_floor(self):
        lg = _logger(epsilon=0.10)
        d = lg.decide(np.ones(D), decision_seed=1)
        assert min(d.propensity_vector) >= lg.floor.per_arm_floor - 1e-9

    def test_logged_propensity_matches_chosen(self):
        d = _logger().decide(np.ones(D), decision_seed=3)
        assert d.propensity == pytest.approx(d.propensity_vector[d.chosen_arm])

    def test_mc_matches_router_select_frequencies(self):
        """MC propensities must equal the frequencies the router's own select() produces
        (no floor) — this is the exactness of the univariate reduction."""
        router = _trained_router(seed=2)
        lg = PropensityLogger(router, ExplorationFloor(epsilon=0.0, k=K), mc_samples=40000)
        x = np.random.default_rng(9).normal(size=D)
        p_mc = lg.ts_propensities(x, seed=123)

        # empirical select() frequencies over many fresh samples
        counts = np.zeros(K)
        r2 = _trained_router(seed=2)
        for _ in range(40000):
            counts[r2.select(x)] += 1
        p_emp = counts / counts.sum()
        assert np.max(np.abs(p_mc - p_emp)) < 0.02


class TestExactReplay:
    def test_decision_replays_bit_exactly(self):
        router = _trained_router(seed=5)
        lg = PropensityLogger(router, ExplorationFloor(epsilon=0.10, k=K),
                              policy_version="v1", mc_samples=10000)
        x = np.random.default_rng(11).normal(size=D)
        original = lg.decide(x, decision_seed=424242)

        replayed = shadow_replay(router.get_state(), x, decision_seed=424242,
                                 floor=ExplorationFloor(epsilon=0.10, k=K),
                                 mc_samples=10000, policy_version="v1")
        assert replayed.chosen_arm == original.chosen_arm
        assert replayed.propensity_vector == original.propensity_vector
        assert replayed.posterior_state_hash == original.posterior_state_hash

    def test_hash_changes_after_update(self):
        router = _trained_router(seed=6)
        lg = _logger(router)
        h1 = lg.posterior_state_hash()
        router.update(0, np.ones(D), 1.0)
        assert lg.posterior_state_hash() != h1


class TestEmpiricalCalibration:
    def test_action_frequencies_match_logged_propensities(self):
        """The realized frequency of each arm over many decisions on the SAME context
        must match the logged propensity vector — the property the G1 gate checks."""
        lg = _logger(epsilon=0.10, mc=20000)
        x = np.random.default_rng(4).normal(size=D)
        ref = lg.decide(x, decision_seed=0).propensity_vector
        counts = np.zeros(K)
        n = 6000
        for s in range(n):
            counts[lg.decide(x, decision_seed=1000 + s).chosen_arm] += 1
        freq = counts / n
        # within MC + sampling error (2/sqrt(n) ~ 0.026)
        assert np.max(np.abs(freq - np.array(ref))) < 0.03

    def test_exploration_flag_fires_at_epsilon_rate(self):
        lg = _logger(epsilon=0.10, mc=4000)
        x = np.ones(D)
        flags = [lg.decide(x, decision_seed=s).exploration_flag for s in range(3000)]
        assert np.mean(flags) == pytest.approx(0.10, abs=0.02)


class TestDecisionLog:
    def test_log_export_and_reward_roundtrip(self):
        lg = _logger()
        dlog = DecisionLog(":memory:")
        d = lg.decide(np.ones(D), decision_seed=2)
        dlog.log(d, decision_id="ep-001", task_id="task-A")
        dlog.attach_reward("ep-001", reward=1.0, outcome="pass")
        rows = dlog.export_episodes()
        assert len(rows) == 1
        row = rows[0]
        assert row["task_id"] == "task-A"
        assert row["chosen_arm"] == d.chosen_arm
        assert row["reward"] == 1.0 and row["outcome"] == "pass"
        assert len(row["propensity_vector"]) == K
        assert abs(sum(row["propensity_vector"]) - 1.0) < 1e-9
        dlog.close()


class TestSchema:
    def test_to_schema_dict_has_required_fields(self):
        d = _logger().decide(np.ones(D), decision_seed=1)
        row = d.to_schema_dict()
        for f in ("chosen_arm", "propensity", "propensity_vector", "exploration_flag",
                  "policy_version", "posterior_seed", "posterior_state_hash",
                  "schema_version"):
            assert f in row
