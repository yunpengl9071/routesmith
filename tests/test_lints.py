# tests/test_lints.py
"""Unit tests for LinTS-27d posterior math — no API calls needed."""
from __future__ import annotations
import numpy as np
import pytest
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))


def test_lints_arm_init():
    from routesmith.predictor.lints import LinTSArm
    arm = LinTSArm(d=3)
    np.testing.assert_array_equal(arm.A, np.eye(3))
    np.testing.assert_array_equal(arm.b, np.zeros(3))


def test_lints_arm_update():
    """After one update with x=[1,0,0], r=1.0: b=[1,0,0], A=I+xx^T."""
    from routesmith.predictor.lints import LinTSArm
    arm = LinTSArm(d=3)
    x = np.array([1.0, 0.0, 0.0])
    arm.update(x, reward=1.0)
    expected_A = np.eye(3) + np.outer(x, x)
    np.testing.assert_array_almost_equal(arm.A, expected_A)
    np.testing.assert_array_almost_equal(arm.b, x)


def test_lints_arm_mu():
    """mu = A^{-1} b. After x=[1,0,0], r=1: A=diag(2,1,1), b=[1,0,0] → mu=[0.5,0,0]."""
    from routesmith.predictor.lints import LinTSArm
    arm = LinTSArm(d=3)
    x = np.array([1.0, 0.0, 0.0])
    arm.update(x, reward=1.0)
    mu = arm.mu
    assert abs(mu[0] - 0.5) < 1e-9
    assert abs(mu[1]) < 1e-9
    assert abs(mu[2]) < 1e-9


def test_lints_arm_sample_shape():
    from routesmith.predictor.lints import LinTSArm
    arm = LinTSArm(d=5)
    rng = np.random.default_rng(42)
    sample = arm.sample(rng, v_sq=1.0)
    assert sample.shape == (5,)


def test_lints_arm_serialization():
    from routesmith.predictor.lints import LinTSArm
    arm = LinTSArm(d=3)
    x = np.array([1.0, 0.5, 0.0])
    arm.update(x, reward=0.8)
    d = arm.to_dict()
    arm2 = LinTSArm.from_dict(d)
    np.testing.assert_array_almost_equal(arm2.A, arm.A)
    np.testing.assert_array_almost_equal(arm2.b, arm.b)


def test_lints_router_selects_arm():
    """Router should select arm 0 most of the time after training it with high reward."""
    from routesmith.predictor.lints import LinTSRouter
    router = LinTSRouter(n_arms=2, d=2, v_sq=1.0, seed=42)
    x = np.array([1.0, 0.0])
    # Train arm 0 with high reward
    for _ in range(20):
        router.update(arm=0, x=x, reward=1.0)
    # Train arm 1 with low reward
    for _ in range(20):
        router.update(arm=1, x=x, reward=0.0)
    # Arm 0 should be selected >70% of the time
    selections = [router.select(x) for _ in range(50)]
    assert selections.count(0) > 35


def test_lints_router_no_alpha():
    """LinTS has v_sq, not alpha (which is LinUCB's parameter)."""
    from routesmith.predictor.lints import LinTSRouter
    r = LinTSRouter(n_arms=3, d=4)
    assert hasattr(r, 'v_sq')
    assert not hasattr(r, 'alpha')


def test_lints_router_state_roundtrip():
    """Router state can be saved and loaded for resume support."""
    from routesmith.predictor.lints import LinTSRouter
    router = LinTSRouter(n_arms=2, d=3, seed=0)
    x = np.array([1.0, 0.5, 0.0])
    router.update(arm=0, x=x, reward=0.8)
    state = router.get_state()

    router2 = LinTSRouter(n_arms=2, d=3, seed=0)
    router2.load_state(state)
    np.testing.assert_array_almost_equal(
        router2.arms[0].A, router.arms[0].A
    )
    np.testing.assert_array_almost_equal(
        router2.arms[0].b, router.arms[0].b
    )
