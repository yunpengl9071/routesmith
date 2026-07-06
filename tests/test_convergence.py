"""Tests for bandit convergence on synthetic workloads (G4)."""

from routesmith.predictor.lints import LinTSPredictor
from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.registry.models import ModelRegistry


MATH = [{"role": "user", "content": "solve the integral of x^2 dx"}]
CHAT = [{"role": "user", "content": "write a short friendly greeting"}]

ARCHETYPES = [MATH, CHAT]

GROUND_TRUTH = {
    "gpt-4o": {0: 0.95, 1: 0.55},
    "gpt-4o-mini": {0: 0.45, 1: 0.90},
}


def _make_registry():
    registry = ModelRegistry()
    registry.register(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
    )
    registry.register(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
    )
    return registry


def _optimal_arm(archetype_idx: int) -> str:
    return "gpt-4o" if archetype_idx == 0 else "gpt-4o-mini"


def test_lints_converges_on_synthetic_workload():
    """G4: >=80% optimal-arm selection within 150 feedback events."""
    registry = _make_registry()
    predictor = LinTSPredictor(registry, seed=42)
    model_ids = ["gpt-4o", "gpt-4o-mini"]

    for i in range(150):
        archetype_idx = i % 2
        msgs = ARCHETYPES[archetype_idx]
        results = predictor.predict(msgs, model_ids)
        chosen = results[0].model_id
        reward = GROUND_TRUTH[chosen][archetype_idx]
        predictor.update(msgs, chosen, actual_quality=reward)

    correct = 0
    for i in range(100):
        archetype_idx = i % 2
        msgs = ARCHETYPES[archetype_idx]
        results = predictor.predict(msgs, model_ids)
        chosen = results[0].model_id
        expected = _optimal_arm(archetype_idx)
        if chosen == expected:
            correct += 1

    assert correct >= 80, f"LinTS correct={correct}, expected >=80"


def test_linucb_converges_on_synthetic_workload():
    """G4: >=80% optimal-arm selection within 150 feedback events."""
    registry = _make_registry()
    predictor = LinUCBPredictor(registry, alpha=1.5)
    model_ids = ["gpt-4o", "gpt-4o-mini"]

    for i in range(150):
        archetype_idx = i % 2
        msgs = ARCHETYPES[archetype_idx]
        results = predictor.predict(msgs, model_ids)
        chosen = results[0].model_id
        reward = GROUND_TRUTH[chosen][archetype_idx]
        predictor.update(msgs, chosen, actual_quality=reward)

    correct = 0
    for i in range(100):
        archetype_idx = i % 2
        msgs = ARCHETYPES[archetype_idx]
        results = predictor.predict(msgs, model_ids)
        chosen = results[0].model_id
        expected = _optimal_arm(archetype_idx)
        if chosen == expected:
            correct += 1

    assert correct >= 80, f"LinUCB correct={correct}, expected >=80"
