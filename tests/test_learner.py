"""Tests for AdaptivePredictor phases and transitions."""

import json
import random

import pytest

from routesmith.feedback.storage import FeedbackStorage
from routesmith.predictor.learner import AdaptivePredictor
from routesmith.registry.models import ModelRegistry


@pytest.fixture
def registry():
    reg = ModelRegistry()
    reg.register(
        "expensive",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        quality_score=0.95,
    )
    reg.register(
        "cheap",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0003,
        quality_score=0.70,
    )
    reg.register(
        "medium",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.003,
        quality_score=0.85,
    )
    return reg


@pytest.fixture
def storage():
    return FeedbackStorage(":memory:")


@pytest.fixture
def predictor(registry, storage):
    return AdaptivePredictor(
        registry=registry,
        storage=storage,
        min_samples=10,  # Low for testing
        retrain_interval=5,
    )


def _seed_storage(storage, registry, n=20):
    """Insert n training records into storage."""
    random.seed(42)
    models = [m.model_id for m in registry.list_models()]
    quality_map = {m.model_id: m.quality_score for m in registry.list_models()}
    for i in range(n):
        model_id = models[i % len(models)]
        msgs = [{"role": "user", "content": f"Query {i}"}]
        quality = max(0.0, min(1.0, quality_map[model_id] + random.gauss(0, 0.1)))
        storage.store_record(
            request_id=f"req-{i}",
            model_id=model_id,
            messages=msgs,
            latency_ms=100.0,
            quality_score=quality,
        )


class TestColdStart:
    def test_initial_phase(self, predictor):
        assert predictor.phase == "cold_start"

    def test_cold_start_returns_priors(self, predictor):
        msgs = [{"role": "user", "content": "Hello"}]
        results = predictor.predict(msgs, ["expensive", "cheap"])
        # Should return registry priors
        expensive = next(r for r in results if r.model_id == "expensive")
        cheap = next(r for r in results if r.model_id == "cheap")
        assert expensive.predicted_quality == 0.95
        assert cheap.predicted_quality == 0.70

    def test_cold_start_confidence(self, predictor):
        results = predictor.predict(
            [{"role": "user", "content": "Hi"}], ["expensive"]
        )
        assert results[0].confidence == 0.3

    def test_cold_start_metadata_has_exploration_score(self, predictor):
        results = predictor.predict(
            [{"role": "user", "content": "Hi"}], ["expensive"]
        )
        assert results[0].metadata is not None
        assert results[0].metadata["exploration_score"] == 0.0
        assert results[0].metadata["phase"] == "cold_start"


class TestWarmUp:
    def test_transitions_to_warm_up_after_first_update(self, predictor):
        assert predictor.phase == "cold_start"
        predictor.update(
            [{"role": "user", "content": "Hi"}], "expensive", 0.9
        )
        assert predictor.phase == "warm_up"

    def test_warm_up_returns_ema_adjusted_priors(self, predictor):
        predictor.update(
            [{"role": "user", "content": "Hi"}], "expensive", 0.5
        )
        assert predictor.phase == "warm_up"

        results = predictor.predict(
            [{"role": "user", "content": "Hi"}], ["expensive"]
        )
        # EMA: 0.1 * 0.5 + 0.9 * 0.95 = 0.905
        assert abs(results[0].predicted_quality - 0.905) < 0.01

    def test_warm_up_confidence(self, predictor):
        predictor.update(
            [{"role": "user", "content": "Hi"}], "expensive", 0.9
        )
        results = predictor.predict(
            [{"role": "user", "content": "Hi"}], ["expensive"]
        )
        assert results[0].confidence == 0.4


class TestLearned:
    def test_transitions_to_learned(self, registry, storage):
        """After min_samples updates with storage data, should transition to learned."""
        _seed_storage(storage, registry, n=20)

        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
            retrain_interval=5,
        )

        # Feed enough updates to trigger retrain
        for i in range(10):
            predictor.update(
                [{"role": "user", "content": f"Query {i}"}],
                "expensive",
                0.9,
            )

        assert predictor.phase == "learned"

    def test_learned_uses_rf_prediction(self, registry, storage):
        _seed_storage(storage, registry, n=20)

        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
            retrain_interval=5,
        )

        for i in range(10):
            predictor.update(
                [{"role": "user", "content": f"Q {i}"}], "expensive", 0.9
            )

        results = predictor.predict(
            [{"role": "user", "content": "Test query"}],
            ["expensive", "cheap"],
        )
        # Should have RF metadata
        for r in results:
            assert r.metadata is not None
            assert r.metadata["phase"] == "learned"
            assert "rf_prediction" in r.metadata
            assert "rf_confidence" in r.metadata

    def test_blending_formula(self, registry, storage):
        """Verify blended = conf * rf + (1-conf) * prior."""
        _seed_storage(storage, registry, n=20)

        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
            retrain_interval=5,
        )

        for i in range(10):
            predictor.update(
                [{"role": "user", "content": f"Q {i}"}], "expensive", 0.9
            )

        results = predictor.predict(
            [{"role": "user", "content": "Test"}], ["expensive"]
        )
        r = results[0]
        rf_pred = r.metadata["rf_prediction"]
        rf_conf = r.metadata["rf_confidence"]
        ema_prior = predictor._ema_priors["expensive"]
        expected = rf_conf * rf_pred + (1.0 - rf_conf) * ema_prior
        assert abs(r.predicted_quality - expected) < 1e-9

    def test_periodic_retrain(self, registry, storage):
        _seed_storage(storage, registry, n=20)

        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
            retrain_interval=5,
        )

        # Get to learned phase
        for i in range(10):
            predictor.update(
                [{"role": "user", "content": f"Q {i}"}], "expensive", 0.9
            )
        assert predictor.phase == "learned"

        # Add more data to storage for retrain
        _seed_storage(storage, registry, n=30)

        # Trigger periodic retrain (5 more updates)
        for i in range(5):
            predictor.update(
                [{"role": "user", "content": f"New {i}"}], "cheap", 0.6
            )

        # Should still be learned (retrain succeeded or at least attempted)
        assert predictor.phase == "learned"


class TestForceRetrain:
    def test_force_retrain_with_data(self, registry, storage):
        _seed_storage(storage, registry, n=20)

        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
        )

        result = predictor.force_retrain()
        assert result is True
        assert predictor.phase == "learned"

    def test_force_retrain_without_data(self, registry, storage):
        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
        )
        result = predictor.force_retrain()
        assert result is False

    def test_force_retrain_without_storage(self, registry):
        predictor = AdaptivePredictor(registry=registry, storage=None)
        result = predictor.force_retrain()
        assert result is False


class TestGracefulFallback:
    def test_no_storage(self, registry):
        """Predictor works without storage (stays in warm_up)."""
        predictor = AdaptivePredictor(registry=registry, storage=None)
        predictor.update(
            [{"role": "user", "content": "Hi"}], "expensive", 0.9
        )
        assert predictor.phase == "warm_up"

        results = predictor.predict(
            [{"role": "user", "content": "Hi"}], ["expensive"]
        )
        assert len(results) == 1
        assert results[0].predicted_quality > 0.0

    def test_insufficient_storage_data(self, registry, storage):
        """Stays in warm_up if not enough data."""
        _seed_storage(storage, registry, n=3)

        predictor = AdaptivePredictor(
            registry=registry,
            storage=storage,
            min_samples=10,
        )

        for i in range(10):
            predictor.update(
                [{"role": "user", "content": f"Q {i}"}], "expensive", 0.9
            )

        # Not enough storage data to train (only 3 records)
        assert predictor.phase == "warm_up"


class TestDiagnostics:
    def test_diagnostics_cold_start(self, predictor):
        diag = predictor.diagnostics()
        assert diag["phase"] == "cold_start"
        assert diag["update_count"] == 0
        assert diag["model_trained"] is False
        assert "ema_priors" in diag

    def test_diagnostics_after_training(self, registry, storage):
        _seed_storage(storage, registry, n=20)
        predictor = AdaptivePredictor(
            registry=registry, storage=storage, min_samples=10
        )
        predictor.force_retrain()
        diag = predictor.diagnostics()
        assert diag["phase"] == "learned"
        assert diag["model_trained"] is True
        assert "feature_importances" in diag


class TestSortOrder:
    def test_predictions_sorted_by_quality(self, predictor):
        """Predictions should be sorted by predicted_quality descending."""
        results = predictor.predict(
            [{"role": "user", "content": "Hi"}],
            ["expensive", "cheap", "medium"],
        )
        qualities = [r.predicted_quality for r in results]
        assert qualities == sorted(qualities, reverse=True)
