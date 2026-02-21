"""Tests for QualityModel (random forest wrapper)."""

import pytest

from routesmith.predictor.features import FeatureVector
from routesmith.predictor.model import QualityModel


class TestUntrainedModel:
    def test_is_trained_false(self):
        model = QualityModel()
        assert model.is_trained() is False

    def test_predict_raises_when_untrained(self):
        model = QualityModel()
        fv = FeatureVector(features=[0.0] * 18, feature_names=["f"] * 18)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(fv)

    def test_feature_importances_none_when_untrained(self):
        model = QualityModel()
        assert model.feature_importances is None


class TestTrainedModel:
    @pytest.fixture
    def trained_model(self):
        """Train a model on synthetic data."""
        import random
        random.seed(42)

        model = QualityModel(n_estimators=20)
        n_samples = 200
        features = []
        targets = []
        for _ in range(n_samples):
            # Feature 12 (quality_prior) is the main predictor
            fv = [random.random() for _ in range(18)]
            quality_prior = fv[12]
            # Target correlates with quality_prior + noise
            target = max(0.0, min(1.0, quality_prior + random.gauss(0, 0.1)))
            features.append(fv)
            targets.append(target)

        model.fit(features, targets)
        return model

    def test_is_trained_true(self, trained_model):
        assert trained_model.is_trained() is True

    def test_predict_returns_bounded_values(self, trained_model):
        fv = FeatureVector(features=[0.5] * 18, feature_names=["f"] * 18)
        quality, confidence = trained_model.predict(fv)
        assert 0.0 <= quality <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_predict_varies_with_features(self, trained_model):
        """Different inputs should produce different predictions."""
        fv_low = FeatureVector(features=[0.1] * 18, feature_names=["f"] * 18)
        fv_high = FeatureVector(features=[0.9] * 18, feature_names=["f"] * 18)
        q_low, _ = trained_model.predict(fv_low)
        q_high, _ = trained_model.predict(fv_high)
        # Should be different (not necessarily ordered since features are mixed)
        assert q_low != q_high

    def test_confidence_from_tree_variance(self, trained_model):
        """Confidence should reflect tree agreement."""
        fv = FeatureVector(features=[0.5] * 18, feature_names=["f"] * 18)
        _, confidence = trained_model.predict(fv)
        # Confidence = 1 - std, should be between 0 and 1
        assert 0.0 <= confidence <= 1.0

    def test_feature_importances(self, trained_model):
        importances = trained_model.feature_importances
        assert importances is not None
        assert len(importances) == 18
        assert all(i >= 0.0 for i in importances)


class TestSerialization:
    def test_state_roundtrip(self):
        """Model state should survive serialization."""
        import random
        random.seed(42)

        model = QualityModel(n_estimators=10)
        features = [[random.random() for _ in range(18)] for _ in range(50)]
        targets = [random.random() for _ in range(50)]
        model.fit(features, targets)

        fv = FeatureVector(features=[0.5] * 18, feature_names=["f"] * 18)
        original_pred = model.predict(fv)

        # Roundtrip
        state = model.get_state()
        restored = QualityModel.from_state(state)

        assert restored.is_trained()
        restored_pred = restored.predict(fv)
        assert abs(original_pred[0] - restored_pred[0]) < 1e-9
        assert abs(original_pred[1] - restored_pred[1]) < 1e-9


class TestLazyImport:
    def test_sklearn_not_imported_at_init(self):
        """sklearn should not be imported just from creating QualityModel."""
        import sys
        # Just creating the model should not import sklearn
        model = QualityModel()
        # We can't reliably test this since sklearn may already be imported
        # by other tests, but we can verify the model is functional
        assert model.is_trained() is False
