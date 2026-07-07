"""Tests for the matrix factorization EmbeddingPredictor."""
import numpy as np
import pytest

_MSGS = [{"role": "user", "content": "What is the capital of France?"}]
_MSGS2 = [{"role": "user", "content": "Explain quantum computing in simple terms."}]


class TestEmbeddingPredictor:
    """Tests for matrix factorization based quality prediction."""

    @pytest.fixture
    def predictor(self):
        from routesmith.predictor.embedding import EmbeddingPredictor
        return EmbeddingPredictor(
            model_quality_priors={"model-a": 0.8, "model-b": 0.6},
            embedding_dim=384,
            learning_rate=0.01,
            l2_reg=0.001,
        )

    def test_predict_returns_sorted_results(self, predictor):
        """Predict returns results sorted by quality descending."""
        results = predictor.predict(_MSGS, ["model-a", "model-b"])
        assert len(results) == 2
        assert results[0].predicted_quality >= results[1].predicted_quality

    def test_predict_all_requested_models(self, predictor):
        """Predict returns a result for every requested model."""
        results = predictor.predict(_MSGS, ["model-a", "model-b"])
        returned = {r.model_id for r in results}
        assert returned == {"model-a", "model-b"}

    def test_update_changes_model_embedding(self, predictor):
        """Update should modify the model embedding via gradient descent."""
        # Predict first to initialize the model embedding
        predictor.predict(_MSGS, ["model-a"])
        emb_before = predictor._model_embeddings["model-a"].copy()
        predictor.update(_MSGS, "model-a", actual_quality=0.9)
        emb_after = predictor._model_embeddings["model-a"]
        assert not np.allclose(emb_before, emb_after)

    def test_update_increases_confidence(self, predictor):
        """More updates should increase confidence score."""
        results_before = predictor.predict(_MSGS, ["model-a"])
        conf_before = results_before[0].confidence

        for _ in range(10):
            predictor.update(_MSGS, "model-a", actual_quality=0.8)

        results_after = predictor.predict(_MSGS, ["model-a"])
        conf_after = results_after[0].confidence
        assert conf_after >= conf_before

    def test_different_queries_produce_different_predictions(self, predictor):
        """Different queries should result in different quality predictions."""
        r1 = predictor.predict(_MSGS, ["model-a", "model-b"])
        r2 = predictor.predict(_MSGS2, ["model-a", "model-b"])
        # Scores should differ for different queries
        scores_1 = [r.predicted_quality for r in r1]
        scores_2 = [r.predicted_quality for r in r2]
        assert not np.allclose(scores_1, scores_2)

    def test_warm_start_returns_all_models(self, predictor):
        """Cold start predictions return results for all models."""
        results = predictor.predict(_MSGS, ["model-a", "model-b"])
        assert len(results) == 2
        returned = {r.model_id for r in results}
        assert returned == {"model-a", "model-b"}

    def test_predict_returns_scaled_quality(self, predictor):
        """Predicted quality should be in [0, 1] range."""
        results = predictor.predict(_MSGS, ["model-a", "model-b"])
        for r in results:
            assert 0.0 <= r.predicted_quality <= 1.0

    def test_learning_improves_prediction_for_good_model(self, predictor):
        """Repeated positive feedback should increase predicted quality."""
        q_before = predictor.predict(_MSGS, ["model-a"])[0].predicted_quality
        for _ in range(20):
            predictor.update(_MSGS, "model-a", actual_quality=1.0)
        q_after = predictor.predict(_MSGS, ["model-a"])[0].predicted_quality
        assert q_after > q_before

    def test_learning_lowers_prediction_for_bad_model(self, predictor):
        """Repeated negative feedback should decrease predicted quality."""
        q_before = predictor.predict(_MSGS, ["model-a"])[0].predicted_quality
        for _ in range(20):
            predictor.update(_MSGS, "model-a", actual_quality=0.0)
        q_after = predictor.predict(_MSGS, ["model-a"])[0].predicted_quality
        assert q_after <= q_before

    def test_add_arm_initializes_embedding(self, predictor):
        """add_arm should initialize model embedding for new models."""
        predictor.add_arm("model-c", quality_score=0.9)
        assert "model-c" in predictor._model_embeddings
        results = predictor.predict(_MSGS, ["model-c"])
        assert results[0].model_id == "model-c"

    def test_remove_arm_cleans_up(self, predictor):
        """remove_arm should delete model data."""
        predictor.add_arm("model-c")
        predictor.remove_arm("model-c")
        assert "model-c" not in predictor._model_embeddings
        assert "model-c" not in predictor._update_counts
