"""Random forest quality model with lazy sklearn import."""

from __future__ import annotations

from typing import Any

from routesmith.predictor.features import FeatureVector


class QualityModel:
    """
    Random forest regressor for quality prediction.

    Wraps sklearn RandomForestRegressor with lazy import to avoid
    hard dependency on sklearn at import time.
    """

    def __init__(self, n_estimators: int = 50) -> None:
        self._n_estimators = n_estimators
        self._model: Any = None
        self._is_trained = False

    def _get_rf_class(self) -> type:
        """Lazy import RandomForestRegressor."""
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor

    def fit(self, features: list[list[float]], targets: list[float]) -> None:
        """
        Train the model on labeled data.

        Args:
            features: List of feature vectors (each a list of floats).
            targets: Continuous quality targets in [0, 1].
        """
        RFR = self._get_rf_class()
        self._model = RFR(
            n_estimators=self._n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        )
        self._model.fit(features, targets)
        self._is_trained = True

    def predict(self, feature_vector: FeatureVector) -> tuple[float, float]:
        """
        Predict quality and confidence for a single feature vector.

        Args:
            feature_vector: Extracted features for a query-model pair.

        Returns:
            (predicted_quality, confidence) where confidence = 1 - std across trees.

        Raises:
            RuntimeError: If model has not been trained.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("QualityModel has not been trained. Call fit() first.")

        import numpy as np

        X = np.array([feature_vector.features])
        # Get individual tree predictions for confidence
        tree_preds = np.array([
            tree.predict(X)[0] for tree in self._model.estimators_
        ])
        predicted = float(np.mean(tree_preds))
        std = float(np.std(tree_preds))
        confidence = max(0.0, min(1.0, 1.0 - std))

        # Clamp predicted quality to [0, 1]
        predicted = max(0.0, min(1.0, predicted))

        return predicted, confidence

    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self._is_trained

    @property
    def feature_importances(self) -> list[float] | None:
        """Feature importances from the trained model, or None."""
        if self._is_trained and self._model is not None:
            return self._model.feature_importances_.tolist()
        return None

    def get_state(self) -> dict[str, Any]:
        """Serialize model state for persistence."""
        import pickle
        state: dict[str, Any] = {
            "n_estimators": self._n_estimators,
            "is_trained": self._is_trained,
        }
        if self._is_trained and self._model is not None:
            state["model_bytes"] = pickle.dumps(self._model)
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> QualityModel:
        """Restore model from serialized state."""
        import pickle
        obj = cls(n_estimators=state["n_estimators"])
        obj._is_trained = state["is_trained"]
        if obj._is_trained and "model_bytes" in state:
            obj._model = pickle.loads(state["model_bytes"])
        return obj
