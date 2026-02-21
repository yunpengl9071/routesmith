"""Adaptive predictor with cold start, warm-up, and learned phases."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from routesmith.predictor.base import BasePredictor, PredictionResult
from routesmith.predictor.features import FeatureExtractor

if TYPE_CHECKING:
    from routesmith.feedback.storage import FeedbackStorage
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


class AdaptivePredictor(BasePredictor):
    """
    Quality predictor that transitions through three phases:

        cold_start  -->  warm_up  -->  learned
        (priors)     (EMA priors)   (RF + prior blend)

    Designed for a future contextual bandit exploration layer.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        storage: FeedbackStorage | None = None,
        min_samples: int = 100,
        retrain_interval: int = 50,
        blend_alpha: float = 0.7,
        n_estimators: int = 50,
    ) -> None:
        self._registry = registry
        self._storage = storage
        self._min_samples = min_samples
        self._retrain_interval = retrain_interval
        self._blend_alpha = blend_alpha
        self._n_estimators = n_estimators

        # Phase tracking
        self._phase = "cold_start"
        self._update_count = 0
        self._samples_since_retrain = 0

        # EMA quality priors per model
        self._ema_priors: dict[str, float] = {
            m.model_id: m.quality_score for m in registry.list_models()
        }

        # Feature extractor
        self._extractor = FeatureExtractor(registry)

        # Quality model (lazy-created on first retrain)
        self._model: Any = None  # QualityModel instance

    @property
    def phase(self) -> str:
        """Current predictor phase."""
        return self._phase

    def predict(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> list[PredictionResult]:
        """Predict quality for each model given input messages."""
        results: list[PredictionResult] = []

        for model_id in model_ids:
            quality, confidence, meta = self._predict_single(
                messages, model_id
            )
            results.append(
                PredictionResult(
                    model_id=model_id,
                    predicted_quality=quality,
                    confidence=confidence,
                    metadata=meta,
                )
            )

        return sorted(
            results, key=lambda r: r.predicted_quality, reverse=True
        )

    def _predict_single(
        self,
        messages: list[dict[str, str]],
        model_id: str,
    ) -> tuple[float, float, dict[str, Any]]:
        """Predict for a single model, returning (quality, confidence, metadata)."""
        ema_prior = self._get_ema_prior(model_id)
        meta: dict[str, Any] = {
            "exploration_score": 0.0,
            "phase": self._phase,
        }

        if self._phase == "cold_start":
            return ema_prior, 0.3, meta

        if self._phase == "warm_up":
            return ema_prior, 0.4, meta

        # learned phase: blend RF prediction with EMA prior
        if self._model is not None and self._model.is_trained():
            fv = self._extractor.extract(messages, model_id)
            try:
                rf_pred, rf_conf = self._model.predict(fv)
                # Blend: conf * rf + (1-conf) * prior
                blended = rf_conf * rf_pred + (1.0 - rf_conf) * ema_prior
                meta["rf_prediction"] = rf_pred
                meta["rf_confidence"] = rf_conf
                return blended, rf_conf, meta
            except Exception:
                logger.warning(
                    "RF prediction failed, falling back to prior",
                    exc_info=True,
                )
                return ema_prior, 0.4, meta

        # Fallback if model disappeared somehow
        return ema_prior, 0.4, meta

    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
        """Update with observed quality feedback."""
        # EMA update
        alpha = 0.1
        current = self._get_ema_prior(model_id)
        self._ema_priors[model_id] = alpha * actual_quality + (1 - alpha) * current

        self._update_count += 1
        self._samples_since_retrain += 1

        # Transition from cold_start to warm_up on first update
        if self._phase == "cold_start":
            self._phase = "warm_up"

        # Attempt retrain when enough samples
        if self._phase == "warm_up" and self._update_count >= self._min_samples:
            self._maybe_retrain()
        elif self._phase == "learned" and self._samples_since_retrain >= self._retrain_interval:
            self._maybe_retrain()

    def force_retrain(self) -> bool:
        """Force a retrain attempt. Returns True if successful."""
        return self._maybe_retrain()

    def _maybe_retrain(self) -> bool:
        """Pull training data, extract features, fit RF. Returns True on success."""
        if self._storage is None:
            return False

        try:
            from routesmith.predictor.model import QualityModel
        except Exception:
            logger.warning("Could not import QualityModel", exc_info=True)
            return False

        try:
            raw_data = self._storage.get_training_data()
            if len(raw_data) < self._min_samples:
                return False

            features_list: list[list[float]] = []
            targets: list[float] = []

            for record in raw_data:
                messages = record.get("messages", [])
                model_id = record.get("model_id", "")
                quality = record.get("quality_score")
                if quality is None:
                    continue
                fv = self._extractor.extract(messages, model_id)
                features_list.append(fv.features)
                targets.append(float(quality))

            if len(targets) < self._min_samples:
                return False

            model = QualityModel(n_estimators=self._n_estimators)
            model.fit(features_list, targets)
            self._model = model
            self._phase = "learned"
            self._samples_since_retrain = 0
            logger.info(
                "Retrained RF model with %d samples, phase=learned",
                len(targets),
            )
            return True

        except Exception:
            logger.warning("Retrain failed, staying in %s", self._phase, exc_info=True)
            return False

    def _get_ema_prior(self, model_id: str) -> float:
        """Get EMA prior for a model, falling back to registry or 0.5."""
        if model_id in self._ema_priors:
            return self._ema_priors[model_id]
        model = self._registry.get(model_id)
        if model is not None:
            return model.quality_score
        return 0.5

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostic info about the predictor state."""
        diag: dict[str, Any] = {
            "phase": self._phase,
            "update_count": self._update_count,
            "samples_since_retrain": self._samples_since_retrain,
            "min_samples": self._min_samples,
            "retrain_interval": self._retrain_interval,
            "ema_priors": dict(self._ema_priors),
            "model_trained": self._model is not None and self._model.is_trained(),
        }
        if self._model is not None and self._model.is_trained():
            diag["feature_importances"] = self._model.feature_importances
        return diag
