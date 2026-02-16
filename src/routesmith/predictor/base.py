"""Base quality predictor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    """Result from quality prediction."""

    model_id: str
    predicted_quality: float  # 0-1 score
    confidence: float  # Confidence in prediction 0-1
    metadata: dict[str, Any] | None = None


class BasePredictor(ABC):
    """
    Abstract base class for quality predictors.

    Quality predictors estimate expected response quality for a given
    query-model pair before execution, enabling intelligent routing.
    """

    @abstractmethod
    def predict(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
    ) -> list[PredictionResult]:
        """
        Predict quality for each model given the input messages.

        Args:
            messages: Input messages for the query.
            model_ids: List of model IDs to evaluate.

        Returns:
            List of PredictionResult for each model, sorted by predicted_quality descending.
        """
        pass

    @abstractmethod
    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
        """
        Update predictor with observed quality feedback.

        Args:
            messages: Input messages that were used.
            model_id: Model that generated the response.
            actual_quality: Observed quality score 0-1.
        """
        pass

    def predict_best(
        self,
        messages: list[dict[str, str]],
        model_ids: list[str],
        min_quality: float = 0.0,
    ) -> PredictionResult | None:
        """
        Get the best model prediction meeting quality threshold.

        Args:
            messages: Input messages for the query.
            model_ids: List of model IDs to evaluate.
            min_quality: Minimum acceptable quality score.

        Returns:
            Best prediction meeting threshold, or None if none qualify.
        """
        predictions = self.predict(messages, model_ids)
        for pred in predictions:
            if pred.predicted_quality >= min_quality:
                return pred
        return None
