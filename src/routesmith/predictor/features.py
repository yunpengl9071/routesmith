"""Feature extraction for quality prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry


@dataclass
class FeatureVector:
    """Extracted features for a query-model pair."""

    features: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


class FeatureExtractor:
    """
    Extracts numeric features from messages and model metadata.

    Produces an 18-dimensional feature vector:
      10 message features + 8 model features.
    All features are cheap to compute (<1ms, no external deps).
    """

    MESSAGE_FEATURE_NAMES = [
        "msg_count",
        "total_char_length",
        "avg_msg_length",
        "max_msg_length",
        "user_msg_count",
        "system_msg_present",
        "last_msg_length",
        "question_mark_count",
        "word_count",
        "avg_word_length",
    ]

    MODEL_FEATURE_NAMES = [
        "cost_per_1k_input",
        "cost_per_1k_output",
        "quality_prior",
        "latency_p50_ms",
        "context_window_log",
        "supports_function_calling",
        "supports_vision",
        "supports_json_mode",
    ]

    ALL_FEATURE_NAMES = MESSAGE_FEATURE_NAMES + MODEL_FEATURE_NAMES

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def extract(
        self,
        messages: list[dict[str, str]],
        model_id: str,
    ) -> FeatureVector:
        """
        Extract features from messages and model metadata.

        Args:
            messages: Input messages for the query.
            model_id: Model to extract metadata features for.

        Returns:
            FeatureVector with 18 features.
        """
        msg_features = self._extract_message_features(messages)
        model_features = self._extract_model_features(model_id)
        return FeatureVector(
            features=msg_features + model_features,
            feature_names=list(self.ALL_FEATURE_NAMES),
        )

    def _extract_message_features(
        self, messages: list[dict[str, str]]
    ) -> list[float]:
        """Extract 10 message-level features."""
        if not messages:
            return [0.0] * len(self.MESSAGE_FEATURE_NAMES)

        msg_count = float(len(messages))
        lengths = [len(m.get("content", "")) for m in messages]
        total_char_length = float(sum(lengths))
        avg_msg_length = total_char_length / msg_count if msg_count else 0.0
        max_msg_length = float(max(lengths)) if lengths else 0.0

        user_msgs = [m for m in messages if m.get("role") == "user"]
        user_msg_count = float(len(user_msgs))
        system_msg_present = 1.0 if any(
            m.get("role") == "system" for m in messages
        ) else 0.0

        # Last user message features
        last_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_content = m.get("content", "")
                break

        last_msg_length = float(len(last_user_content))
        question_mark_count = float(last_user_content.count("?"))

        words = last_user_content.split()
        word_count = float(len(words))
        avg_word_length = (
            sum(len(w) for w in words) / word_count if words else 0.0
        )

        return [
            msg_count,
            total_char_length,
            avg_msg_length,
            max_msg_length,
            user_msg_count,
            system_msg_present,
            last_msg_length,
            question_mark_count,
            word_count,
            avg_word_length,
        ]

    def _extract_model_features(self, model_id: str) -> list[float]:
        """Extract 8 model-level features."""
        model = self._registry.get(model_id)
        if model is None:
            # Unknown model — return neutral defaults
            return [
                0.001,   # cost_per_1k_input
                0.002,   # cost_per_1k_output
                0.5,     # quality_prior
                500.0,   # latency_p50_ms
                math.log(128000),  # context_window_log
                1.0,     # supports_function_calling
                0.0,     # supports_vision
                1.0,     # supports_json_mode
            ]

        return [
            model.cost_per_1k_input,
            model.cost_per_1k_output,
            model.quality_score,
            model.latency_p50_ms,
            math.log(max(model.context_window, 1)),
            1.0 if model.supports_function_calling else 0.0,
            1.0 if model.supports_vision else 0.0,
            1.0 if model.supports_json_mode else 0.0,
        ]
