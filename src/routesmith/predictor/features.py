"""Feature extraction for quality prediction.

Produces a 27-dimensional feature vector per (query, model) pair:
  17 message features + 8 model features + 2 interaction features.

The original 11 message features are preserved for backward compatibility.
New features (indices 11-18) add query type classification, difficulty
estimation, and vocabulary richness signals that improve context-dependent
routing in the bandit predictors.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry


@dataclass
class FeatureVector:
    """Extracted features for a query-model pair."""

    features: list[float] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


# --- Query type keyword sets ---

_MATH_WORDS = frozenset([
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "sum", "product", "factorial", "probability", "percent", "ratio",
    "algebra", "geometry", "trigonometry", "calculus", "matrix", "vector",
    "theorem", "proof", "polynomial", "logarithm", "exponent",
])

_REASONING_WORDS = frozenset([
    "why", "because", "therefore", "however", "although", "explain",
    "reason", "logic", "argument", "conclude", "imply", "infer",
    "deduce", "analyze", "evaluate", "compare", "contrast", "consider",
])

_CODE_WORDS = frozenset([
    "function", "class", "variable", "loop", "array", "string", "integer",
    "implement", "algorithm", "code", "program", "debug", "compile",
    "syntax", "api", "database", "sql", "python", "javascript", "rust",
    "java", "typescript", "html", "css", "git", "docker", "kubernetes",
])

_CREATIVE_WORDS = frozenset([
    "write", "story", "poem", "creative", "imagine", "fiction",
    "character", "narrative", "dialogue", "scene", "metaphor", "lyric",
    "compose", "draft", "essay", "blog", "article", "describe",
])

_RE_CODE_BLOCK = re.compile(r"```")


def _keyword_overlap(words: set[str], target: frozenset[str]) -> float:
    """Fraction of query words found in a keyword set, scaled to [0, 1]."""
    if not words:
        return 0.0
    return min(1.0, len(words & target) / len(words) * 10)


class FeatureExtractor:
    """
    Extracts numeric features from messages and model metadata.

    Produces a 27-dimensional feature vector:
      17 message features + 8 model features + 2 interaction features.

    Original 11 message features are at indices 0-10 (backward compatible).
    New features at indices 11-16 add query type and difficulty signals.
    Interaction features at indices 25-26 combine query and model info.
    """

    MESSAGE_FEATURE_NAMES = [
        # Original 11 features (indices 0-10)
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
        "tools_present",
        # New features (indices 11-16)
        "math_type_score",
        "reasoning_type_score",
        "code_type_score",
        "creative_type_score",
        "difficulty_estimate",
        "vocabulary_richness",
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

    INTERACTION_FEATURE_NAMES = [
        "estimated_response_tokens",
        "difficulty_x_quality_prior",
    ]

    ALL_FEATURE_NAMES = (
        MESSAGE_FEATURE_NAMES + MODEL_FEATURE_NAMES + INTERACTION_FEATURE_NAMES
    )

    def __init__(self, registry: "ModelRegistry") -> None:
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
            FeatureVector with 27 features.
        """
        msg_features = self._extract_message_features(messages)
        model_features = self._extract_model_features(model_id)
        interaction_features = self._extract_interaction_features(
            msg_features, model_features
        )
        return FeatureVector(
            features=msg_features + model_features + interaction_features,
            feature_names=list(self.ALL_FEATURE_NAMES),
        )

    def _extract_message_features(
        self, messages: list[dict[str, str]]
    ) -> list[float]:
        """Extract 17 message-level features."""
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

        # Detect tool usage in the conversation
        tools_present = 1.0 if any(
            m.get("role") == "tool"
            or (m.get("role") == "assistant" and m.get("tool_calls"))
            for m in messages
        ) else 0.0

        # --- New features ---

        all_text = " ".join(m.get("content", "") for m in messages)
        all_words_lower = set(all_text.lower().split())

        # Query type classification (keyword overlap heuristic)
        math_score = _keyword_overlap(all_words_lower, _MATH_WORDS)
        reasoning_score = _keyword_overlap(all_words_lower, _REASONING_WORDS)
        code_score = _keyword_overlap(all_words_lower, _CODE_WORDS)
        creative_score = _keyword_overlap(all_words_lower, _CREATIVE_WORDS)

        # Also boost code score if code blocks present
        has_code = bool(_RE_CODE_BLOCK.search(all_text)) or any(
            k in all_text for k in ("def ", "function ", "class ")
        )
        if has_code:
            code_score = min(1.0, code_score + 0.3)

        # Difficulty estimate: combines length, vocabulary complexity,
        # structural indicators
        length_signal = min(1.0, len(words) / 200.0)
        vocab_signal = min(1.0, avg_word_length / 8.0)
        structure_signal = (
            (0.3 if has_code else 0.0)
            + 0.2 * math_score
            + 0.2 * reasoning_score
        )
        difficulty = min(
            1.0,
            0.3 * length_signal + 0.25 * vocab_signal + 0.35 * structure_signal
            + 0.1 * min(1.0, question_mark_count / 3.0),
        )

        # Vocabulary richness (type-token ratio)
        vocab_richness = (
            len(set(w.lower() for w in words)) / max(1, len(words))
        )

        return [
            # Original 11
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
            tools_present,
            # New 6
            math_score,
            reasoning_score,
            code_score,
            creative_score,
            difficulty,
            vocab_richness,
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

    def _extract_interaction_features(
        self,
        msg_features: list[float],
        model_features: list[float],
    ) -> list[float]:
        """Extract 2 interaction features combining query and model info."""
        word_count = msg_features[8]  # index 8 = word_count
        difficulty = msg_features[15]  # index 15 = difficulty_estimate
        quality_prior = model_features[2]  # index 2 = quality_prior

        # Estimated response tokens: proxy for how long the response will be.
        # Harder/longer queries tend to produce longer responses.
        estimated_response_tokens = min(1.0, (word_count * 2 + difficulty * 500) / 2000.0)

        # Interaction: difficulty * quality_prior
        # High difficulty + high quality prior = this model is a good match
        difficulty_x_quality = difficulty * quality_prior

        return [estimated_response_tokens, difficulty_x_quality]
