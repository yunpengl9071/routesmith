"""Quality signal extraction from LLM responses."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualitySignal:
    """A quality signal extracted from a response."""

    signal_type: str  # "implicit" or "explicit"
    signal_name: str
    signal_value: float  # Normalized 0-1 (1 = good quality)
    raw_value: Any = None
    timestamp: float = field(default_factory=time.time)


# Common refusal patterns across models
_REFUSAL_PATTERNS = [
    r"I(?:'m| am) (?:sorry|unable|not able)",
    r"I can(?:'t|not) (?:help|assist|provide|do that)",
    r"I (?:cannot|can't) (?:fulfill|comply|generate)",
    r"as an AI(?: language model)?",
    r"I(?:'m| am) not (?:allowed|permitted|able) to",
    r"violat(?:es?|ing) (?:my|our) (?:policy|guidelines|terms)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


class SignalExtractor:
    """
    Extracts implicit quality signals from LLM responses.

    Maintains running statistics per model for anomaly detection.
    """

    def __init__(self, model_latency_p95: dict[str, float] | None = None) -> None:
        """
        Initialize signal extractor.

        Args:
            model_latency_p95: Map of model_id -> expected p95 latency (ms)
                from the model registry.
        """
        self._model_latency_p95 = model_latency_p95 or {}
        # Running token count averages per model: {model_id: [counts]}
        self._token_counts: dict[str, list[int]] = defaultdict(list)

    def extract(
        self,
        response: Any,
        model_id: str,
        latency_ms: float,
    ) -> list[QualitySignal]:
        """
        Extract all implicit signals from a response.

        Args:
            response: The LLM response object.
            model_id: The model that generated the response.
            latency_ms: Total latency for the request.

        Returns:
            List of extracted quality signals.
        """
        signals = []

        content = self._get_content(response)
        finish_reason = self._get_finish_reason(response)
        completion_tokens = self._get_completion_tokens(response)

        signals.append(self._check_error(finish_reason))
        signals.append(self._check_refusal(content))
        signals.append(self._check_empty(content))
        signals.append(self._check_truncated(finish_reason))
        signals.append(self._check_length_anomaly(model_id, completion_tokens))
        signals.append(self._check_latency_anomaly(model_id, latency_ms))

        return signals

    def _get_content(self, response: Any) -> str:
        """Extract text content from response."""
        try:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                message = getattr(choices[0], "message", None)
                if message:
                    return getattr(message, "content", "") or ""
        except (IndexError, AttributeError):
            pass
        return ""

    def _get_finish_reason(self, response: Any) -> str | None:
        """Extract finish_reason from response."""
        try:
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                return getattr(choices[0], "finish_reason", None)
        except (IndexError, AttributeError):
            pass
        return None

    def _get_completion_tokens(self, response: Any) -> int:
        """Extract completion token count from response."""
        try:
            usage = getattr(response, "usage", None)
            if usage:
                return getattr(usage, "completion_tokens", 0) or 0
        except AttributeError:
            pass
        return 0

    def _check_error(self, finish_reason: str | None) -> QualitySignal:
        """Check if the response ended with an error."""
        is_error = finish_reason == "error"
        return QualitySignal(
            signal_type="implicit",
            signal_name="error_detected",
            signal_value=0.0 if is_error else 1.0,
            raw_value=finish_reason,
        )

    def _check_refusal(self, content: str) -> QualitySignal:
        """Check if the response contains refusal patterns."""
        is_refusal = bool(_REFUSAL_RE.search(content)) if content else False
        return QualitySignal(
            signal_type="implicit",
            signal_name="refusal_detected",
            signal_value=0.0 if is_refusal else 1.0,
            raw_value=is_refusal,
        )

    def _check_empty(self, content: str) -> QualitySignal:
        """Check if the response is empty."""
        is_empty = not content or not content.strip()
        return QualitySignal(
            signal_type="implicit",
            signal_name="empty_response",
            signal_value=0.0 if is_empty else 1.0,
            raw_value=len(content) if content else 0,
        )

    def _check_truncated(self, finish_reason: str | None) -> QualitySignal:
        """Check if the response was truncated due to length."""
        is_truncated = finish_reason == "length"
        return QualitySignal(
            signal_type="implicit",
            signal_name="truncated_response",
            signal_value=0.0 if is_truncated else 1.0,
            raw_value=finish_reason,
        )

    def _check_length_anomaly(
        self, model_id: str, completion_tokens: int
    ) -> QualitySignal:
        """Check if token count is anomalous vs running average."""
        history = self._token_counts[model_id]

        if len(history) < 5 or completion_tokens == 0:
            # Not enough data for anomaly detection
            if completion_tokens > 0:
                history.append(completion_tokens)
            return QualitySignal(
                signal_type="implicit",
                signal_name="response_length_anomaly",
                signal_value=1.0,  # No anomaly when insufficient data
                raw_value=completion_tokens,
            )

        avg = sum(history) / len(history)
        # Anomalous if > 3x or < 0.1x the average
        ratio = completion_tokens / avg if avg > 0 else 1.0
        is_anomalous = ratio > 3.0 or ratio < 0.1
        # Keep rolling window of last 100
        history.append(completion_tokens)
        if len(history) > 100:
            self._token_counts[model_id] = history[-100:]

        return QualitySignal(
            signal_type="implicit",
            signal_name="response_length_anomaly",
            signal_value=0.5 if is_anomalous else 1.0,
            raw_value={"tokens": completion_tokens, "avg": round(avg, 1), "ratio": round(ratio, 2)},
        )

    def _check_latency_anomaly(
        self, model_id: str, latency_ms: float
    ) -> QualitySignal:
        """Check if latency exceeds model's registered p95."""
        p95 = self._model_latency_p95.get(model_id)
        if p95 is None or p95 <= 0:
            return QualitySignal(
                signal_type="implicit",
                signal_name="latency_anomaly",
                signal_value=1.0,  # No anomaly when no baseline
                raw_value=latency_ms,
            )

        is_anomalous = latency_ms > p95
        # Scale: at p95 = 1.0, at 2*p95 = 0.5, at 3*p95 = 0.0
        if is_anomalous:
            value = max(0.0, 1.0 - (latency_ms - p95) / p95)
        else:
            value = 1.0

        return QualitySignal(
            signal_type="implicit",
            signal_name="latency_anomaly",
            signal_value=round(value, 3),
            raw_value={"latency_ms": latency_ms, "p95_ms": p95},
        )
