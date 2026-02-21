"""Feedback collection for routing improvement."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from routesmith.config import RouteSmithConfig
from routesmith.feedback.signals import QualitySignal, SignalExtractor
from routesmith.feedback.storage import FeedbackStorage

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry


@dataclass
class FeedbackRecord:
    """Record of a completion with feedback."""

    request_id: str
    messages: list[dict[str, str]]
    model_id: str
    response: Any
    latency_ms: float
    timestamp: float
    quality_score: float | None = None
    user_feedback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackCollector:
    """
    Collects feedback on routing decisions for continuous improvement.

    Supports:
    - Automatic sampling of requests for evaluation
    - Manual quality scoring
    - User feedback collection
    - Integration with LLM-as-judge evaluators
    - Implicit signal extraction from responses
    - SQLite persistence for training data
    """

    def __init__(
        self,
        config: RouteSmithConfig,
        registry: ModelRegistry | None = None,
        max_records: int = 10000,
    ) -> None:
        """
        Initialize feedback collector.

        Args:
            config: RouteSmith configuration.
            registry: Model registry (enables signal extraction with latency baselines).
            max_records: Maximum feedback records to retain in memory.
        """
        self.config = config
        self.max_records = max_records
        self._records: list[FeedbackRecord] = []
        self._request_index: dict[str, FeedbackRecord] = {}
        self._quality_evaluator: Callable[[FeedbackRecord], float] | None = None

        # Initialize storage if configured
        self._storage: FeedbackStorage | None = None
        if config.feedback_storage_path:
            self._storage = FeedbackStorage(config.feedback_storage_path)

        # Initialize signal extractor if registry provided
        self._signal_extractor: SignalExtractor | None = None
        if registry is not None:
            latency_p95 = {
                m.model_id: m.latency_p99_ms
                for m in registry.list_models()
            }
            self._signal_extractor = SignalExtractor(model_latency_p95=latency_p95)

    def record(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        model: str,
        response: Any,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord | None:
        """
        Record a completion for potential feedback collection.

        Based on sample_rate, may not store every request.

        Args:
            request_id: Unique request identifier.
            messages: Input messages.
            model: Model that was used.
            response: Model response.
            latency_ms: Completion latency.
            metadata: Additional metadata.

        Returns:
            FeedbackRecord if sampled, None otherwise.
        """
        if not self.config.feedback_enabled:
            return None

        # Probabilistic sampling
        if random.random() > self.config.feedback_sample_rate:
            return None

        record = FeedbackRecord(
            request_id=request_id,
            messages=messages,
            model_id=model,
            response=response,
            latency_ms=latency_ms,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Auto-evaluate if evaluator is set
        if self._quality_evaluator:
            record.quality_score = self._quality_evaluator(record)

        # Store in memory
        self._records.append(record)
        self._request_index[request_id] = record

        # Evict old records if over capacity
        if len(self._records) > self.max_records:
            evicted = self._records[: -self.max_records]
            for r in evicted:
                self._request_index.pop(r.request_id, None)
            self._records = self._records[-self.max_records :]

        # Persist to SQLite if enabled
        if self._storage is not None:
            self._storage.store_record(
                request_id=request_id,
                model_id=model,
                messages=messages,
                latency_ms=latency_ms,
                quality_score=record.quality_score,
                metadata=metadata,
            )

        # Extract implicit signals
        if self._signal_extractor is not None:
            signals = self._signal_extractor.extract(response, model, latency_ms)
            if self._storage is not None:
                for sig in signals:
                    self._storage.store_signal(
                        request_id=request_id,
                        signal_type=sig.signal_type,
                        signal_name=sig.signal_name,
                        signal_value=sig.signal_value,
                        raw_value=sig.raw_value,
                    )

        return record

    def record_outcome(
        self,
        request_id: str,
        success: bool | None = None,
        score: float | None = None,
        feedback: str | None = None,
    ) -> bool:
        """
        Record explicit user feedback for a previous request.

        Args:
            request_id: The request ID to attach feedback to.
            success: Whether the response was successful (converted to 1.0/0.0).
            score: Explicit quality score (0-1).
            feedback: Free-text user feedback.

        Returns:
            True if the request was found and updated, False otherwise.
        """
        quality = score
        if quality is None and success is not None:
            quality = 1.0 if success else 0.0

        # Update in-memory record
        record = self._request_index.get(request_id)
        if record is not None:
            if quality is not None:
                record.quality_score = quality
            if feedback is not None:
                record.user_feedback = feedback

        # Update in storage
        if self._storage is not None:
            self._storage.update_record(
                request_id=request_id,
                quality_score=quality,
                user_feedback=feedback,
            )
            # Store explicit signal
            if quality is not None:
                self._storage.store_signal(
                    request_id=request_id,
                    signal_type="explicit",
                    signal_name="user_quality_score",
                    signal_value=quality,
                    raw_value={"success": success, "score": score, "feedback": feedback},
                )

        return record is not None

    def add_quality_score(
        self,
        record_index: int,
        score: float,
    ) -> None:
        """
        Add quality score to a feedback record.

        Args:
            record_index: Index of the record.
            score: Quality score 0-1.
        """
        if 0 <= record_index < len(self._records):
            self._records[record_index].quality_score = score

    def add_user_feedback(
        self,
        record_index: int,
        feedback: str,
    ) -> None:
        """
        Add user feedback to a record.

        Args:
            record_index: Index of the record.
            feedback: User feedback string.
        """
        if 0 <= record_index < len(self._records):
            self._records[record_index].user_feedback = feedback

    def set_quality_evaluator(
        self,
        evaluator: Callable[[FeedbackRecord], float],
    ) -> None:
        """
        Set automatic quality evaluator.

        Args:
            evaluator: Function that takes FeedbackRecord and returns quality 0-1.
        """
        self._quality_evaluator = evaluator

    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get aggregated statistics per model.

        Returns:
            Dict mapping model_id to stats (avg_quality, avg_latency, count).
        """
        from collections import defaultdict

        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"quality_scores": [], "latencies": [], "count": 0}
        )

        for record in self._records:
            model_stats = stats[record.model_id]
            model_stats["count"] += 1
            model_stats["latencies"].append(record.latency_ms)
            if record.quality_score is not None:
                model_stats["quality_scores"].append(record.quality_score)

        result = {}
        for model_id, model_stats in stats.items():
            latencies = model_stats["latencies"]
            qualities = model_stats["quality_scores"]
            result[model_id] = {
                "count": model_stats["count"],
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "avg_quality": sum(qualities) / len(qualities) if qualities else None,
                "quality_samples": len(qualities),
            }
        return result

    def get_records(
        self,
        model_id: str | None = None,
        min_quality: float | None = None,
        max_quality: float | None = None,
    ) -> list[FeedbackRecord]:
        """
        Get filtered feedback records.

        Args:
            model_id: Filter by model.
            min_quality: Minimum quality score.
            max_quality: Maximum quality score.

        Returns:
            Filtered list of records.
        """
        records = self._records

        if model_id:
            records = [r for r in records if r.model_id == model_id]

        if min_quality is not None:
            records = [
                r for r in records
                if r.quality_score is not None and r.quality_score >= min_quality
            ]

        if max_quality is not None:
            records = [
                r for r in records
                if r.quality_score is not None and r.quality_score <= max_quality
            ]

        return records

    def get_record_by_id(self, request_id: str) -> FeedbackRecord | None:
        """Look up a feedback record by request ID."""
        return self._request_index.get(request_id)

    def export_training_data(self) -> list[dict[str, Any]]:
        """
        Export records as training data for predictor improvement.

        Returns:
            List of dicts suitable for training quality predictors.
        """
        return [
            {
                "messages": r.messages,
                "model_id": r.model_id,
                "quality_score": r.quality_score,
                "latency_ms": r.latency_ms,
            }
            for r in self._records
            if r.quality_score is not None
        ]

    def clear(self) -> None:
        """Clear all feedback records."""
        self._records.clear()
        self._request_index.clear()

    def __len__(self) -> int:
        return len(self._records)
