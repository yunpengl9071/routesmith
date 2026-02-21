"""A/B test framework for comparing routing strategies."""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from routesmith.client import RouteSmith


@dataclass
class VariantStats:
    """Statistics for a single A/B test variant."""

    name: str
    request_count: int
    total_cost_usd: float
    avg_cost_per_request: float
    quality_scores: list[float]
    avg_quality: float | None
    models_used: dict[str, int]


@dataclass
class ABTestResults:
    """Aggregated A/B test results with comparison."""

    variant_a: VariantStats
    variant_b: VariantStats
    winner: str | None  # "A", "B", or None if insufficient data
    quality_diff: float | None  # B - A
    cost_diff_pct: float | None  # (B - A) / A * 100


MIN_SAMPLES_FOR_WINNER = 30


class ABTestRunner:
    """
    A/B test runner that compares two RouteSmith configurations.

    Randomly assigns each request to variant A or B based on traffic_split,
    tracks outcomes per variant, and provides statistical comparison.

    Example:
        >>> client_a = RouteSmith(config_a)
        >>> client_b = RouteSmith(config_b)
        >>> ab = ABTestRunner(client_a, client_b, traffic_split=0.5)
        >>> response = ab.completion(messages=[{"role": "user", "content": "Hello"}])
        >>> results = ab.results()
    """

    def __init__(
        self,
        client_a: RouteSmith,
        client_b: RouteSmith,
        traffic_split: float = 0.5,
        name: str = "",
    ) -> None:
        """
        Initialize A/B test runner.

        Args:
            client_a: RouteSmith client for variant A (control).
            client_b: RouteSmith client for variant B (experiment).
            traffic_split: Fraction of requests routed to variant B (0.0-1.0).
            name: Optional name for this test.
        """
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("traffic_split must be between 0.0 and 1.0")

        self.client_a = client_a
        self.client_b = client_b
        self.traffic_split = traffic_split
        self.name = name

        self._request_map: dict[str, str] = {}  # request_id -> "A" or "B"
        self._variant_costs: dict[str, float] = {"A": 0.0, "B": 0.0}
        self._variant_counts: dict[str, int] = {"A": 0, "B": 0}
        self._variant_quality: dict[str, list[float]] = {"A": [], "B": []}
        self._variant_models: dict[str, dict[str, int]] = {
            "A": defaultdict(int),
            "B": defaultdict(int),
        }

    def _assign_variant(self) -> str:
        """Randomly assign a request to variant A or B."""
        return "B" if random.random() < self.traffic_split else "A"

    def _get_client(self, variant: str) -> RouteSmith:
        """Get the client for a given variant."""
        return self.client_b if variant == "B" else self.client_a

    def _track_response(self, variant: str, response: Any) -> None:
        """Track cost and model usage from a response."""
        request_id = getattr(response, "_routesmith_request_id", None)
        if request_id:
            self._request_map[request_id] = variant

        self._variant_counts[variant] += 1

        # Track cost from the client's last routing metadata
        client = self._get_client(variant)
        metadata = client.last_routing_metadata
        if metadata:
            self._variant_costs[variant] += metadata.estimated_cost_usd
            self._variant_models[variant][metadata.model_selected] += 1

    def completion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        Execute a completion request, randomly assigning to variant A or B.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional arguments passed to the client's completion().

        Returns:
            ModelResponse from the selected variant's client.
        """
        variant = self._assign_variant()
        client = self._get_client(variant)
        response = client.completion(messages=messages, **kwargs)
        self._track_response(variant, response)
        return response

    async def acompletion(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        Async version of completion().

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional arguments passed to the client's acompletion().

        Returns:
            ModelResponse from the selected variant's client.
        """
        variant = self._assign_variant()
        client = self._get_client(variant)
        response = await client.acompletion(messages=messages, **kwargs)
        self._track_response(variant, response)
        return response

    def record_outcome(
        self,
        request_id: str,
        success: bool | None = None,
        score: float | None = None,
        feedback: str | None = None,
    ) -> bool:
        """
        Record feedback for a previous request, routing to the correct variant's client.

        Args:
            request_id: Request ID from response._routesmith_request_id.
            success: Whether the response was successful.
            score: Explicit quality score (0-1).
            feedback: Free-text user feedback.

        Returns:
            True if the request was found, False otherwise.
        """
        variant = self._request_map.get(request_id)
        if variant is None:
            return False

        client = self._get_client(variant)
        found = client.record_outcome(
            request_id=request_id,
            success=success,
            score=score,
            feedback=feedback,
        )

        if found and score is not None:
            self._variant_quality[variant].append(score)

        return found

    def _build_variant_stats(self, variant: str) -> VariantStats:
        """Build stats for a single variant."""
        count = self._variant_counts[variant]
        total_cost = self._variant_costs[variant]
        quality_scores = self._variant_quality[variant]

        return VariantStats(
            name=variant,
            request_count=count,
            total_cost_usd=round(total_cost, 6),
            avg_cost_per_request=round(total_cost / count, 6) if count > 0 else 0.0,
            quality_scores=list(quality_scores),
            avg_quality=(
                round(sum(quality_scores) / len(quality_scores), 4)
                if quality_scores
                else None
            ),
            models_used=dict(self._variant_models[variant]),
        )

    def results(self) -> ABTestResults:
        """
        Get aggregated A/B test results with comparison.

        Returns:
            ABTestResults with per-variant stats, winner determination,
            quality difference, and cost difference percentage.
        """
        stats_a = self._build_variant_stats("A")
        stats_b = self._build_variant_stats("B")

        # Determine winner based on avg quality
        winner: str | None = None
        quality_diff: float | None = None

        if (
            stats_a.avg_quality is not None
            and stats_b.avg_quality is not None
            and len(self._variant_quality["A"]) >= MIN_SAMPLES_FOR_WINNER
            and len(self._variant_quality["B"]) >= MIN_SAMPLES_FOR_WINNER
        ):
            quality_diff = round(stats_b.avg_quality - stats_a.avg_quality, 4)
            if stats_b.avg_quality > stats_a.avg_quality:
                winner = "B"
            elif stats_a.avg_quality > stats_b.avg_quality:
                winner = "A"
            # Equal quality → no winner

        # Cost difference percentage
        cost_diff_pct: float | None = None
        if stats_a.avg_cost_per_request > 0:
            cost_diff_pct = round(
                (stats_b.avg_cost_per_request - stats_a.avg_cost_per_request)
                / stats_a.avg_cost_per_request
                * 100,
                2,
            )

        return ABTestResults(
            variant_a=stats_a,
            variant_b=stats_b,
            winner=winner,
            quality_diff=quality_diff,
            cost_diff_pct=cost_diff_pct,
        )

    def reset(self) -> None:
        """Reset all A/B test state."""
        self._request_map.clear()
        self._variant_costs = {"A": 0.0, "B": 0.0}
        self._variant_counts = {"A": 0, "B": 0}
        self._variant_quality = {"A": [], "B": []}
        self._variant_models = {"A": defaultdict(int), "B": defaultdict(int)}
