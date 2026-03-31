"""WarmStartLinUCB — LinUCB with offline pre-initialization.

Extends LinUCBPredictor with the ability to pre-train the A and b matrices
from a set of labeled (messages, model_id, quality) examples before going
online. This gives it an advantage over cold-start LinUCB (and over RouteLLM)
by combining supervised initialization with online bandit exploration.

Typical usage:
    1. Collect ~500 labeled examples from RouteLLM's public Chatbot Arena data
       or from a held-out evaluation set.
    2. Call warm_start(examples) to pre-initialize the arm matrices.
    3. Deploy online — the predictor continues learning via LinUCB updates.

The warm-start simply runs the standard LinUCB update rule on each labeled
example, building up the A and b matrices as if the examples had been
observed online. This preserves the theoretical properties of LinUCB
(sub-linear regret) while giving a much better starting point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from routesmith.predictor.linucb import LinUCBPredictor

if TYPE_CHECKING:
    from routesmith.registry.models import ModelRegistry

logger = logging.getLogger(__name__)


class WarmStartLinUCBPredictor(LinUCBPredictor):
    """LinUCB with offline warm-start from labeled examples.

    Inherits all behavior from LinUCBPredictor. Adds:
    - warm_start(): pre-initialize A/b matrices from labeled data
    - Tracks warm-start statistics for diagnostics

    Parameters
    ----------
    registry : ModelRegistry
        For feature extraction and cost lookups.
    alpha : float
        UCB exploration parameter. Default 1.5.
    cost_lambda : float
        Cost penalty weight. Default 0.3.
    latency_lambda : float
        Latency penalty weight. Default 0.1.
    warmup_rounds : int
        After warm-start, this can be set to 0 since arms are pre-initialized.
        Default 0.
    """

    def __init__(
        self,
        registry: "ModelRegistry",
        alpha: float = 1.5,
        cost_lambda: float = 0.3,
        latency_lambda: float = 0.1,
        warmup_rounds: int = 0,
    ) -> None:
        super().__init__(
            registry=registry,
            alpha=alpha,
            cost_lambda=cost_lambda,
            warmup_rounds=warmup_rounds,
        )
        self._latency_lambda = latency_lambda
        self._warm_start_count = 0
        self._warm_started = False

        # Latency normalization
        models = registry.list_models()
        lats = [m.latency_p50_ms for m in models]
        self._max_latency = max(lats) if lats else 1.0

    def warm_start(
        self,
        examples: list[tuple[list[dict[str, str]], str, float]],
        epochs: int = 1,
    ) -> None:
        """Pre-initialize arm matrices from labeled examples.

        Each example is a (messages, model_id, quality_score) tuple.
        The standard LinUCB update rule is applied to each example,
        building up the A and b matrices.

        Multiple epochs replay the data to strengthen the initialization.
        With 500 examples and 1 epoch, each arm typically gets ~150-200
        updates (assuming ~3 models), which is enough for stable theta.

        Args:
            examples: List of (messages, model_id, quality_score) tuples.
            epochs: Number of passes over the examples. Default 1.
        """
        count = 0
        for _ in range(epochs):
            for messages, model_id, quality_score in examples:
                if self._registry.get(model_id) is None:
                    continue
                # Use the parent's update() which handles feature extraction,
                # cost penalty, Sherman-Morrison update, etc.
                self.update(messages, model_id, quality_score)
                count += 1

        self._warm_start_count = count
        self._warm_started = True
        logger.info(
            "WarmStartLinUCB: initialized with %d updates from %d examples "
            "(%d epochs)",
            count, len(examples), epochs,
        )

    def update(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        actual_quality: float,
    ) -> None:
        """Update with latency penalty added to the reward.

        Extends LinUCB's update to include latency in the reward signal:
            reward = quality - cost_lambda * norm_cost - latency_lambda * norm_latency
        """
        # We override to add latency penalty. The parent uses:
        #   reward = quality - cost_lambda * norm_cost
        # We need to adjust the quality before passing to parent so that
        # the effective reward includes latency.
        model_config = self._registry.get(model_id)
        latency_adjustment = 0.0
        if model_config is not None and self._max_latency > 0:
            norm_latency = model_config.latency_p50_ms / self._max_latency
            latency_adjustment = self._latency_lambda * norm_latency

        # Pass adjusted quality to parent's update
        adjusted_quality = actual_quality - latency_adjustment
        super().update(messages, model_id, adjusted_quality)

    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostics with warm-start info."""
        diag = super().diagnostics()
        diag["type"] = "warmstart_linucb"
        diag["latency_lambda"] = self._latency_lambda
        diag["warm_started"] = self._warm_started
        diag["warm_start_updates"] = self._warm_start_count
        return diag
