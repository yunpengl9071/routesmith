"""Quality poll generation and adaptive sampling.

Polls collect user feedback on LLM responses to feed into the bandit predictor
for per-agent quality fine-tuning. Sampling adapts to posterior uncertainty —
new agents get polled more, converged agents less.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PollOption:
    """A single poll option with ID and label."""

    id: int
    label: str


@dataclass
class Poll:
    """A quality poll attached to a routed response."""

    id: str
    type: str
    question: str
    options: list[PollOption]
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "id": self.id,
            "type": self.type,
            "question": self.question,
            "options": [asdict(o) for o in self.options],
            "context": self.context,
        }


@dataclass
class PollAnswer:
    """A user's answer to a quality poll."""

    poll_id: str
    option: int
    timestamp: float


def generate_poll(
    request_id: str,
    model_id: str,
    cost_usd: float,
) -> Poll:
    """Generate a quality poll for a response.

    Args:
        request_id: The request ID this poll is associated with.
        model_id: The model that generated the response.
        cost_usd: Estimated cost of the response.

    Returns:
        Poll with 5 numbered options (Good, wrong output, too slow,
        too expensive, too verbose) and context metadata.
    """
    return Poll(
        id=request_id,
        type="numbered",
        question="How was this response?",
        options=[
            PollOption(id=1, label="Good"),
            PollOption(id=2, label="Not good — wrong output"),
            PollOption(id=3, label="Not good — too slow"),
            PollOption(id=4, label="Not good — too expensive"),
            PollOption(id=5, label="Not good — too verbose"),
        ],
        context={
            "model": model_id,
            "cost": f"${cost_usd:.4f}",
            "request_id": request_id,
        },
    )


class PollSampler:
    """Adaptive sampling rate based on agent convergence.

    New agents (convergence=0) get sampled at base_rate.
    Converged agents (convergence=1) get sampled at 0%.
    """

    def __init__(self, base_rate: float = 0.1) -> None:
        self._base_rate = base_rate
        self._agent_sample_count: dict[str, int] = {}

    def should_sample(self, agent_id: str | None, convergence: float) -> bool:
        """Determine if we should poll this response.

        Args:
            agent_id: Agent identifier (for per-agent tracking).
            convergence: Agent convergence score 0-1 (0=new, 1=converged).

        Returns:
            True if a poll should be generated for this response.
        """
        effective_rate = self._base_rate * (1.0 - convergence)
        return random.random() < effective_rate


class PollSignalMapper:
    """Maps poll option IDs to quality feedback signals.

    The signal contains a quality score (0-1) and a human-readable reason.
    These are fed into the bandit predictor's quality_threshold adjustment.
    """

    SIGNAL_MAP: dict[int, dict[str, Any]] = {
        1: {"quality": 1.0, "reason": "quality_good"},
        2: {"quality": 0.0, "reason": "quality_low"},
        3: {"quality": 0.5, "reason": "too_slow"},
        4: {"quality": 0.3, "reason": "too_expensive"},
        5: {"quality": 0.4, "reason": "too_verbose"},
    }

    @staticmethod
    def map(option: int) -> dict[str, Any] | None:
        """Map a poll option ID to a quality feedback signal.

        Args:
            option: The poll option ID (1-5).

        Returns:
            Signal dict with 'quality' and 'reason', or None if unknown.
        """
        return PollSignalMapper.SIGNAL_MAP.get(option)
