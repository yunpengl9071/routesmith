"""Stateful conversation tracker for multi-turn agent sessions."""

from __future__ import annotations

import uuid
from typing import Any

from routesmith.config import RouteContext

_CORRECTION_WORDS = frozenset([
    "no", "wrong", "incorrect", "actually", "wait",
    "not right", "try again", "mistake",
])


class ConversationTracker:
    """Tracks turns in a conversation and emits RouteContext per call.

    Optional — callers who don't need stateful tracking can pass a
    RouteContext directly to completion() instead.

    Args:
        agent_role: Explicit agent role. If not provided, AgentInferencer
            attempts inference from the system prompt on each turn.
        agent_id: Optional agent instance identifier.
        conversation_id: Fixed conversation ID. Auto-generated if None.
        reward_fn: Optional per-instance reward function override.
    """

    def __init__(
        self,
        agent_role: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        reward_fn: Any = None,
    ) -> None:
        self._agent_role = agent_role
        self._agent_id = agent_id
        self._conversation_id = conversation_id or uuid.uuid4().hex[:16]
        self.reward_fn = reward_fn

        self._turn_count = 0
        self._cumulative_chars = 0
        self._correction_count = 0
        self._first_user_content: str | None = None

    def next_context(self, messages: list[dict]) -> RouteContext:
        """Compute RouteContext for the next turn.

        Call immediately before each completion() call, passing the current
        full message list. Pass the returned context to completion(context=...).
        """
        # Record first user message for topic drift
        for msg in messages:
            if msg.get("role") == "user" and self._first_user_content is None:
                self._first_user_content = str(msg.get("content", ""))
                break

        # Detect correction in the latest user message (skip turn 0)
        if self._turn_count > 0:
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user:
                content_lower = str(last_user.get("content", "")).lower()
                if any(word in content_lower for word in _CORRECTION_WORDS):
                    self._correction_count += 1

        self._cumulative_chars += sum(
            len(str(m.get("content", ""))) for m in messages
        )

        # Topic drift: word overlap between first and current user message
        topic_drift = 0.0
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if self._first_user_content and last_user_msg and self._turn_count > 0:
            first_words = set(self._first_user_content.lower().split())
            curr_words = set(str(last_user_msg.get("content", "")).lower().split())
            if first_words:
                overlap = len(first_words & curr_words) / len(first_words)
                topic_drift = 1.0 - min(1.0, overlap)

        request_id = uuid.uuid4().hex[:16]

        ctx = RouteContext(
            agent_id=self._agent_id,
            agent_role=self._agent_role,
            conversation_id=self._conversation_id,
            turn_index=self._turn_count,
            metadata={
                "correction_count": self._correction_count,
                "topic_drift": topic_drift,
                "cumulative_token_estimate": self._cumulative_chars / 4.0,
                "request_id": request_id,
            },
        )

        self._turn_count += 1
        return ctx

    def record_outcome(
        self,
        request_id: str,
        quality_score: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Optional: record outcome for a completed turn.

        Implicit signals are collected automatically by FeedbackCollector
        on every completion() call. Call this only when the application
        has an explicit quality signal (task completion, user rating, etc.).
        Does nothing by default — callers wire this to rs.record_outcome()
        if they need the explicit signal to reach the predictor.
        """
        pass

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def conversation_id(self) -> str:
        return self._conversation_id
