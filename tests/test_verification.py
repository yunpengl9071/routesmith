"""Tests for shadow execution trust-but-verify."""

from unittest.mock import MagicMock, patch

import pytest

from routesmith.verification import (
    VerificationTracker,
    compare_responses,
    shadow_execute,
)


class TestCompareResponses:
    def test_identical_responses(self):
        """Identical responses are marked equivalent."""
        resp1 = MagicMock()
        resp1.choices[0].message.content = "Hello world"
        resp2 = MagicMock()
        resp2.choices[0].message.content = "Hello world"

        result = compare_responses(resp1, resp2)

        assert result["equivalent"] is True
        assert "identical" in result["summary"].lower()

    def test_similar_responses(self):
        """Substantially similar responses are marked equivalent."""
        resp1 = MagicMock()
        resp1.choices[0].message.content = "The quick brown fox jumps over the lazy dog"
        resp2 = MagicMock()
        resp2.choices[0].message.content = "The quick brown fox jumped over a lazy dog"

        result = compare_responses(resp1, resp2)

        # High similarity should be equivalent
        assert result["equivalent"] is True

    def test_different_responses(self):
        """Substantially different responses are marked non-equivalent."""
        resp1 = MagicMock()
        resp1.choices[0].message.content = "The quick brown fox"
        resp2 = MagicMock()
        resp2.choices[0].message.content = "Completely different content here"

        result = compare_responses(resp1, resp2)

        assert result["equivalent"] is False

    def test_different_lengths_reduce_similarity(self):
        """Shorter vs longer responses are penalized."""
        resp1 = MagicMock()
        resp1.choices[0].message.content = "Yes."
        resp2 = MagicMock()
        resp2.choices[0].message.content = "Yes, absolutely, without a doubt, certainly."

        result = compare_responses(resp1, resp2)
        assert result["similarity"] < 1.0


class TestVerificationTracker:
    def test_record_and_stats(self):
        """Tracker accumulates results and returns stats."""
        tracker = VerificationTracker()

        tracker.record(
            agent_role="coder",
            cheap_model="gpt-4o-mini",
            expensive_model="gpt-4o",
            equivalent=True,
            summary="identical",
            savings="$0.012 (93%)",
        )
        tracker.record(
            agent_role="coder",
            cheap_model="gpt-4o-mini",
            expensive_model="gpt-4o",
            equivalent=False,
            summary="substantially different",
            savings="$0.001 (10%)",
        )

        stats = tracker.stats("coder")
        assert stats["verified"] == 2
        assert stats["equivalence_rate"] == pytest.approx(0.5)

    def test_empty_tracker(self):
        """Empty tracker returns zero stats."""
        tracker = VerificationTracker()
        stats = tracker.stats()
        assert stats["verified"] == 0

    def test_multiple_agents(self):
        """Stats are tracked per agent."""
        tracker = VerificationTracker()
        tracker.record("coder", "a", "b", True, "ok", "$1")
        tracker.record("writer", "c", "d", False, "bad", "$2")

        assert tracker.stats("coder")["verified"] == 1
        assert tracker.stats("writer")["verified"] == 1
        global_stats = tracker.stats()
        assert global_stats["verified"] == 2