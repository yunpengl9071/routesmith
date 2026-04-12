"""Tests for ConversationTracker — stateful multi-turn context tracking."""

from __future__ import annotations

from routesmith.feedback.conversation import ConversationTracker


class TestConversationTracker:
    def test_first_turn_has_index_zero(self):
        tracker = ConversationTracker(agent_role="research")
        ctx = tracker.next_context([{"role": "user", "content": "Hello"}])
        assert ctx.turn_index == 0
        assert ctx.agent_role == "research"
        assert ctx.conversation_id is not None

    def test_turn_index_increments(self):
        tracker = ConversationTracker()
        ctx1 = tracker.next_context([{"role": "user", "content": "Hi"}])
        ctx2 = tracker.next_context([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        assert ctx2.turn_index == ctx1.turn_index + 1

    def test_consistent_conversation_id(self):
        tracker = ConversationTracker(conversation_id="conv_abc")
        ctx1 = tracker.next_context([{"role": "user", "content": "Hi"}])
        ctx2 = tracker.next_context([{"role": "user", "content": "Hi"}])
        assert ctx1.conversation_id == "conv_abc"
        assert ctx2.conversation_id == "conv_abc"

    def test_correction_count_increments(self):
        tracker = ConversationTracker()
        tracker.next_context([{"role": "user", "content": "What is the capital of France?"}])
        tracker.next_context([{"role": "user", "content": "No, that's wrong. Try again."}])
        assert tracker._correction_count == 1

    def test_metadata_includes_correction_count(self):
        tracker = ConversationTracker()
        tracker.next_context([{"role": "user", "content": "Hi"}])
        ctx = tracker.next_context([{"role": "user", "content": "Actually that is incorrect."}])
        assert "correction_count" in ctx.metadata

    def test_auto_generated_conversation_id(self):
        tracker = ConversationTracker()
        ctx = tracker.next_context([{"role": "user", "content": "Hi"}])
        assert isinstance(ctx.conversation_id, str)
        assert len(ctx.conversation_id) > 0

    def test_record_outcome_does_not_raise(self):
        tracker = ConversationTracker()
        ctx = tracker.next_context([{"role": "user", "content": "Hi"}])
        tracker.record_outcome(ctx.metadata["request_id"], quality_score=0.9)
        tracker.record_outcome("nonexistent_id")
