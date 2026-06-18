"""Tests for routing explanation formatting."""

from routesmith.explanation import format_explanation
from routesmith.client import RoutingMetadata


class TestFormatExplanation:
    def test_formats_basic_explanation(self):
        """format_explanation produces readable text from metadata."""
        metadata = RoutingMetadata(
            request_id="abc123",
            model_selected="gpt-4o-mini",
            routing_strategy="direct",
            routing_reason="cheapest above quality threshold",
            routing_latency_ms=2.3,
            estimated_cost_usd=0.0012,
            counterfactual_cost_usd=0.015,
            cost_savings_usd=0.0138,
            models_considered=["gpt-4o", "gpt-4o-mini", "claude-haiku"],
            cache_hit=False,
        )

        explanation = format_explanation(metadata, qualifying=2, rejected=1)

        assert "gpt-4o-mini" in explanation
        assert "cheapest above quality threshold" in explanation
        assert "Cache: miss" in explanation

    def test_includes_cache_hit(self):
        """format_explanation shows cache hit status."""
        metadata = RoutingMetadata(
            request_id="def456",
            model_selected="gpt-4o",
            routing_strategy="direct",
            routing_reason="explicit model specified",
            routing_latency_ms=0.5,
            estimated_cost_usd=0.0,
            counterfactual_cost_usd=0.0,
            cost_savings_usd=0.0,
            models_considered=["gpt-4o"],
            cache_hit=True,
        )

        explanation = format_explanation(metadata, qualifying=1, rejected=0)
        assert "Cache: hit" in explanation

    def test_includes_conversation_context(self):
        """format_explanation shows conversation context when available."""
        metadata = RoutingMetadata(
            request_id="ghi789",
            model_selected="deepseek-chat",
            routing_strategy="direct",
            routing_reason="best quality-cost tradeoff",
            routing_latency_ms=1.8,
            estimated_cost_usd=0.0005,
            counterfactual_cost_usd=0.015,
            cost_savings_usd=0.0145,
            models_considered=["gpt-4o", "deepseek-chat"],
            cache_hit=False,
        )

        explanation = format_explanation(
            metadata,
            qualifying=1,
            rejected=1,
            conversation_id="chat-001",
            turn_index=3,
        )

        assert "chat-001" in explanation
        assert "turn 3" in explanation