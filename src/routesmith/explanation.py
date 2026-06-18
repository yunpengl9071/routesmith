"""Format routing metadata into human-readable explanations."""


def format_explanation(
    metadata,  # RoutingMetadata
    qualifying: int,
    rejected: int,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> str:
    """Produce a human-readable routing explanation string."""
    lines = [
        f"Model used: {metadata.model_selected}",
        f"Reason: {metadata.routing_reason}",
        f"Models considered: {len(metadata.models_considered)}"
        f"  |  Qualified: {qualifying}  |  Rejected: {rejected}",
        f"Routing latency: {metadata.routing_latency_ms:.1f}ms",
        f"Estimated cost: ${metadata.estimated_cost_usd:.4f}",
        f"Savings vs best: ${metadata.cost_savings_usd:.4f}"
        f" ({_savings_pct(metadata)}%)",
        f"Cache: {'hit' if metadata.cache_hit else 'miss'}",
    ]

    if conversation_id:
        turn_str = f" (turn {turn_index})" if turn_index is not None else ""
        lines.append(f"Conversation: {conversation_id}{turn_str}")

    return "\n".join(lines)


def _savings_pct(metadata) -> str:
    if metadata.counterfactual_cost_usd > 0:
        return f"{metadata.cost_savings_usd / metadata.counterfactual_cost_usd * 100:.1f}"
    return "0.0"