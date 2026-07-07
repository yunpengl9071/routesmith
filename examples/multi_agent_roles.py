"""
Multi-agent role routing with RouteSmith.

Demonstrates per-role routing policies: planner, coder, summarizer
each get routed to the optimal model for their task type.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/multi_agent_roles.py

Requires: routesmith (no extra deps)
"""

from routesmith import RouteContext, RouteSmith


def main() -> None:
    rs = RouteSmith()
    rs.register_model(
        "openai/gpt-4o-mini",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        quality_score=0.85,
    )
    rs.register_model(
        "openai/gpt-4o",
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.00,
        quality_score=0.95,
    )
    rs.register_model(
        "anthropic/claude-sonnet-4",
        cost_per_1k_input=3.00,
        cost_per_1k_output=15.00,
        quality_score=0.92,
    )

    roles = ["planner", "coder", "summarizer"]
    prompts = [
        "Plan the architecture for a microservice that processes user uploads",
        "Write a Python function to validate email addresses using regex",
        "Summarize the key differences between REST and GraphQL APIs",
    ]

    for role, prompt in zip(roles, prompts):
        ctx = RouteContext(agent_role=role)
        response = rs.completion(
            messages=[{"role": "user", "content": prompt}],
            context=ctx,
            max_tokens=200,
        )
        meta = rs.last_routing_metadata
        print(f"[{role}] Model: {meta.model_selected if meta else '?'}, "
              f"Cost: ${meta.estimated_cost_usd:.6f}" if meta else "")
        print(f"  Response: {response.choices[0].message.content[:80]}...")
        print()


if __name__ == "__main__":
    main()
