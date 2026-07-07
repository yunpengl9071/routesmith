"""
Quickstart: Route Smith in Python.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/quickstart_python.py

Requires: routesmith (no extra deps)
"""

from routesmith import RouteSmith


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

    response = rs.completion(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=50,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Stats: {rs.stats}")
    meta = rs.last_routing_metadata
    if meta:
        print(f"Model: {meta.model_selected}, Cost: ${meta.estimated_cost_usd:.6f}")
    if hasattr(response, "_routesmith_request_id") and response._routesmith_request_id:
        rs.record_outcome(response._routesmith_request_id, score=1.0)


if __name__ == "__main__":
    main()
