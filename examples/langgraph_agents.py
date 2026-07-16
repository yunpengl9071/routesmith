"""
LangGraph multi-agent with per-role routing.

Two agents (planner, executor) each with their own ChatRouteSmith
over a shared RouteSmith instance.

Requires: routesmith-llm[langchain]
          pip install langgraph
"""

from routesmith import RouteSmith
from routesmith.integrations.langchain import ChatRouteSmith


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

    planner_llm = ChatRouteSmith(routesmith=rs, agent_role="planner")
    executor_llm = ChatRouteSmith(routesmith=rs, agent_role="executor")

    prompt = "Design a REST API for a todo app, then implement the list endpoint"
    response = planner_llm.invoke(
        [{"role": "user", "content": f"Plan: {prompt}"}]
    )
    print(f"[Planner] {response.content[:100]}...")

    response = executor_llm.invoke(
        [{"role": "user", "content": f"Implement: {prompt}"}]
    )
    print(f"[Executor] {response.content[:100]}...")

    print(f"\nStats: {rs.stats}")


if __name__ == "__main__":
    main()
