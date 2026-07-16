"""
CrewAI multi-agent crew with RouteSmith.

Two agents (researcher, writer) using a shared RouteSmith instance
via the native ChatRouteSmith integration.

Requires: routesmith-llm[crewai]
"""

from routesmith import RouteSmith
from routesmith.integrations.crewai import routesmith_crewai_chat_model


def main() -> None:
    try:
        from crewai import Agent, Crew, Task
    except ImportError:
        print("crewai not installed. Install with: pip install 'routesmith-llm[crewai]'")
        return

    rs = RouteSmith()
    rs.register_model(
        "openai/gpt-4o-mini",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        quality_score=0.85,
    )

    llm = routesmith_crewai_chat_model(routesmith=rs)

    researcher = Agent(
        role="Researcher",
        goal="Find key facts about AI history",
        backstory="You are a thorough researcher.",
        llm=llm,
    )
    writer = Agent(
        role="Writer",
        goal="Write a concise summary",
        backstory="You are a clear writer.",
        llm=llm,
    )

    task1 = Task(description="List 3 milestones in AI development", agent=researcher)
    task2 = Task(description="Summarize the milestones in 2 sentences", agent=writer)

    crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
    result = crew.kickoff()
    print(f"Crew result: {result}")

    print(f"\nStats: {rs.stats}")


if __name__ == "__main__":
    main()
