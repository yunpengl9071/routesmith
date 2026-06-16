# CrewAI Integration

RouteSmith integrates with CrewAI as a custom LLM provider.

```python
from routesmith.integrations.crewai import RouteSmithCrewAI
from crewai import Agent, Task, Crew

# Configure RouteSmith
llm_config = RouteSmithCrewAI()

# Use in CrewAI agents
agent = Agent(
    role="Researcher",
    goal="Research topics thoroughly",
    backstory="You are an expert researcher",
    llm_config=llm_config,
)

task = Task(
    description="What is machine learning?",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

## Custom Configuration

```python
from routesmith import RouteSmith, RouteSmithConfig

rs = RouteSmith(RouteSmithConfig().with_budget(max_cost_per_day=5.0))
llm_config = RouteSmithCrewAI(routesmith=rs)
```