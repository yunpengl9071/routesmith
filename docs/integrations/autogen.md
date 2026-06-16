# AutoGen Integration

RouteSmith integrates with Microsoft AutoGen as a custom LLM configuration.

```python
from routesmith.integrations.autogen import get_routesmith_llm_config, create_routesmith_agents

# Create agents with RouteSmith-powered LLM
assistant, user_proxy = create_routesmith_agents(
    system_message="You are a helpful AI assistant.",
    routesmith_kwargs={"cache_seed": None},
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="What is the capital of France?",
)
```

## Custom Configuration

```python
from routesmith import RouteSmith, RouteSmithConfig

rs = RouteSmith(RouteSmithConfig().with_budget(max_cost_per_day=10.0))
llm_config = get_routesmith_llm_config(routesmith=rs)
```