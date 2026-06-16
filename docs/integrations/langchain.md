# LangChain Integration

RouteSmith provides `ChatRouteSmith`, a drop-in replacement for LangChain chat models.

## Installation

```bash
pip install routesmith[langchain]
```

## Usage

```python
from routesmith.integrations.langchain import ChatRouteSmith

# Create with provider presets
llm = ChatRouteSmith.with_openai_models()
llm = ChatRouteSmith.with_groq_models()
llm = ChatRouteSmith.with_anthropic_models()

# Use like any LangChain chat model!
result = llm.invoke("What is 2+2?")
print(result.content)
```

## With Tools

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm = ChatRouteSmith.with_groq_models()
bound = llm.bind_tools([multiply])
agent = create_react_agent(bound, tools=[multiply])

result = agent.invoke({"messages": [{"role": "user", "content": "What is 137 * 42?"}]})
print(result["messages"][-1].content)
```

## Cost Tracking

```python
stats = llm.routesmith.stats
print(f"Cost: ${stats['total_cost_usd']:.4f}")
```

## Streaming & Async

```python
# Streaming
for chunk in llm.stream("Count to 5"):
    print(chunk.content, end="")

# Async
result = await llm.ainvoke("Hello!")
```