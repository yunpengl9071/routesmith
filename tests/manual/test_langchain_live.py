#!/usr/bin/env python3
"""
Live API test for LangChain integration.

Supports OpenAI or Groq. Set one of:
  export OPENAI_API_KEY=sk-...
  export GROQ_API_KEY=gsk_...

Run with: python tests/manual/test_langchain_live.py

NOTE: Skipped by pytest unless API keys are set.
"""

import os
import sys
import warnings

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"),
    reason="Requires OPENAI_API_KEY or GROQ_API_KEY to run.",
)


def _detect_provider() -> str:
    """Detect which provider to use based on env vars."""
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return ""


def _make_llm(**kwargs):
    """Create a ChatRouteSmith with the detected provider's models."""
    from routesmith.integrations.langchain import ChatRouteSmith

    provider = _detect_provider()
    if provider == "groq":
        return ChatRouteSmith.with_groq_models(**kwargs)
    return ChatRouteSmith.with_openai_models(**kwargs)


def _make_llm_for_tools(**kwargs):
    """Create a ChatRouteSmith suited for tool calling.

    Sets min_quality=0.85 so the router picks a model that reliably
    follows the OpenAI tool-calling format (e.g. llama-3.3-70b on Groq).
    """
    kwargs.setdefault("min_quality", 0.85)
    return _make_llm(**kwargs)


def _retry_on_groq_tool_error(fn, max_retries=2):
    """Retry a test function if Groq returns a tool_use_failed error.

    Groq's Llama models sometimes generate malformed tool calls
    (XML-style instead of structured JSON). Retrying usually works.
    """
    import litellm

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except litellm.BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < max_retries:
                print(f"   (Groq tool_use_failed, retrying... attempt {attempt + 2})")
                continue
            raise


def check_api_key():
    provider = _detect_provider()
    if not provider:
        print("ERROR: No API key found.")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    print(f"Provider: {provider}")
    return provider


# ── 1. Basic invoke ──────────────────────────────────────────────────────

def test_basic_invoke():
    """ChatRouteSmith.invoke() returns an AIMessage with routing metadata."""
    from langchain_core.messages import AIMessage

    print("1. Basic invoke...")

    llm = _make_llm()
    result = llm.invoke("What is 2+2? Reply with just the number.")

    assert isinstance(result, AIMessage)
    assert result.content.strip()
    print(f"   Content: {result.content.strip()}")

    meta = result.response_metadata
    assert "model_name" in meta, "Missing model_name in response_metadata"
    assert "routesmith" in meta, "Missing routesmith in response_metadata"
    assert "routesmith_request_id" in meta

    print(f"   Model: {meta['model_name']}")
    print(f"   Tokens: {meta.get('token_usage', {})}")
    print(f"   Routing: {meta['routesmith'].get('routing_reason', '')}")
    print("   PASSED")


# ── 2. Tool calling via bind_tools ────────────────────────────────────────

def test_tool_calling():
    """bind_tools() + invoke returns AIMessage with tool_calls."""
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    print("2. Tool calling (bind_tools)...")

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    llm = _make_llm_for_tools()
    bound = llm.bind_tools([multiply])
    result = bound.invoke("What is 137 times 42? Use the multiply tool.")

    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) > 0, "Expected tool_calls in response"

    tc = result.tool_calls[0]
    print(f"   Tool called: {tc['name']}")
    print(f"   Args: {tc['args']}")
    print(f"   ID: {tc['id']}")
    assert tc["name"] == "multiply"
    assert tc["args"]["a"] in (137, 42)
    assert tc["args"]["b"] in (137, 42)
    print("   PASSED")


# ── 3. ReAct agent full loop ─────────────────────────────────────────────

def test_react_agent_loop():
    """create_react_agent runs a full tool-call loop and returns correct answer."""
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool

    print("3. ReAct agent (full tool-call loop)...")

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    llm = _make_llm_for_tools()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = create_react_agent(llm, tools=[add])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is 19 + 23? Use the add tool."}]}
    )
    messages = result["messages"]

    for m in messages:
        role = type(m).__name__
        if isinstance(m, AIMessage) and m.tool_calls:
            print(f"   {role}: [tool_calls: {[tc['name'] for tc in m.tool_calls]}]")
        else:
            content = str(m.content)[:80]
            print(f"   {role}: {content}")

    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1, "Tool was not called"
    assert "42" in tool_msgs[0].content, f"Tool returned wrong result: {tool_msgs[0].content}"

    final = messages[-1]
    assert isinstance(final, AIMessage)
    assert "42" in final.content, f"Final answer missing 42: {final.content}"
    print("   PASSED")


# ── 4. ReAct agent with multiple tools ────────────────────────────────────

def test_react_agent_multi_tool():
    """Agent picks the right tool from multiple options."""
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool

    print("4. ReAct agent (multi-tool selection)...")

    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city. Accepts city names like 'SF', 'San Francisco', 'NYC'."""
        city_lower = city.lower().replace(" ", "")
        weather_data = {
            "nyc": "65°F, cloudy", "newyork": "65°F, cloudy", "newyorkcity": "65°F, cloudy",
            "sf": "58°F, foggy", "sanfrancisco": "58°F, foggy",
            "london": "52°F, rainy",
        }
        return weather_data.get(city_lower, f"No data for {city}")

    @tool
    def get_population(city: str) -> str:
        """Get population of a city. Accepts city names like 'SF', 'San Francisco', 'NYC'."""
        city_lower = city.lower().replace(" ", "")
        pop_data = {
            "nyc": "8.3 million", "newyork": "8.3 million", "newyorkcity": "8.3 million",
            "sf": "870,000", "sanfrancisco": "870,000",
            "london": "9 million",
        }
        return pop_data.get(city_lower, f"No data for {city}")

    llm = _make_llm_for_tools()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = create_react_agent(llm, tools=[get_weather, get_population])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
    )
    messages = result["messages"]

    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    print(f"   Tool result: {tool_msgs[0].content}")
    assert "58°F" in tool_msgs[0].content or "foggy" in tool_msgs[0].content
    print("   PASSED")


# ── 5. Streaming ──────────────────────────────────────────────────────────

def test_streaming():
    """stream() yields chunks with content."""
    from langchain_core.messages import AIMessageChunk

    print("5. Streaming...")

    llm = _make_llm()
    chunks = []
    for chunk in llm.stream("Count from 1 to 5. Just the numbers."):
        chunks.append(chunk)

    assert len(chunks) > 0, "No chunks received"
    full_text = "".join(c.content for c in chunks)
    print(f"   Chunks received: {len(chunks)}")
    print(f"   Full text: {full_text.strip()[:80]}")
    assert isinstance(chunks[0], AIMessageChunk)
    print("   PASSED")


# ── 6. Async invoke ───────────────────────────────────────────────────────

def test_async_invoke():
    """ainvoke() returns correct result."""
    import asyncio
    from langchain_core.messages import AIMessage

    print("6. Async invoke...")

    async def run():
        llm = _make_llm()
        result = await llm.ainvoke("Say 'hello' and nothing else.")
        return result

    result = asyncio.run(run())
    assert isinstance(result, AIMessage)
    assert result.content.strip()
    print(f"   Content: {result.content.strip()}")
    print("   PASSED")


# ── 7. record_outcome feedback loop ──────────────────────────────────────

def test_record_outcome():
    """record_outcome() stores feedback for a completed request."""
    import routesmith as rs_mod
    from routesmith import RouteSmithConfig

    print("7. record_outcome feedback loop...")

    config = RouteSmithConfig(feedback_sample_rate=1.0)
    llm = _make_llm(routesmith=rs_mod.RouteSmith(config=config))

    result = llm.invoke("What is the capital of Japan?")
    request_id = result.response_metadata.get("routesmith_request_id")
    assert request_id, "No request_id in response_metadata"

    found = llm.record_outcome(request_id=request_id, success=True, score=0.95)
    assert found is True, "record_outcome could not find the request"

    print(f"   Request ID: {request_id}")
    print(f"   Outcome recorded: success=True, score=0.95")
    print("   PASSED")


# ── 8. Cost tracking across agent run ─────────────────────────────────────

def test_cost_tracking():
    """RouteSmith stats accumulate across an agent's tool-calling loop."""
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool

    print("8. Cost tracking across agent run...")

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    llm = _make_llm_for_tools()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = create_react_agent(llm, tools=[add])

    agent.invoke(
        {"messages": [{"role": "user", "content": "What is 10 + 20? Use the add tool."}]}
    )

    stats = llm.routesmith.stats
    print(f"   Requests: {stats['request_count']}")
    print(f"   Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"   Savings: ${stats['cost_savings_usd']:.6f} ({stats['savings_percent']}%)")

    assert stats["request_count"] >= 2, f"Expected >=2 requests, got {stats['request_count']}"
    assert stats["total_cost_usd"] > 0, "Cost should be > 0"
    print("   PASSED")


# ── 9. Supervisor multi-agent ──────────────────────────────────────────────

def test_supervisor_agent():
    """Supervisor delegates to specialist sub-agents."""
    from langgraph_supervisor import create_supervisor
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool

    print("9. Supervisor multi-agent...")

    llm = _make_llm_for_tools()

    @tool
    def add(a: str, b: str) -> str:
        """Add two numbers together. Arguments are numeric strings."""
        return str(int(a) + int(b))

    @tool
    def multiply(a: str, b: str) -> str:
        """Multiply two numbers together. Arguments are numeric strings."""
        return str(int(a) * int(b))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        math_agent = create_react_agent(
            llm, [add, multiply],
            name="math_agent",
            prompt="You are a math specialist. Always use tools for calculations.",
        )
        general_agent = create_react_agent(
            llm, [],
            name="general_agent",
            prompt="You answer general knowledge questions briefly.",
        )

        supervisor = create_supervisor(
            [math_agent, general_agent],
            model=llm,
            prompt=(
                "You are a supervisor. Route math questions to math_agent "
                "and everything else to general_agent."
            ),
        ).compile()

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "What is 15 + 27?"}]}
    )
    messages = result["messages"]

    # Print conversation trace
    for m in messages:
        role = type(m).__name__
        name = getattr(m, "name", None)
        prefix = f"{role}({name})" if name else role
        if isinstance(m, AIMessage) and m.tool_calls:
            tools = [tc["name"] for tc in m.tool_calls]
            print(f"   {prefix}: [tool_calls: {tools}]")
        else:
            print(f"   {prefix}: {str(m.content)[:80]}")

    # The math agent should have been invoked and used the add tool
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1, "No tools were called"

    # 42 should appear in at least one AIMessage (the math agent's answer)
    # The supervisor's final message may be a pleasantry, so check all messages.
    ai_contents = [m.content for m in messages if isinstance(m, AIMessage) and m.content]
    assert any("42" in c for c in ai_contents), \
        f"Expected 42 in some AI message, got: {ai_contents}"

    # Verify cost tracking across all agents
    stats = llm.routesmith.stats
    # Supervisor call + math_agent calls (at least tool call + final answer)
    assert stats["request_count"] >= 3, f"Expected >=3 requests, got {stats['request_count']}"
    print(f"   Total LLM calls: {stats['request_count']}")
    print(f"   Total cost: ${stats['total_cost_usd']:.6f}")
    print("   PASSED")


# ── 10. Swarm multi-agent ─────────────────────────────────────────────────

def test_swarm_agent():
    """Swarm agents hand off to each other."""
    from langgraph_swarm import create_swarm, create_handoff_tool
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    print("10. Swarm multi-agent (handoff)...")

    llm = _make_llm_for_tools()

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order by ID."""
        orders = {"ORD-123": "Shipped, arriving tomorrow", "ORD-456": "Processing"}
        return orders.get(order_id, f"Order {order_id} not found")

    @tool
    def check_inventory(product: str) -> str:
        """Check inventory for a product."""
        inventory = {"widget": "52 in stock", "gadget": "out of stock"}
        return inventory.get(product.lower(), f"{product}: unknown")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        support_agent = create_react_agent(
            llm,
            [lookup_order, create_handoff_tool(agent_name="sales_agent")],
            name="support_agent",
            prompt=(
                "You are a support agent. Help with order lookups. "
                "If the user asks about buying or inventory, hand off to sales_agent."
            ),
        )
        sales_agent = create_react_agent(
            llm,
            [check_inventory, create_handoff_tool(agent_name="support_agent")],
            name="sales_agent",
            prompt=(
                "You are a sales agent. Help with inventory and purchases. "
                "If the user asks about an existing order, hand off to support_agent."
            ),
        )

        swarm = create_swarm(
            [support_agent, sales_agent],
            default_active_agent="support_agent",
        ).compile()

    # Ask about an order — support_agent should handle directly
    result = swarm.invoke(
        {"messages": [{"role": "user", "content": "What's the status of order ORD-123?"}]}
    )
    messages = result["messages"]

    for m in messages:
        role = type(m).__name__
        name = getattr(m, "name", None)
        prefix = f"{role}({name})" if name else role
        print(f"   {prefix}: {str(m.content)[:80]}")

    # Should have used lookup_order tool
    from langchain_core.messages import ToolMessage
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1, "No tools were called"

    # Final answer should mention shipping/arriving
    final = messages[-1]
    assert isinstance(final, AIMessage)
    content_lower = final.content.lower()
    assert "ship" in content_lower or "tomorrow" in content_lower or "arriving" in content_lower, \
        f"Expected shipping info in answer: {final.content}"

    stats = llm.routesmith.stats
    print(f"   Total LLM calls: {stats['request_count']}")
    print(f"   Total cost: ${stats['total_cost_usd']:.6f}")
    print("   PASSED")


# ── Runner ────────────────────────────────────────────────────────────────

# Tests that involve tool calling may hit flaky Groq tool_use_failed errors.
# Mark them so the runner can retry.
_TOOL_CALLING_TESTS = {
    test_tool_calling,
    test_react_agent_loop,
    test_react_agent_multi_tool,
    test_cost_tracking,
    test_supervisor_agent,
    test_swarm_agent,
}

ALL_TESTS = [
    test_basic_invoke,
    test_tool_calling,
    test_react_agent_loop,
    test_react_agent_multi_tool,
    test_streaming,
    test_async_invoke,
    test_record_outcome,
    test_cost_tracking,
    test_supervisor_agent,
    test_swarm_agent,
]


if __name__ == "__main__":
    print("=" * 60)
    print("ChatRouteSmith Live API Tests")
    print("=" * 60)
    print()

    provider = check_api_key()
    print()

    passed = 0
    failed = 0

    for test_fn in ALL_TESTS:
        try:
            if test_fn in _TOOL_CALLING_TESTS:
                _retry_on_groq_tool_error(test_fn)
            else:
                test_fn()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed" + (f", {failed} failed" if failed else ""))
    print("=" * 60)
    sys.exit(1 if failed else 0)
