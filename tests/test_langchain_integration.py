"""Tests for LangChain integration (ChatRouteSmith)."""

import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.integrations.langchain import (
    ChatRouteSmith,
    _langchain_messages_to_dicts,
    _litellm_response_to_ai_message,
    _litellm_chunk_to_generation_chunk,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


class TestLangchainMessagesToDicts:
    """Tests for _langchain_messages_to_dicts."""

    def test_human_message(self):
        msgs = [HumanMessage(content="Hello")]
        result = _langchain_messages_to_dicts(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_message(self):
        msgs = [SystemMessage(content="You are a bot.")]
        result = _langchain_messages_to_dicts(msgs)
        assert result == [{"role": "system", "content": "You are a bot."}]

    def test_ai_message_plain(self):
        msgs = [AIMessage(content="Hi there")]
        result = _langchain_messages_to_dicts(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_ai_message_with_tool_calls(self):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "SF"}, "id": "call_1"},
                ],
            )
        ]
        result = _langchain_messages_to_dicts(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "SF"}

    def test_tool_message(self):
        msgs = [ToolMessage(content="72°F", tool_call_id="call_1")]
        result = _langchain_messages_to_dicts(msgs)
        assert result == [
            {"role": "tool", "content": "72°F", "tool_call_id": "call_1"}
        ]

    def test_mixed_conversation(self):
        msgs = [
            SystemMessage(content="You help with weather."),
            HumanMessage(content="Weather in SF?"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_weather", "args": {"city": "SF"}, "id": "call_1"},
                ],
            ),
            ToolMessage(content="72°F", tool_call_id="call_1"),
        ]
        result = _langchain_messages_to_dicts(msgs)
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "tool"


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestLitellmResponseToAIMessage:
    """Tests for _litellm_response_to_ai_message."""

    def _make_response(
        self,
        content="Hello!",
        tool_calls=None,
        model="gpt-4o-mini",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
    ):
        response = MagicMock()
        response.model = model
        response.choices[0].message.content = content
        response.choices[0].message.tool_calls = tool_calls
        response.choices[0].finish_reason = finish_reason
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        response.usage.total_tokens = prompt_tokens + completion_tokens
        return response

    def test_basic_response(self):
        response = self._make_response(content="Hi!")
        msg = _litellm_response_to_ai_message(response)
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hi!"
        assert msg.tool_calls == []

    def test_response_with_tool_calls(self):
        tc = MagicMock()
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "SF"}'
        tc.id = "call_abc"

        response = self._make_response(content="", tool_calls=[tc])
        msg = _litellm_response_to_ai_message(response)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "SF"}
        assert msg.tool_calls[0]["id"] == "call_abc"

    def test_response_metadata(self):
        response = self._make_response()
        routing_meta = {
            "request_id": "req_123",
            "model_selected": "gpt-4o-mini",
            "routing_strategy": "direct",
        }
        msg = _litellm_response_to_ai_message(response, routing_meta)
        assert msg.response_metadata["model_name"] == "gpt-4o-mini"
        assert msg.response_metadata["finish_reason"] == "stop"
        assert msg.response_metadata["token_usage"]["prompt_tokens"] == 10
        assert msg.response_metadata["routesmith"] == routing_meta
        assert msg.response_metadata["routesmith_request_id"] == "req_123"

    def test_response_no_routing_meta(self):
        response = self._make_response()
        msg = _litellm_response_to_ai_message(response)
        assert "routesmith" not in msg.response_metadata


# ---------------------------------------------------------------------------
# Streaming chunk conversion
# ---------------------------------------------------------------------------


class TestLitellmChunkToGenerationChunk:
    """Tests for _litellm_chunk_to_generation_chunk."""

    def test_content_chunk(self):
        chunk = MagicMock()
        chunk.choices[0].delta.content = "Hello"
        chunk.choices[0].delta.tool_calls = None
        result = _litellm_chunk_to_generation_chunk(chunk)
        assert result is not None
        assert result.text == "Hello"

    def test_empty_choices_returns_none(self):
        chunk = MagicMock()
        chunk.choices = []
        result = _litellm_chunk_to_generation_chunk(chunk)
        assert result is None

    def test_tool_call_chunk(self):
        tc = MagicMock()
        tc.function.name = "search"
        tc.function.arguments = '{"q":'
        tc.id = "call_1"
        tc.index = 0

        chunk = MagicMock()
        chunk.choices[0].delta.content = ""
        chunk.choices[0].delta.tool_calls = [tc]

        result = _litellm_chunk_to_generation_chunk(chunk)
        assert result is not None
        assert len(result.message.tool_call_chunks) == 1
        assert result.message.tool_call_chunks[0]["name"] == "search"


# ---------------------------------------------------------------------------
# ChatRouteSmith class
# ---------------------------------------------------------------------------


class TestChatRouteSmith:
    """Tests for ChatRouteSmith."""

    @pytest.fixture
    def llm(self):
        """Create a ChatRouteSmith with test models registered."""
        rs = RouteSmith()
        rs.register_model(
            "gpt-4o",
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            quality_score=0.95,
        )
        rs.register_model(
            "gpt-4o-mini",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
        )
        return ChatRouteSmith(routesmith=rs)

    def test_llm_type(self, llm):
        assert llm._llm_type == "routesmith"

    def test_default_routesmith_created(self):
        llm = ChatRouteSmith()
        assert llm.routesmith is not None
        assert isinstance(llm.routesmith, RouteSmith)

    @patch("routesmith.client.litellm")
    def test_generate_end_to_end(self, mock_litellm, llm):
        """Test full _generate flow."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_litellm.completion.return_value = mock_response

        result = llm._generate([HumanMessage(content="Hi")])

        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello!"

        # Verify litellm was called with correct message format
        call_kwargs = mock_litellm.completion.call_args
        messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages_arg == [{"role": "user", "content": "Hi"}]

    @patch("routesmith.client.litellm")
    def test_generate_with_routing_metadata(self, mock_litellm, llm):
        """Test that routing metadata appears in response_metadata."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices[0].message.content = "Hi"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        # Simulate routesmith_metadata being attached
        mock_response.routesmith_metadata = {
            "request_id": "abc123",
            "model_selected": "gpt-4o-mini",
            "routing_strategy": "direct",
        }
        mock_litellm.completion.return_value = mock_response

        result = llm._generate([HumanMessage(content="Hi")])
        msg = result.generations[0].message
        assert "routesmith" in msg.response_metadata
        assert msg.response_metadata["routesmith"]["model_selected"] == "gpt-4o-mini"
        # request_id is generated by routesmith.completion(), not from mock metadata
        assert "routesmith_request_id" in msg.response_metadata
        assert isinstance(msg.response_metadata["routesmith_request_id"], str)

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_agenerate(self, mock_litellm, llm):
        """Test async _agenerate flow."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices[0].message.content = "Async hello"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        result = await llm._agenerate([HumanMessage(content="Hi")])
        msg = result.generations[0].message
        assert msg.content == "Async hello"

    @patch("routesmith.client.litellm")
    def test_tool_calling_round_trip(self, mock_litellm, llm):
        """Test that bind_tools() flows through to litellm."""
        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices[0].message.content = ""

        tc = MagicMock()
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "SF"}'
        tc.id = "call_xyz"
        mock_response.choices[0].message.tool_calls = [tc]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        mock_litellm.completion.return_value = mock_response

        # bind_tools stores tools as kwargs that flow through
        tool_schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
        bound_llm = llm.bind(tools=[tool_schema])
        result = bound_llm.invoke([HumanMessage(content="Weather in SF?")])

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"city": "SF"}

        # Verify tools were passed through to litellm
        call_kwargs = mock_litellm.completion.call_args
        assert "tools" in (call_kwargs.kwargs or {})

    @patch("routesmith.client.litellm")
    def test_record_outcome_delegates(self, mock_litellm):
        """Test record_outcome delegates to routesmith."""
        # Use sample_rate=1.0 so feedback is always recorded
        config = RouteSmithConfig(feedback_sample_rate=1.0)
        rs = RouteSmith(config=config)
        rs.register_model("model", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.85)
        test_llm = ChatRouteSmith(routesmith=rs)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_litellm.completion.return_value = mock_response

        # Make a completion to create a feedback record
        test_llm._generate([HumanMessage(content="Hi")])

        request_id = test_llm.routesmith._last_routing_metadata.request_id
        result = test_llm.record_outcome(request_id=request_id, score=0.9)
        assert result is True

    def test_record_outcome_unknown_id(self, llm):
        """Test record_outcome returns False for unknown request."""
        result = llm.record_outcome(request_id="nonexistent", success=True)
        assert result is False

    @patch("routesmith.client.litellm")
    def test_stream(self, mock_litellm, llm):
        """Test _stream yields generation chunks."""
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.tool_calls = None

        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = " world"
        chunk2.choices[0].delta.tool_calls = None

        mock_litellm.completion.return_value = iter([chunk1, chunk2])

        chunks = list(llm._stream([HumanMessage(content="Hi")]))
        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    """Tests for convenience factory methods."""

    def test_with_openai_models(self):
        llm = ChatRouteSmith.with_openai_models()
        assert isinstance(llm, ChatRouteSmith)
        models = [m.model_id for m in llm.routesmith.registry.list_models()]
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    def test_with_anthropic_models(self):
        llm = ChatRouteSmith.with_anthropic_models()
        assert isinstance(llm, ChatRouteSmith)
        models = [m.model_id for m in llm.routesmith.registry.list_models()]
        assert any("claude" in m for m in models)

    def test_factory_with_custom_kwargs(self):
        llm = ChatRouteSmith.with_openai_models(min_quality=0.9, max_cost=0.01)
        assert llm.min_quality == 0.9
        assert llm.max_cost == 0.01


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error cases."""

    @patch("routesmith.client.litellm")
    def test_no_models_registered(self, mock_litellm):
        """Test clear error when no models are registered."""
        llm = ChatRouteSmith()
        with pytest.raises(ValueError):
            llm._generate([HumanMessage(content="Hi")])


class TestImportGuard:
    """Test that import guard provides helpful error."""

    def test_langchain_is_importable(self):
        """Verify langchain-core is installed for this test suite."""
        from routesmith.integrations.langchain import ChatRouteSmith  # noqa: F811
        assert ChatRouteSmith is not None


# ---------------------------------------------------------------------------
# bind_tools
# ---------------------------------------------------------------------------


class TestBindTools:
    """Tests for bind_tools with LangChain tool schemas."""

    def test_bind_tools_with_langchain_tool(self):
        """Test bind_tools converts @tool functions to OpenAI format."""
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"72°F in {city}"

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )
        bound = llm.bind_tools([get_weather])
        # bound should be a Runnable with tools in kwargs
        assert bound is not None

    @patch("routesmith.client.litellm")
    def test_bind_tools_passes_tools_to_litellm(self, mock_litellm):
        """Test that bind_tools passes formatted tools through to litellm."""
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )

        mock_response = MagicMock()
        mock_response.model = "test"
        mock_response.choices[0].message.content = "4"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_litellm.completion.return_value = mock_response

        bound = llm.bind_tools([add])
        bound.invoke([HumanMessage(content="What is 2+2?")])

        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "add"

    def test_bind_tools_with_tool_choice(self):
        """Test bind_tools respects tool_choice parameter."""
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )
        bound = llm.bind_tools([search], tool_choice="any")
        assert bound is not None


# ---------------------------------------------------------------------------
# LangGraph agent integration
# ---------------------------------------------------------------------------


def _make_litellm_response(content="", tool_calls=None, model="test"):
    """Helper to create a mock litellm response."""
    response = MagicMock()
    response.model = model
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls
    response.choices[0].finish_reason = "tool_calls" if tool_calls else "stop"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 10
    response.usage.total_tokens = 30
    return response


class TestLangGraphAgent:
    """Tests for ChatRouteSmith with LangGraph prebuilt agents."""

    @patch("routesmith.client.litellm")
    def test_create_react_agent_construction(self, mock_litellm):
        """Test that create_react_agent accepts ChatRouteSmith."""
        import warnings
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = create_react_agent(llm, tools=[add])

        # Agent should be a compiled graph
        assert agent is not None

    @patch("routesmith.client.litellm")
    def test_react_agent_tool_call_loop(self, mock_litellm):
        """Test full ReAct loop: LLM calls tool -> tool executes -> LLM responds."""
        import warnings
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )

        # First call: LLM decides to call the multiply tool
        tc = MagicMock()
        tc.function.name = "multiply"
        tc.function.arguments = '{"a": 6, "b": 7}'
        tc.id = "call_001"
        tool_call_response = _make_litellm_response(
            content="", tool_calls=[tc]
        )

        # Second call: LLM produces final answer after seeing tool result
        final_response = _make_litellm_response(
            content="The answer is 42."
        )

        mock_litellm.completion.side_effect = [tool_call_response, final_response]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = create_react_agent(llm, tools=[multiply])

        result = agent.invoke({"messages": [{"role": "user", "content": "What is 6 * 7?"}]})

        # Verify the agent completed the loop
        messages = result["messages"]

        # Should have: HumanMessage, AIMessage (tool call), ToolMessage, AIMessage (final)
        assert len(messages) >= 4

        # Final message should be the answer
        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)
        assert "42" in final_msg.content

        # Tool message should contain the actual multiplication result
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "42" in tool_msgs[0].content

        # litellm.completion was called twice (tool call + final answer)
        assert mock_litellm.completion.call_count == 2

    @patch("routesmith.client.litellm")
    def test_react_agent_no_tool_needed(self, mock_litellm):
        """Test ReAct agent when LLM answers directly without tools."""
        import warnings
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )

        # LLM responds directly without calling any tools
        direct_response = _make_litellm_response(content="Hello! How can I help?")
        mock_litellm.completion.return_value = direct_response

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = create_react_agent(llm, tools=[search])

        result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
        messages = result["messages"]

        # Should have: HumanMessage, AIMessage (direct response)
        assert len(messages) == 2
        assert isinstance(messages[-1], AIMessage)
        assert messages[-1].content == "Hello! How can I help?"
        assert mock_litellm.completion.call_count == 1

    @patch("routesmith.client.litellm")
    def test_react_agent_multi_tool_calls(self, mock_litellm):
        """Test ReAct agent with parallel tool calls in a single response."""
        import warnings
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"72°F in {city}"

        @tool
        def get_time(city: str) -> str:
            """Get time in a city."""
            return f"3:00 PM in {city}"

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )

        # LLM calls both tools in parallel
        tc1 = MagicMock()
        tc1.function.name = "get_weather"
        tc1.function.arguments = '{"city": "NYC"}'
        tc1.id = "call_w1"
        tc2 = MagicMock()
        tc2.function.name = "get_time"
        tc2.function.arguments = '{"city": "NYC"}'
        tc2.id = "call_t1"

        tool_call_response = _make_litellm_response(
            content="", tool_calls=[tc1, tc2]
        )
        final_response = _make_litellm_response(
            content="In NYC it's 72°F and 3:00 PM."
        )
        mock_litellm.completion.side_effect = [tool_call_response, final_response]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = create_react_agent(llm, tools=[get_weather, get_time])

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Weather and time in NYC?"}]}
        )
        messages = result["messages"]

        # Should have: Human, AI (2 tool calls), ToolMessage x2, AI (final)
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        assert any("72°F" in m.content for m in tool_msgs)
        assert any("3:00 PM" in m.content for m in tool_msgs)

        final_msg = messages[-1]
        assert isinstance(final_msg, AIMessage)
        assert "72°F" in final_msg.content

    @patch("routesmith.client.litellm")
    def test_react_agent_tools_kwarg_forwarded(self, mock_litellm):
        """Test that tools are forwarded to litellm.completion via kwargs."""
        import warnings
        from langgraph.prebuilt import create_react_agent
        from langchain_core.tools import tool

        @tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return str(eval(expression))

        llm = ChatRouteSmith()
        llm.routesmith.register_model(
            "test", cost_per_1k_input=0.001, cost_per_1k_output=0.002
        )

        direct_response = _make_litellm_response(content="Sure!")
        mock_litellm.completion.return_value = direct_response

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = create_react_agent(llm, tools=[calculator])

        agent.invoke({"messages": [{"role": "user", "content": "Hi"}]})

        # Verify tools were passed to litellm
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert "tools" in call_kwargs
        tools = call_kwargs["tools"]
        assert any(t["function"]["name"] == "calculator" for t in tools)
