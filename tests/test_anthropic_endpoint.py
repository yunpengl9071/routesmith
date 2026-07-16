"""Tests for the Anthropic-compatible /v1/messages proxy endpoint."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routesmith.proxy.anthropic_compat import (
    AnthropicSSEStream,
    anthropic_to_internal,
    internal_to_anthropic,
)

# ---------------------------------------------------------------------------
# anthropic_to_internal
# ---------------------------------------------------------------------------

class TestAnthropicToInternal:
    def test_basic_translation(self):
        data = {
            "model": "auto",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        messages, kwargs = anthropic_to_internal(data)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert kwargs["max_tokens"] == 100
        assert kwargs["model"] == "auto"

    def test_system_prompt_translated(self):
        data = {
            "model": "auto",
            "max_tokens": 50,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        messages, _ = anthropic_to_internal(data)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    def test_system_as_list(self):
        data = {
            "max_tokens": 50,
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        messages, _ = anthropic_to_internal(data)
        assert messages[0]["content"] == "Be concise."

    def test_content_blocks_concatenated(self):
        data = {
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part. "},
                    {"type": "text", "text": "Second part."},
                ],
            }],
        }
        messages, _ = anthropic_to_internal(data)
        assert "First part." in messages[0]["content"]
        assert "Second part." in messages[0]["content"]

    def test_image_block_converted(self):
        data = {
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                ],
            }],
        }
        messages, kwargs = anthropic_to_internal(data)
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Describe this:"
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64,abc" in content[1]["image_url"]["url"]
    # changed per 2026-07-16 integration-dx spec §9.1

    def test_missing_max_tokens_raises(self):
        data = {
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with pytest.raises(ValueError, match="max_tokens"):
            anthropic_to_internal(data)

    def test_missing_messages_raises(self):
        data = {"max_tokens": 50}
        with pytest.raises(ValueError, match="messages"):
            anthropic_to_internal(data)

    def test_temperature_and_top_p_passed(self):
        data = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9

    def test_stop_sequences_passed(self):
        data = {
            "max_tokens": 100,
            "stop_sequences": ["\n\n", "END"],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert kwargs["stop"] == ["\n\n", "END"]

    def test_stream_flag(self):
        data = {
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert kwargs["stream"] is True


# ---------------------------------------------------------------------------
# internal_to_anthropic
# ---------------------------------------------------------------------------

class TestInternalToAnthropic:
    def _make_mock_response(self, content="Hello!", finish="stop", model="gpt-4o-mini"):
        return MagicMock(
            choices=[MagicMock(
                message=MagicMock(content=content, tool_calls=None),
                finish_reason=finish,
            )],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            model=model,
        )

    def test_basic_response_shape(self):
        resp = self._make_mock_response()
        result = internal_to_anthropic(resp, request_model="auto", request_id="req_123")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["id"] == "msg_req_123"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_stop_reason_mapping(self):
        resp = self._make_mock_response(finish="length")
        result = internal_to_anthropic(resp, request_model="auto")
        assert result["stop_reason"] == "max_tokens"

        resp2 = self._make_mock_response(finish="stop")
        result2 = internal_to_anthropic(resp2, request_model="auto")
        assert result2["stop_reason"] == "end_turn"

    def test_routesmith_metadata_included(self):
        resp = self._make_mock_response()
        resp.routesmith_metadata = {"model_selected": "gpt-4o-mini"}
        result = internal_to_anthropic(resp, request_model="auto")
        assert result["routesmith_metadata"]["model_selected"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# AnthropicSSEStream
# ---------------------------------------------------------------------------

class TestAnthropicSSEStream:
    def test_streaming_event_sequence(self):
        request_id = "req_abc123"
        stream = AnthropicSSEStream(request_id=request_id, request_model="claude-sonnet-4")

        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]},
        ]

        events = stream.iter_chunks(chunks)
        event_names = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("event: "):
                    event_names.append(line.split(": ")[1])

        assert event_names[0] == "message_start"
        assert event_names[1] == "content_block_start"
        assert "content_block_delta" in event_names
        assert event_names[-2] == "message_delta"
        assert event_names[-1] == "message_stop"

    def test_message_start_has_request_id(self):
        stream = AnthropicSSEStream(request_id="test_req", request_model="auto")
        events = stream.iter_chunks([])
        assert any("test_req" in e for e in events)


# ---------------------------------------------------------------------------
# Proxy-level integration tests (litellm mocked)
# ---------------------------------------------------------------------------

class TestAnthropicEndpointIntegration:
    """Test /v1/messages through the proxy handler with litellm mocked."""

    @patch("routesmith.client.litellm")
    async def test_proxy_anthropic_completion(self, mock_litellm):
        from routesmith import RouteSmith
        from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_message = MagicMock()
        mock_message.content = "Hello from Anthropic!"
        mock_message.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o-mini"
        mock_response.id = "resp_mock"
        mock_response.model_dump.return_value = {
            "choices": [{
                "message": {"content": "Hello from Anthropic!", "role": "assistant"},
                "finish_reason": "stop",
                "index": 0,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4o-mini",
            "id": "resp_mock",
        }
        mock_response.routesmith_metadata = {}
        mock_litellm.completion = MagicMock(return_value=mock_response)
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.15, cost_per_1k_output=0.60, quality_score=0.85)

        server = RouteSmithProxyServer(rs, ServerConfig(port=0))

        body = json.dumps({
            "model": "auto",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }).encode()

        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock(return_value=None)

        # Mock _send_json
        sent_data = {}

        async def capture_json(w, data, status):
            sent_data["data"] = data
            sent_data["status"] = status

        server._send_json = capture_json

        await server._handle_anthropic_messages(writer, body)

        resp = sent_data.get("data", {})
        assert resp.get("type") == "message"
        assert resp.get("content", [{}])[0].get("text") == "Hello from Anthropic!"
        assert resp.get("stop_reason") == "end_turn"


# ---------------------------------------------------------------------------
# Tool-use tests (R4)
# ---------------------------------------------------------------------------

class TestAnthropicTools:
    """Tests for tool-use translation in anthropic_to_internal."""

    def test_tools_param_converted(self):
        data = {
            "model": "auto",
            "max_tokens": 100,
            "tools": [
                {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}},
            ],
            "messages": [{"role": "user", "content": "Weather?"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["type"] == "function"
        assert kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert kwargs["tools"][0]["function"]["parameters"]["type"] == "object"

    def test_tool_choice_auto(self):
        data = {
            "max_tokens": 50,
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert kwargs["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        data = {
            "max_tokens": 50,
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert kwargs["tool_choice"] == "required"

    def test_tool_choice_tool(self):
        data = {
            "max_tokens": 50,
            "tool_choice": {"type": "tool", "name": "get_weather"},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, kwargs = anthropic_to_internal(data)
        assert kwargs["tool_choice"]["type"] == "function"
        assert kwargs["tool_choice"]["function"]["name"] == "get_weather"

    def test_assistant_tool_use_block(self):
        data = {
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check:"},
                        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"loc": "NYC"}},
                    ],
                },
            ],
        }
        messages, _ = anthropic_to_internal(data)
        assert len(messages) == 3
        # First message: user
        assert messages[0]["role"] == "user"
        # Second message: assistant text
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Let me check:"
        # Third message: assistant tool_calls
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] is None
        assert len(messages[2]["tool_calls"]) == 1
        tc = messages[2]["tool_calls"][0]
        assert tc["id"] == "tu_1"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"loc": "NYC"}

    def test_user_tool_result_block(self):
        data = {
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "Sunny, 72F"},
                    ],
                },
            ],
        }
        messages, _ = anthropic_to_internal(data)
        tool_msg = messages[-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "tu_1"
        assert tool_msg["content"] == "Sunny, 72F"

    def test_user_mixed_text_and_tool_result(self):
        data = {
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "Result"},
                        {"type": "text", "text": "Follow up"},
                    ],
                },
            ],
        }
        messages, _ = anthropic_to_internal(data)
        # tool message first, then user text
        assert messages[-2]["role"] == "tool"
        assert messages[-2]["tool_call_id"] == "tu_1"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Follow up"

    def test_thinking_block_dropped(self):
        data = {
            "max_tokens": 50,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "answer"},
                ]},
            ],
        }
        messages, _ = anthropic_to_internal(data)
        assert messages[0]["content"] == "answer"

    def test_unsupported_block_type_raises(self):
        data = {
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": [{"type": "unsupported_thing", "data": "x"}],
            }],
        }
        with pytest.raises(ValueError, match="unsupported_thing"):
            anthropic_to_internal(data)

    def test_tool_result_flattens_list_content(self):
        data = {
            "max_tokens": 50,
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}]},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": "Result text"}],
                    }],
                },
            ],
        }
        messages, _ = anthropic_to_internal(data)
        assert messages[-1]["content"] == "Result text"


class TestInternalToAnthropicTools:
    """Tests for tool_calls in response translation."""

    def test_tool_calls_converted(self):
        from types import SimpleNamespace
        tc = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="get_weather", arguments='{"loc": "NYC"}'),
        )
        resp = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="Checking...", tool_calls=[tc]),
                finish_reason="tool_calls",
            )],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
            model="claude-sonnet",
        )
        result = internal_to_anthropic(resp, request_model="claude-sonnet")
        assert result["stop_reason"] == "tool_use"
        blocks = result["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Checking..."
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "call_abc"
        assert blocks[1]["name"] == "get_weather"
        assert blocks[1]["input"] == {"loc": "NYC"}

    def test_malformed_json_arguments(self):
        from types import SimpleNamespace
        tc = SimpleNamespace(
            id="call_xyz",
            function=SimpleNamespace(name="bad_func", arguments="not valid json{{{"),
        )
        resp = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="", tool_calls=[tc]),
                finish_reason="tool_calls",
            )],
            usage=MagicMock(prompt_tokens=5, completion_tokens=3),
            model="test",
        )
        result = internal_to_anthropic(resp, request_model="test")
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["input"] == {}  # fallback to empty dict

    def test_mixed_text_and_tool_calls(self):
        from types import SimpleNamespace
        tc = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="f1", arguments="{}"),
        )
        resp = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="Text then tool.", tool_calls=[tc]),
                finish_reason="tool_calls",
            )],
            usage=MagicMock(prompt_tokens=5, completion_tokens=3),
            model="test",
        )
        result = internal_to_anthropic(resp, request_model="test")
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"


class TestAnthropicSSEStreamTools:
    """Tests for streaming tool-use events."""

    def test_tool_call_stream_start_events(self):
        stream = AnthropicSSEStream(request_id="req_1", request_model="claude-sonnet")
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc":'}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "NYC"}'}}]}, "finish_reason": "tool_calls"}]},
        ]
        events = stream.iter_chunks(chunks)
        event_sequence = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("event: "):
                    event_sequence.append(line.split(": ")[1])

        assert event_sequence[0] == "message_start"
        assert event_sequence[1] == "content_block_start"
        assert event_sequence[2] == "content_block_delta"
        assert event_sequence[3] == "content_block_delta"
        assert event_sequence[4] == "content_block_stop"
        assert event_sequence[5] == "message_delta"
        assert event_sequence[6] == "message_stop"

    def test_tool_call_content_block_data(self):
        stream = AnthropicSSEStream(request_id="req_2", request_model="claude-sonnet")
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_x", "function": {"name": "search", "arguments": ""}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":'}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = stream.iter_chunks(chunks)

        tool_start_found = False
        json_delta_found = False
        stop_found = False
        for e in events:
            if e.startswith("event: content_block_start") and '"tool_use"' in e:
                data = json.loads([line for line in e.split("\n") if line.startswith("data:")][0][6:])
                assert data["content_block"]["id"] == "call_x"
                assert data["content_block"]["name"] == "search"
                assert data["content_block"]["input"] == {}
                tool_start_found = True
            if e.startswith("event: content_block_delta") and '"input_json_delta"' in e:
                json_delta_found = True
            if e.startswith("event: message_delta") and '"tool_use"' in e:
                data = json.loads([line for line in e.split("\n") if line.startswith("data:")][0][6:])
                assert data["delta"]["stop_reason"] == "tool_use"
                stop_found = True

        assert tool_start_found
        assert json_delta_found
        assert stop_found

    def test_text_and_tool_interleaving(self):
        stream = AnthropicSSEStream(request_id="req_3", request_model="claude-sonnet")
        chunks = [
            {"choices": [{"delta": {"content": "Thinking"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "..."}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run", "arguments": "{}"}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        events = stream.iter_chunks(chunks)
        event_names = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("event: "):
                    event_names.append(line.split(": ")[1])

        assert event_names[0] == "message_start"
        assert event_names[1] == "content_block_start"
        assert event_names[2] == "content_block_delta"
        assert event_names[3] == "content_block_delta"
        assert event_names[4] == "content_block_start"
        assert "content_block_stop" in event_names


class TestAnthropicToolsIntegration:
    """End-to-end test of tool-use round-trip through proxy."""

    @patch("routesmith.client.litellm")
    async def test_tool_use_round_trip_non_streaming(self, mock_litellm):
        from routesmith import RouteSmith
        from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [
            MagicMock(
                id="call_abc",
                function=MagicMock(name="get_weather", arguments='{"loc": "NYC"}'),
            )
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model = "gpt-4o-mini"
        mock_response.model_dump.return_value = {
            "choices": [{
                "message": {"content": "", "role": "assistant",
                            "tool_calls": [{"id": "call_abc", "type": "function",
                                           "function": {"name": "get_weather", "arguments": '{"loc": "NYC"}'}}]},
                "finish_reason": "tool_calls",
                "index": 0,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4o-mini",
            "id": "resp_mock",
        }
        mock_response.routesmith_metadata = {}
        mock_litellm.completion = MagicMock(return_value=mock_response)
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.15, cost_per_1k_output=0.60, quality_score=0.85)

        server = RouteSmithProxyServer(rs, ServerConfig(port=0))

        body = json.dumps({
            "model": "auto",
            "max_tokens": 100,
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "messages": [
                {"role": "user", "content": "Weather in NYC?"},
            ],
        }).encode()

        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock(return_value=None)

        sent_data = {}
        async def capture_json(w, data, status):
            sent_data["data"] = data
            sent_data["status"] = status
        server._send_json = capture_json

        await server._handle_anthropic_messages(writer, body)

        resp = sent_data.get("data", {})
        assert resp.get("type") == "message"
        blocks = resp.get("content", [])
        assert any(b.get("type") == "tool_use" for b in blocks)
        tool_block = [b for b in blocks if b["type"] == "tool_use"][0]
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"loc": "NYC"}
        assert resp.get("stop_reason") == "tool_use"

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"
