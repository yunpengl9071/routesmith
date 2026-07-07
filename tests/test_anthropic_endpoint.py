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

    def test_image_block_rejected_clearly(self):
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
        with pytest.raises(ValueError, match="image"):
            anthropic_to_internal(data)

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
