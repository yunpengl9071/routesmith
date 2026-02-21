"""Tests for proxy server components."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from routesmith import RouteSmith
from routesmith.proxy.handler import RequestHandler, ChatCompletionRequest
from routesmith.proxy.responses import (
    format_chat_completion,
    format_stream_chunk,
    format_stream_done,
    format_error,
    format_models_list,
)


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest parsing."""

    def test_from_dict_minimal(self):
        """Test parsing with minimal required fields."""
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = ChatCompletionRequest.from_dict(data)

        assert request.model == "gpt-4o"
        assert len(request.messages) == 1
        assert request.messages[0]["content"] == "Hello"
        assert request.stream is False
        assert request.temperature == 1.0
        assert request.routesmith_strategy is None

    def test_from_dict_with_all_fields(self):
        """Test parsing with all fields."""
        data = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stream": True,
            "routesmith_strategy": "cascade",
            "routesmith_min_quality": 0.9,
            "routesmith_max_cost": 0.01,
        }
        request = ChatCompletionRequest.from_dict(data)

        assert request.model == "auto"
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stream is True
        assert request.routesmith_strategy == "cascade"
        assert request.routesmith_min_quality == 0.9
        assert request.routesmith_max_cost == 0.01

    def test_from_dict_default_model(self):
        """Test default model is 'auto'."""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.model == "auto"

    def test_from_dict_missing_messages(self):
        """Test error on missing messages."""
        data = {"model": "gpt-4o"}
        with pytest.raises(ValueError, match="'messages' is required"):
            ChatCompletionRequest.from_dict(data)

    def test_from_dict_invalid_messages(self):
        """Test error on invalid messages format."""
        data = {"messages": "not a list"}
        with pytest.raises(ValueError, match="'messages' must be a list"):
            ChatCompletionRequest.from_dict(data)

    def test_from_dict_invalid_message_format(self):
        """Test error on invalid message object."""
        data = {"messages": [{"content": "no role"}]}
        with pytest.raises(ValueError, match="must have a 'role' field"):
            ChatCompletionRequest.from_dict(data)

    def test_to_litellm_kwargs(self):
        """Test conversion to litellm kwargs."""
        request = ChatCompletionRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=50,
            stop=["END"],
        )
        kwargs = request.to_litellm_kwargs()

        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 50
        assert kwargs["stop"] == ["END"]

    def test_extra_kwargs_preserved(self):
        """Test that unknown fields are preserved in extra_kwargs."""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "custom_field": "custom_value",
            "another_field": 123,
        }
        request = ChatCompletionRequest.from_dict(data)

        assert request.extra_kwargs["custom_field"] == "custom_value"
        assert request.extra_kwargs["another_field"] == 123


class TestRequestHandler:
    """Tests for RequestHandler."""

    @pytest.fixture
    def routesmith(self):
        """Create RouteSmith with test models."""
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
        return rs

    @pytest.fixture
    def handler(self, routesmith):
        """Create RequestHandler."""
        return RequestHandler(routesmith)

    @pytest.mark.asyncio
    async def test_handle_models(self, handler):
        """Test /v1/models endpoint."""
        result = await handler.handle_models()

        assert result["object"] == "list"
        assert len(result["data"]) == 3  # 2 models + auto
        model_ids = [m["id"] for m in result["data"]]
        assert "gpt-4o" in model_ids
        assert "gpt-4o-mini" in model_ids
        assert "routesmith/auto" in model_ids

    @pytest.mark.asyncio
    async def test_handle_stats(self, handler):
        """Test /v1/stats endpoint."""
        result = await handler.handle_stats()

        assert "request_count" in result
        assert "total_cost_usd" in result
        assert "cost_savings_usd" in result
        assert result["registered_models"] == 2

    @pytest.mark.asyncio
    async def test_handle_health(self, handler):
        """Test /health endpoint."""
        result = await handler.handle_health()

        assert result["status"] == "healthy"
        assert result["registered_models"] == 2

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_handle_completion_auto_routing(self, mock_litellm, handler):
        """Test completion with auto routing."""
        # Mock response
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {
            "id": "test",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "Hi!"}}],
        }
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        request = ChatCompletionRequest(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = await handler.handle_completion(request)

        assert result["model"] == "gpt-4o-mini"
        # Verify routing happened (called without explicit model)
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        # Model should be set by routing
        assert "model" in call_kwargs or mock_litellm.acompletion.call_args.args

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_handle_completion_explicit_model(self, mock_litellm, handler):
        """Test completion with explicit model (bypass routing)."""
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {
            "id": "test",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hi!"}}],
        }
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        request = ChatCompletionRequest(
            model="gpt-4o",  # Explicit model
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = await handler.handle_completion(request)

        # Explicit model should be used
        call_args = mock_litellm.acompletion.call_args
        assert call_args.kwargs.get("model") == "gpt-4o"


class TestResponseFormatting:
    """Tests for response formatting functions."""

    def test_format_chat_completion(self):
        """Test chat completion response formatting."""
        result = format_chat_completion(
            content="Hello!",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4o-mini"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert "id" in result
        assert result["id"].startswith("chatcmpl-")

    def test_format_chat_completion_with_metadata(self):
        """Test chat completion with RouteSmith metadata."""
        metadata = {"model_selected": "gpt-4o-mini", "cost_savings_usd": 0.001}
        result = format_chat_completion(
            content="Hello!",
            model="gpt-4o-mini",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            routesmith_metadata=metadata,
        )

        assert "routesmith_metadata" in result
        assert result["routesmith_metadata"]["model_selected"] == "gpt-4o-mini"

    def test_format_stream_chunk(self):
        """Test streaming chunk formatting."""
        result = format_stream_chunk(content="Hello", model="gpt-4o-mini")

        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        data = json.loads(result[6:-2])
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_format_stream_chunk_final(self):
        """Test final streaming chunk with finish_reason."""
        result = format_stream_chunk(content="", model="gpt-4o-mini", finish_reason="stop")

        data = json.loads(result[6:-2])
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_format_stream_done(self):
        """Test stream done marker."""
        result = format_stream_done()
        assert result == "data: [DONE]\n\n"

    def test_format_error(self):
        """Test error response formatting."""
        error_body, status = format_error("Something went wrong", status_code=400)

        assert status == 400
        assert error_body["error"]["message"] == "Something went wrong"
        assert error_body["error"]["type"] == "invalid_request_error"

    def test_format_error_with_code(self):
        """Test error response with custom code."""
        error_body, _ = format_error(
            "Invalid model",
            error_type="model_not_found",
            code="INVALID_MODEL",
        )

        assert error_body["error"]["type"] == "model_not_found"
        assert error_body["error"]["code"] == "INVALID_MODEL"

    def test_format_models_list(self):
        """Test models list formatting."""
        models = [
            {"id": "gpt-4o"},
            {"id": "gpt-4o-mini", "owned_by": "openai"},
        ]
        result = format_models_list(models)

        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "gpt-4o"
        assert result["data"][0]["object"] == "model"
        assert result["data"][1]["owned_by"] == "openai"

    def test_format_models_list_empty(self):
        """Test models list formatting with empty list."""
        result = format_models_list([])
        assert result["object"] == "list"
        assert result["data"] == []


class TestChatCompletionRequestEdgeCases:
    """Edge case tests for ChatCompletionRequest."""

    def test_empty_messages_list(self):
        """Test parsing with empty messages list - should not raise."""
        data = {"messages": []}
        request = ChatCompletionRequest.from_dict(data)
        assert request.messages == []

    def test_multi_turn_conversation(self):
        """Test parsing multi-turn conversation."""
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ]
        }
        request = ChatCompletionRequest.from_dict(data)
        assert len(request.messages) == 4
        assert request.messages[0]["role"] == "system"
        assert request.messages[3]["role"] == "user"

    def test_message_with_extra_fields(self):
        """Test message with extra fields like name."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello", "name": "Alice"}
            ]
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.messages[0]["name"] == "Alice"

    def test_unicode_content(self):
        """Test unicode content handling."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello 你好 مرحبا 🎉"}
            ]
        }
        request = ChatCompletionRequest.from_dict(data)
        assert "你好" in request.messages[0]["content"]
        assert "🎉" in request.messages[0]["content"]

    def test_very_long_content(self):
        """Test handling very long content."""
        long_content = "x" * 100000
        data = {
            "messages": [{"role": "user", "content": long_content}]
        }
        request = ChatCompletionRequest.from_dict(data)
        assert len(request.messages[0]["content"]) == 100000

    def test_temperature_boundaries(self):
        """Test temperature at boundaries."""
        data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.0
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.temperature == 0.0

        data["temperature"] = 2.0
        request = ChatCompletionRequest.from_dict(data)
        assert request.temperature == 2.0

    def test_null_values(self):
        """Test null values for optional fields."""
        data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": None,
            "stop": None,
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.max_tokens is None
        assert request.stop is None

    def test_stop_as_string(self):
        """Test stop as a single string (OpenAI allows both)."""
        data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.stop == "END"

    def test_stop_as_list(self):
        """Test stop as a list of strings."""
        data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["END", "STOP", "###"],
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.stop == ["END", "STOP", "###"]


class TestRequestHandlerEdgeCases:
    """Edge case tests for RequestHandler."""

    @pytest.fixture
    def empty_routesmith(self):
        """RouteSmith with no models."""
        return RouteSmith()

    @pytest.fixture
    def empty_handler(self, empty_routesmith):
        """Handler with no models."""
        return RequestHandler(empty_routesmith)

    @pytest.mark.asyncio
    async def test_handle_models_empty_registry(self, empty_handler):
        """Test /v1/models with no models registered."""
        result = await empty_handler.handle_models()
        # Should still include routesmith/auto
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "routesmith/auto"

    @pytest.mark.asyncio
    async def test_handle_stats_empty(self, empty_handler):
        """Test /v1/stats with no requests made."""
        result = await empty_handler.handle_stats()
        assert result["request_count"] == 0
        assert result["total_cost_usd"] == 0.0
        assert result["savings_percent"] == 0.0

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_handle_completion_with_invalid_strategy(self, mock_litellm):
        """Test completion with invalid routesmith_strategy."""
        rs = RouteSmith()
        rs.register_model("test", cost_per_1k_input=0.001, cost_per_1k_output=0.002)
        handler = RequestHandler(rs)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "test", "model": "test", "choices": []}
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        # Invalid strategy should be ignored, not raise
        request = ChatCompletionRequest(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}],
            routesmith_strategy="invalid_strategy",
        )
        # Should not raise, just use default strategy
        result = await handler.handle_completion(request)
        assert result is not None

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_handle_completion_preserves_kwargs(self, mock_litellm):
        """Test that extra kwargs are passed to litellm."""
        rs = RouteSmith()
        rs.register_model("test", cost_per_1k_input=0.001, cost_per_1k_output=0.002)
        handler = RequestHandler(rs)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "test", "model": "test", "choices": []}
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.5,
            max_tokens=100,
        )
        await handler.handle_completion(request)

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.5
        assert call_kwargs.get("max_tokens") == 100

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_handle_completion_routesmith_auto_variants(self, mock_litellm):
        """Test all 'auto' model name variants trigger routing."""
        rs = RouteSmith()
        rs.register_model("test", cost_per_1k_input=0.001, cost_per_1k_output=0.002, quality_score=0.9)
        handler = RequestHandler(rs)

        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "test", "model": "test", "choices": []}
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        # All these should trigger routing (not pass model directly)
        for model_name in ["auto", "Auto", "AUTO", "routesmith", "routesmith/auto"]:
            request = ChatCompletionRequest(
                model=model_name,
                messages=[{"role": "user", "content": "Hi"}],
            )
            await handler.handle_completion(request)
            # The model passed to litellm should be "test" (from routing), not the auto variant
            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs.get("model") != model_name


class TestResponseFormattingEdgeCases:
    """Edge case tests for response formatting."""

    def test_format_chat_completion_empty_content(self):
        """Test formatting with empty content."""
        result = format_chat_completion(
            content="",
            model="test",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        assert result["choices"][0]["message"]["content"] == ""

    def test_format_chat_completion_multiline_content(self):
        """Test formatting with multiline content."""
        content = "Line 1\nLine 2\nLine 3"
        result = format_chat_completion(
            content=content,
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )
        assert result["choices"][0]["message"]["content"] == content

    def test_format_stream_chunk_empty_content(self):
        """Test streaming chunk with empty content."""
        result = format_stream_chunk(content="", model="test")
        data = json.loads(result[6:-2])
        assert data["choices"][0]["delta"] == {}

    def test_format_error_all_status_codes(self):
        """Test error formatting with various status codes."""
        for status in [400, 401, 403, 404, 429, 500, 502, 503]:
            error_body, returned_status = format_error(f"Error {status}", status_code=status)
            assert returned_status == status
            assert error_body["error"]["message"] == f"Error {status}"

    def test_format_chat_completion_unique_ids(self):
        """Test that each response gets a unique ID."""
        ids = set()
        for _ in range(100):
            result = format_chat_completion(
                content="test",
                model="test",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )
            ids.add(result["id"])
        # All IDs should be unique
        assert len(ids) == 100

    def test_format_chat_completion_timestamp(self):
        """Test that created timestamp is reasonable."""
        import time
        before = int(time.time())
        result = format_chat_completion(
            content="test",
            model="test",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        after = int(time.time())
        assert before <= result["created"] <= after
