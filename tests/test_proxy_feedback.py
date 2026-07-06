"""Tests for proxy feedback endpoint POST /v1/feedback."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routesmith import RouteSmith
from routesmith.proxy.handler import ChatCompletionRequest, RequestHandler


class TestFeedbackHandler:
    """Tests for /v1/feedback endpoint."""

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
    @patch("routesmith.client.litellm")
    async def test_feedback_updates_predictor(self, mock_litellm, routesmith, handler):
        """Test that feedback updates the predictor's update counter."""
        mock_message = MagicMock()
        mock_message.content = "Hi!"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {
            "id": "test",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "Hi!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        request = ChatCompletionRequest(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = await handler.handle_completion(request)

        request_id = result["routesmith_metadata"]["request_id"]

        body = json.dumps({"request_id": request_id, "score": 1.0}).encode()
        response, status = await handler.handle_feedback(body)

        assert status == 200
        assert response["status"] == "ok"
        assert response["request_id"] == request_id
        assert routesmith.router.predictor._total_updates == 1

    @pytest.mark.asyncio
    async def test_feedback_unknown_request_id_404(self, handler):
        """Test that unknown request_id returns 404."""
        body = json.dumps({"request_id": "nonexistent", "score": 0.5}).encode()
        response, status = await handler.handle_feedback(body)

        assert status == 404
        assert "Unknown request_id" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_feedback_requires_exactly_one_of_score_success(self, handler):
        """Test that exactly one of score/success is required."""
        body = json.dumps({"request_id": "test-id"}).encode()
        response, status = await handler.handle_feedback(body)
        assert status == 400
        assert "exactly one" in response["error"]["message"].lower()

        body = json.dumps({"request_id": "test-id", "score": 0.5, "success": True}).encode()
        response, status = await handler.handle_feedback(body)
        assert status == 400
        assert "exactly one" in response["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_feedback_score_out_of_range_400(self, handler):
        """Test that score out of [0,1] returns 400."""
        body = json.dumps({"request_id": "test-id", "score": 1.5}).encode()
        response, status = await handler.handle_feedback(body)

        assert status == 400
        assert "must be a number in [0, 1]" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_feedback_invalid_json_400(self, handler):
        """Test that invalid JSON returns 400."""
        body = b"not json"
        response, status = await handler.handle_feedback(body)

        assert status == 400
        assert "Invalid JSON" in response["error"]["message"]
