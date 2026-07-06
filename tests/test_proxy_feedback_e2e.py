"""End-to-end proxy feedback tests (G6)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routesmith import RouteSmith
from routesmith.config import BudgetConfig, RouteSmithConfig
from routesmith.proxy.handler import ChatCompletionRequest, RequestHandler


class TestE2EFeedback:
    """G6: End-to-end proxy feedback convergence."""

    @pytest.fixture
    def routesmith(self):
        """Create RouteSmith with quality_threshold=0.0 and no implicit feedback."""
        config = RouteSmithConfig(
            budget=BudgetConfig(quality_threshold=0.0),
            implicit_feedback_enabled=False,
        )
        rs = RouteSmith(config=config)
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
        return RequestHandler(routesmith)

    @pytest.fixture
    def mock_response(self):
        mock = MagicMock()
        mock.usage.prompt_tokens = 10
        mock.usage.completion_tokens = 20
        mock.model_dump.return_value = {
            "id": "test",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "Hi!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        return mock

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_e2e_feedback_basic(self, mock_litellm, mock_response, routesmith, handler):
        """Feedback via handler updates predictor and persists explicit signal."""
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        request = ChatCompletionRequest(
            model="auto",
            messages=[{"role": "user", "content": "Hello"}],
        )
        result = await handler.handle_completion(request)

        request_id = result["routesmith_metadata"]["request_id"]
        updates_before = routesmith.router.predictor._total_updates

        body = json.dumps({"request_id": request_id, "score": 1.0}).encode()
        response, status = await handler.handle_feedback(body)

        assert status == 200
        assert response["status"] == "ok"
        assert response["request_id"] == request_id
        assert routesmith.router.predictor._total_updates == updates_before + 1

        record = routesmith.feedback.get_record_by_id(request_id)
        assert record is not None
        assert record.quality_score == 1.0

    @pytest.mark.asyncio
    @patch("routesmith.client.litellm")
    async def test_e2e_feedback_shifts_routing(self, mock_litellm, mock_response, routesmith, handler):
        """60 feedback rounds rewarding gpt-4o-mini shifts routing toward it."""
        mock_litellm.acompletion = AsyncMock(return_value=mock_response)

        msg = [{"role": "user", "content": "hello"}]

        for _ in range(60):
            request = ChatCompletionRequest(
                model="auto",
                messages=msg,
                routesmith_min_quality=0.0,
            )
            result = await handler.handle_completion(request)
            rid = result["routesmith_metadata"]["request_id"]
            chosen = result["routesmith_metadata"]["model_selected"]
            score = 0.95 if chosen == "gpt-4o-mini" else 0.3
            body = json.dumps({"request_id": rid, "score": score}).encode()
            await handler.handle_feedback(body)

        gpt4o_mini_count = 0
        for _ in range(20):
            request = ChatCompletionRequest(
                model="auto",
                messages=msg,
                routesmith_min_quality=0.0,
            )
            result = await handler.handle_completion(request)
            chosen = result["routesmith_metadata"]["model_selected"]
            if chosen == "gpt-4o-mini":
                gpt4o_mini_count += 1

        assert gpt4o_mini_count >= 15, (
            f"gpt-4o-mini selected {gpt4o_mini_count}/20, expected >=15"
        )
