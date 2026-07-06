"""Tests for proxy server budget enforcement."""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from routesmith.config import BudgetConfig
from routesmith.proxy.server import RouteSmithProxyServer
from tests.helpers import make_rs


class TestProxyBudgetEnforcement:
    """Tests for 429 budget-exceeded response from proxy."""

    @pytest.fixture
    def over_budget_routesmith(self):
        """RouteSmith with daily budget limit and pre-seeded spend."""
        budget = BudgetConfig(max_cost_per_day=1.0)
        rs = make_rs(budget=budget)
        now = time.time()
        rs._budget.record(1.0, now=now)
        return rs

    @pytest.mark.asyncio
    async def test_over_budget_returns_429(self, over_budget_routesmith):
        """Over-budget completion returns 429 with budget_exceeded error."""
        server = RouteSmithProxyServer(over_budget_routesmith)

        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        body = json.dumps({
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode("utf-8")

        await server._handle_completion(writer, body)

        # Collect all writes
        written_data = b"".join(
            call[0][0] for call in writer.write.call_args_list
            if call[0] and isinstance(call[0][0], bytes)
        )
        response_text = written_data.decode("utf-8")
        assert "429" in response_text or "Too Many Requests" in response_text
        assert "budget_exceeded" in response_text
