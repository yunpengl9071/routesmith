"""Real HTTP transport integration tests.

Starts the proxy server in-process on a real port, sends real HTTP/1.1
requests via httpx. Only litellm is mocked — the HTTP transport layer,
routing decisions, response formatting, and error handling are real.

Run: pytest tests/test_proxy_transport.py -x -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig
from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig


@pytest.fixture
async def proxy_server():
    rs = RouteSmith(config=RouteSmithConfig(intercept="all"))
    rs.register_model(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
    )
    rs.register_model(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
    )
    server = RouteSmithProxyServer(rs, ServerConfig(port=0))
    await server.start()
    port = server.port
    yield port, rs
    await server.stop()


@pytest.fixture
def mock_litellm():
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.model_dump.return_value = {
        "id": "cmpl_mock",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"content": "Hello from RouteSmith!"}, "finish_reason": "stop", "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    mock_response.routesmith_metadata = {}
    mock_response.choices[0].message.content = "Hello from RouteSmith!"
    mock_response.choices[0].finish_reason = "stop"

    with patch("routesmith.client.litellm") as mock:
        mock.acompletion = AsyncMock(return_value=mock_response)
        yield mock


class TestProxyRealHTTPTransport:
    """Real HTTP/1.1 transport tests against an in-process proxy server."""

    @pytest.mark.asyncio
    async def test_chat_completion_routed(self, proxy_server, mock_litellm):
        """POST /v1/chat/completions with concrete model under intercept: all routes it."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routesmith_metadata"]["routed"] is True
        assert data["routesmith_metadata"]["requested_model"] == "gpt-4o"
        assert data["routesmith_metadata"]["selected_model"] == "gpt-4o-mini"
        assert data["routesmith_metadata"]["passthrough_reason"] is None

    @pytest.mark.asyncio
    async def test_chat_completion_passthrough_header(self, proxy_server, mock_litellm):
        """X-RouteSmith-Passthrough header bypasses routing."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"X-RouteSmith-Passthrough": "true"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routesmith_metadata"]["routed"] is False
        assert data["routesmith_metadata"]["passthrough_reason"] == "explicit_header"

    @pytest.mark.asyncio
    async def test_chat_completion_unregistered_model(self, proxy_server, mock_litellm):
        """Unregistered model passes through under intercept: all."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routesmith_metadata"]["routed"] is False
        assert data["routesmith_metadata"]["passthrough_reason"] == "unregistered_model"

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, proxy_server, mock_litellm):
        """GET /v1/stats returns proxy counters."""
        port, rs = proxy_server

        # Make a routed request to increment counters
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            )

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["routed_requests"] >= 1
        assert "passthrough_requests" in data
        assert "passthrough_by_reason" in data

    @pytest.mark.asyncio
    async def test_health_endpoint(self, proxy_server, mock_litellm):
        """GET /health returns ok."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("ok", "healthy")

    @pytest.mark.asyncio
    async def test_models_endpoint(self, proxy_server, mock_litellm):
        """GET /v1/models returns registered models list."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "gpt-4o-mini" in model_ids
        assert "gpt-4o" in model_ids

    @pytest.mark.asyncio
    async def test_auto_model_always_routes(self, proxy_server, mock_litellm):
        """Model 'auto' always routes regardless of intercept setting."""
        port, rs = proxy_server
        rs.config.intercept = "auto"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routesmith_metadata"]["routed"] is True

    @pytest.mark.asyncio
    async def test_intercept_auto_passthrough(self, proxy_server, mock_litellm):
        """Under intercept: auto, concrete model passes through."""
        port, rs = proxy_server
        rs.config.intercept = "auto"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routesmith_metadata"]["routed"] is False
        assert data["routesmith_metadata"]["passthrough_reason"] == "intercept_auto"

    @pytest.mark.asyncio
    async def test_auth_required(self, proxy_server, mock_litellm):
        """Proxy with api_key requires Bearer auth."""
        port, rs = proxy_server
        rs.config = RouteSmithConfig(intercept="all")
        server = RouteSmithProxyServer(rs, ServerConfig(port=0, api_key="secret"))
        await server.start()
        auth_port = server.port

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{auth_port}/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
            )
        assert resp.status_code == 401

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{auth_port}/v1/chat/completions",
                json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer secret"},
            )
        assert resp.status_code == 200

        await server.stop()


class TestAnthropicRealHTTPTransport:
    """Real HTTP tests for /v1/messages endpoint."""

    @pytest.mark.asyncio
    async def test_messages_endpoint_text_only(self, proxy_server, mock_litellm):
        """POST /v1/messages with text-only request returns Anthropic-format response."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/messages",
                json={
                    "model": "auto",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "text"
        assert "routesmith_metadata" in data

    @pytest.mark.asyncio
    async def test_messages_endpoint_routed_metadata(self, proxy_server, mock_litellm):
        """routesmith_metadata on /v1/messages includes routing info."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/messages",
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        meta = data["routesmith_metadata"]
        assert meta["routed"] is False
        assert meta["passthrough_reason"] == "unregistered_model"

    @pytest.mark.asyncio
    async def test_count_tokens_estimate(self, proxy_server, mock_litellm, monkeypatch):
        """POST /v1/messages/count_tokens returns estimate when no API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/messages/count_tokens",
                json={
                    "messages": [{"role": "user", "content": "Hello world"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "input_tokens" in data
        assert isinstance(data["input_tokens"], int)
        assert data["input_tokens"] >= 1

    @pytest.mark.asyncio
    async def test_404_unknown_path(self, proxy_server, mock_litellm):
        """Unknown path returns 404."""
        port, _ = proxy_server
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://127.0.0.1:{port}/v1/unknown")
        assert resp.status_code == 404
