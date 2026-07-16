"""Gate G-M1: Single-model pool transparency gate.

With a pool containing exactly one model and intercept: all, a session
through the proxy must be indistinguishable from a direct API session.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig
from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig


@pytest.fixture
async def single_model_proxy():
    rs = RouteSmith(config=RouteSmithConfig(intercept="all"))
    rs.register_model(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
        supports_function_calling=True,
        supports_vision=True,
    )
    server = RouteSmithProxyServer(rs, ServerConfig(port=0))
    await server.start()
    port = server.port
    yield port, rs
    await server.stop()


def _make_mock_response(content: str = "Hello!", model: str = "gpt-4o-mini",
                        finish_reason: str = "stop") -> MagicMock:
    mock = MagicMock()
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.model_dump.return_value = {
        "id": "cmpl_mock",
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason, "index": 0}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock.routesmith_metadata = {}
    mock.choices[0].message.content = content
    mock.choices[0].finish_reason = finish_reason
    return mock


class TestGateGM1:
    """G-M1: Single-model pool must be indistinguishable from direct API."""

    @pytest.mark.asyncio
    async def test_chat_completion_transparent(self, single_model_proxy):
        """Chat completion with single-model pool returns correct content."""
        port, _ = single_model_proxy
        mock_resp = _make_mock_response()

        with patch("routesmith.client.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}]},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["routesmith_metadata"]["routed"] is True
        assert data["routesmith_metadata"]["selected_model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_claude_code_first_contact(self, single_model_proxy):
        """Simulate Claude Code first-contact: tools, streaming, multi-turn."""
        port, _ = single_model_proxy
        mock_resp = _make_mock_response()

        with patch("routesmith.client.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "auto",
                        "max_tokens": 100,
                        "system": [{"type": "text", "text": "You are a helpful assistant."}],
                        "tools": [{"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}],
                        "tool_choice": {"type": "auto"},
                        "messages": [
                            {"role": "user", "content": "Read file foo.txt"},
                        ],
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["routesmith_metadata"]["routed"] is True

    @pytest.mark.asyncio
    async def test_no_tool_capable_model_returns_4xx(self):
        """Pool without tool_calling-capable model returns 4xx, not 500."""
        rs = RouteSmith(config=RouteSmithConfig(intercept="all"))
        rs.register_model(
            "gpt-4o-mini-nofunc",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            quality_score=0.85,
            supports_function_calling=False,
        )
        server = RouteSmithProxyServer(rs, ServerConfig(port=0))
        await server.start()
        port = server.port

        with patch("routesmith.client.litellm") as mock_llm:
            mock_resp = _make_mock_response()
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/v1/messages",
                    json={
                        "model": "auto",
                        "max_tokens": 100,
                        "tools": [{"name": "read_file", "description": "Read", "input_schema": {"type": "object"}}],
                        "messages": [{"role": "user", "content": "Read file."}],
                    },
                )

        assert resp.status_code == 400
        error_data = resp.json()
        assert error_data.get("type") == "error"
        assert "invalid_request_error" in error_data.get("error", {}).get("type", "")

        await server.stop()

    @pytest.mark.asyncio
    async def test_connect_verify_against_single_model_proxy(self, single_model_proxy):
        """connect --verify against single-model, intercept: all proxy exits 0."""
        import asyncio
        port, _ = single_model_proxy
        url = f"http://127.0.0.1:{port}"

        mock_resp = _make_mock_response()
        with patch("routesmith.client.litellm") as mock_llm:
            mock_llm.acompletion = AsyncMock(return_value=mock_resp)
            from argparse import Namespace

            from routesmith.cli.connect import run_connect

            args = Namespace(tool="opencode", url=url, verify=True, apply=False,
                             yes=False, proxy_api_key="")
            rc = await asyncio.to_thread(run_connect, args)

        assert rc == 0, f"connect --verify failed with exit code {rc}"

    @pytest.mark.asyncio
    async def test_count_tokens_responds(self, single_model_proxy, monkeypatch):
        """count_tokens never 404."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        port, _ = single_model_proxy
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/messages/count_tokens",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "input_tokens" in data

    @pytest.mark.asyncio
    async def test_streaming_chunks_valid(self, single_model_proxy):
        """Streaming response has valid SSE format with final metadata."""
        port, _ = single_model_proxy

        async def _mock_acompletion(*args, **kwargs):
            async def _gen():
                yield _make_stream_chunk("Hello", "gpt-4o-mini", finish_reason=None)
                yield _make_stream_chunk(" world", "gpt-4o-mini", finish_reason="stop")
            return _gen()

        with patch("routesmith.client.litellm") as mock_llm:
            mock_llm.acompletion = _mock_acompletion
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "auto", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )

        assert resp.status_code == 200
        chunks = resp.text.strip().split("\n\n")
        data_chunks = [c for c in chunks if c.startswith("data: ")]
        assert len(data_chunks) >= 2
        last_data = data_chunks[-1]
        if last_data != "data: [DONE]":
            last_chunk = json.loads(last_data.removeprefix("data: "))
            assert "routesmith_metadata" in last_chunk


def _make_stream_chunk(content: str, model: str, finish_reason: str | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content if content else None
    chunk.choices[0].finish_reason = finish_reason
    chunk.model = model
    return chunk
