"""Real API integration test for the proxy server.

Starts the proxy in-process, sends a real HTTP request through it,
and verifies routed metadata. Requires API keys.

Run: OPENROUTER_API_KEY=sk-or-... pytest tests/test_proxy_real_api.py -x -v
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

pytestmark = pytest.mark.requires_api


def _pick_model():
    """Pick the cheapest available model based on env keys."""
    import litellm
    _ = litellm  # ensure litellm is importable

    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(("gpt-4o-mini", "openai"))
    if os.environ.get("OPENROUTER_API_KEY"):
        providers.append(("openrouter/openai/gpt-4o-mini", "openrouter"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(("claude-3-5-haiku-latest", "anthropic"))
    if os.environ.get("GROQ_API_KEY"):
        providers.append(("groq/llama-3.1-8b-instant", "groq"))

    if not providers:
        pytest.skip("No API keys found. Set OPENAI_API_KEY, OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY.")

    return providers[0]


@pytest.mark.asyncio
async def test_proxy_routes_real_request():
    """Start proxy, send real HTTP request with a concrete model, verify routed metadata."""
    from routesmith import RouteSmith
    from routesmith.config import RouteSmithConfig
    from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

    model_id, provider = _pick_model()
    print(f"\n  Using model: {model_id} (via {provider})")

    rs = RouteSmith(config=RouteSmithConfig(intercept="all"))
    rs.register_model(
        model_id,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        quality_score=0.85,
    )

    server = RouteSmithProxyServer(rs, ServerConfig(port=0))
    await server.start()
    port = server.port

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                    "max_tokens": 20,
                },
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()

        # Verify proxy wired routing correctly
        meta = data.get("routesmith_metadata", {})
        assert meta.get("routed") is True, (
            f"Request was NOT routed through RouteSmith! "
            f"metadata={json.dumps(meta, indent=2)}"
        )
        assert meta.get("requested_model") == model_id
        assert meta.get("passthrough_reason") is None

        # Verify response is a real LLM output
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        assert len(content) > 0, "Empty response content from real API call"
        print(f"  Response: {content[:100]}...")
        print(f"  Selected model: {meta.get('selected_model')}")
        print("  Real API proxy test PASSED")

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_anthropic_messages_via_proxy():
    """Proxy /v1/messages with ANTHROPIC_API_KEY, real Claude model, full P4 translation path.

    Tests: P3 intercept routing, P4 bidirectional translation, real API call.
    Uses model='auto' to avoid the protocol-native fast path (R4.5) and exercise
    the full anthropic->openai->anthropic translation round-trip.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    from routesmith import RouteSmith
    from routesmith.config import RouteSmithConfig
    from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

    model_id = "claude-3-5-haiku-latest"
    print(f"\n  Using Anthropic model: {model_id}")

    rs = RouteSmith(config=RouteSmithConfig(intercept="all"))
    rs.register_model(
        model_id,
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
        quality_score=0.85,
        supports_function_calling=True,
        supports_vision=True,
    )

    server = RouteSmithProxyServer(rs, ServerConfig(port=0))
    await server.start()
    port = server.port

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/v1/messages",
                json={
                    "model": "auto",
                    "max_tokens": 50,
                    "messages": [{"role": "user", "content": "Reply with exactly: hello world"}],
                },
            )

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text[:200]}"
        data = resp.json()

        # Verify Anthropic-format response
        assert data.get("type") == "message", f"Not an Anthropic response: {json.dumps(data, indent=2)[:200]}"
        assert data.get("role") == "assistant"
        assert len(data.get("content", [])) > 0
        assert data["content"][0].get("type") == "text"
        assert len(data["content"][0].get("text", "")) > 0

        # Verify proxy routing metadata
        meta = data.get("routesmith_metadata", {})
        assert meta.get("routed") is True, (
            f"Request was NOT routed through RouteSmith! "
            f"metadata={json.dumps(meta, indent=2)}"
        )
        assert meta.get("passthrough_reason") is None

        print(f"  Response: {data['content'][0]['text'][:100]}...")
        print(f"  Selected model: {meta.get('selected_model')}")
        print("  Anthropic /v1/messages proxy test PASSED")

    finally:
        await server.stop()
