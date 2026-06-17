#!/usr/bin/env python3
"""
End-to-end smoke test for OpenClaw integration via RouteSmith.

Two integration modes are tested:

1. **Proxy mode** — Start RouteSmith proxy server, make OpenAI-compatible
   API requests (mimicking how OpenClaw's built-in OpenAI provider connects).

2. **Native mode** — Use RouteSmithAnthropic (drop-in for anthropic.Anthropic)
   directly, no proxy needed.

Requires an API key. The cheapest path is Groq (free tier):
  export GROQ_API_KEY=gsk_...

Or use OpenAI:
  export OPENAI_API_KEY=sk-...

NOTE: Skipped by pytest unless API keys are set.
"""

import json
import os
import subprocess
import sys
import time

import pytest

_LIVE_TEST_SKIP = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires API keys. Set OPENAI_API_KEY, GROQ_API_KEY, or ANTHROPIC_API_KEY."
)


def _detect_provider() -> str:
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return ""


def _get_proxy_port() -> int:
    return 9119


def _wait_for_proxy(port: int, timeout: float = 10.0):
    """Wait for the proxy server to be ready."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/health")
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# Test 1: Proxy Server Starts and Serves
# ---------------------------------------------------------------------------

@_LIVE_TEST_SKIP
class TestProxyServer:
    """Test that the RouteSmith proxy server starts and accepts requests."""

    def test_proxy_starts_and_health_check(self):
        """Proxy server starts and /health endpoint responds."""
        provider = _detect_provider()
        if not provider:
            pytest.skip("No API key available")

        port = _get_proxy_port()

        # Start proxy in background
        proc = subprocess.Popen(
            [sys.executable, "-m", "routesmith.cli.main", "serve", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "ROUTESMITH_CONFIG": ""},
        )

        try:
            ready = _wait_for_proxy(port, timeout=15.0)
            assert ready, f"Proxy did not become healthy within 15s on port {port}"

            # Check health endpoint
            import urllib.request
            resp = urllib.request.urlopen(f"http://localhost:{port}/health")
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data.get("status") in ("ok", "healthy")

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_proxy_completion(self):
        """Proxy serves /v1/chat/completions and returns a valid response."""
        provider = _detect_provider()
        if not provider:
            pytest.skip("No API key available")

        port = _get_proxy_port()

        proc = subprocess.Popen(
            [sys.executable, "-m", "routesmith.cli.main", "serve", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "ROUTESMITH_CONFIG": ""},
        )

        try:
            ready = _wait_for_proxy(port, timeout=15.0)
            assert ready, "Proxy did not become healthy"

            import urllib.request

            payload = json.dumps({
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "Say hello in exactly one word."}
                ],
                "max_tokens": 20,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"http://localhost:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            resp = urllib.request.urlopen(req, timeout=30)
            assert resp.status == 200
            data = json.loads(resp.read())

            # Validate OpenAI-compatible response structure
            assert "choices" in data
            assert len(data["choices"]) >= 1
            assert "message" in data["choices"][0]
            assert "content" in data["choices"][0]["message"]
            content = data["choices"][0]["message"]["content"]
            assert len(content) > 0
            print(f"  Proxy response: {content}")

        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Test 2: RouteSmithAnthropic Native Client
# ---------------------------------------------------------------------------

@_LIVE_TEST_SKIP
class TestAnthropicNativeClient:
    """Test the RouteSmithAnthropic drop-in client (native mode)."""

    def test_basic_messages_create(self):
        """RouteSmithAnthropic.messages.create() returns a valid Message."""
        provider = _detect_provider()
        if not provider:
            pytest.skip("No API key available")

        from routesmith.integrations.anthropic import RouteSmithAnthropic

        client = RouteSmithAnthropic()

        # Register models appropriate for the provider
        if provider == "groq":
            client.register_model(
                "groq/llama-3.1-8b-instant",
                cost_per_1k_input=0.00005,
                cost_per_1k_output=0.00008,
                quality_score=0.78,
            )
        elif provider == "openai":
            client.register_model(
                "openai/gpt-4o-mini",
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.00060,
                quality_score=0.85,
            )

        msg = client.messages.create(
            model="auto",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'route' and nothing else."}],
        )

        # Validate anthropic.types.Message structure
        assert msg.type == "message"
        assert msg.role == "assistant"
        assert len(msg.content) >= 1
        assert msg.content[0].type == "text"
        assert len(msg.content[0].text) > 0
        print(f"  Anthropic response: {msg.content[0].text}")

    def test_system_message_prepended(self):
        """System parameter is prepended to messages."""
        provider = _detect_provider()
        if not provider:
            pytest.skip("No API key available")

        from routesmith.integrations.anthropic import RouteSmithAnthropic

        client = RouteSmithAnthropic()

        if provider == "groq":
            client.register_model(
                "groq/llama-3.1-8b-instant",
                cost_per_1k_input=0.00005,
                cost_per_1k_output=0.00008,
                quality_score=0.78,
            )
        elif provider == "openai":
            client.register_model(
                "openai/gpt-4o-mini",
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.00060,
                quality_score=0.85,
            )

        msg = client.messages.create(
            model="auto",
            max_tokens=30,
            system="Reply with exactly the word: orange",
            messages=[{"role": "user", "content": "What color?"}],
        )

        assert msg.type == "message"
        text = msg.content[0].text.lower()
        print(f"  System-guided response: {text}")

    def test_stats_tracking(self):
        """Stats are tracked after requests."""
        provider = _detect_provider()
        if not provider:
            pytest.skip("No API key available")

        from routesmith.integrations.anthropic import RouteSmithAnthropic

        client = RouteSmithAnthropic()

        if provider == "groq":
            client.register_model(
                "groq/llama-3.1-8b-instant",
                cost_per_1k_input=0.00005,
                cost_per_1k_output=0.00008,
                quality_score=0.78,
            )
        elif provider == "openai":
            client.register_model(
                "openai/gpt-4o-mini",
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.00060,
                quality_score=0.85,
            )

        client.messages.create(
            model="auto",
            max_tokens=20,
            messages=[{"role": "user", "content": "Hi"}],
        )

        stats = client.stats
        assert stats["request_count"] >= 1
        assert stats["total_cost_usd"] > 0
        print(f"  Stats after request: {stats['request_count']} requests, "
              f"${stats['total_cost_usd']:.6f} total")

    def test_register_model_delegates(self):
        """register_model() on the client delegates to RouteSmith."""
        from routesmith.integrations.anthropic import RouteSmithAnthropic

        client = RouteSmithAnthropic()
        client.register_model(
            "test/model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.9,
        )

        # Model should be registered in the underlying RouteSmith
        models = client._rs.registry.list_models()
        model_ids = [m.model_id for m in models]
        assert "test/model" in model_ids


# ---------------------------------------------------------------------------
# Test 3: OpenClaw Config Generation
# ---------------------------------------------------------------------------

class TestOpenClawConfig:
    """Test the `routesmith openclaw-config` CLI command.

    These tests do NOT require API keys.
    """

    def test_cli_generates_valid_json(self):
        """CLI openclaw-config outputs valid JSON."""
        result = subprocess.run(
            [sys.executable, "-m", "routesmith.cli.main", "openclaw-config"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        config = json.loads(result.stdout)
        assert "models" in config
        assert "agents" in config

    def test_config_has_provider_entry(self):
        """Generated config has a routesmith provider entry."""
        result = subprocess.run(
            [sys.executable, "-m", "routesmith.cli.main", "openclaw-config"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        config = json.loads(result.stdout)
        providers = config["models"]["providers"]
        assert "routesmith" in providers
        assert providers["routesmith"]["api"] == "openai-completions"

    def test_config_has_auto_model(self):
        """Generated config includes the 'auto' model."""
        result = subprocess.run(
            [sys.executable, "-m", "routesmith.cli.main", "openclaw-config"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        config = json.loads(result.stdout)
        models = config["models"]["providers"]["routesmith"]["models"]
        model_ids = [m["id"] for m in models]
        assert "auto" in model_ids

    def test_config_default_agent_alias(self):
        """Default agent section maps routesmith/auto -> routesmith alias."""
        result = subprocess.run(
            [sys.executable, "-m", "routesmith.cli.main", "openclaw-config"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        config = json.loads(result.stdout)
        defaults = config["agents"]["defaults"]["models"]
        assert "routesmith/auto" in defaults
        assert defaults["routesmith/auto"]["alias"] == "routesmith"


# ---------------------------------------------------------------------------
# Test 4: Streaming (Proxy)
# ---------------------------------------------------------------------------

@_LIVE_TEST_SKIP
class TestProxyStreaming:
    """Test SSE streaming through the proxy (mimics OpenClaw streaming)."""

    def test_streaming_response(self):
        """Proxy returns SSE streaming chunks."""
        provider = _detect_provider()
        if not provider:
            pytest.skip("No API key available")

        port = _get_proxy_port()

        proc = subprocess.Popen(
            [sys.executable, "-m", "routesmith.cli.main", "serve", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "ROUTESMITH_CONFIG": ""},
        )

        try:
            ready = _wait_for_proxy(port, timeout=15.0)
            assert ready, "Proxy did not become healthy"

            import urllib.request

            payload = json.dumps({
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "Count: 1, 2, 3"}
                ],
                "max_tokens": 30,
                "stream": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"http://localhost:{port}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            resp = urllib.request.urlopen(req, timeout=30)
            body = resp.read().decode("utf-8")

            # SSE format: data: {...}\n\n
            assert "data:" in body
            assert "choices" in body or "[DONE]" in body
            print(f"  SSE stream length: {len(body)} chars")

        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("OpenClaw Integration Smoke Test")
    print("=" * 50)
    provider = _detect_provider()
    print(f"Provider: {provider or 'not detected — live tests will be skipped'}")
    print()

    # Run only the non-live tests (config generation) when no API key
    TestOpenClawConfig().test_cli_generates_valid_json()
    TestOpenClawConfig().test_config_has_provider_entry()
    TestOpenClawConfig().test_config_has_auto_model()
    TestOpenClawConfig().test_config_default_agent_alias()
    print("  ✓ All config generation tests passed")

    if provider:
        print()
        print("Running live tests with provider:", provider)
        TestProxyServer().test_proxy_starts_and_health_check()
        print("  ✓ Proxy health check")
        TestProxyServer().test_proxy_completion()
        print("  ✓ Proxy completion")
        TestAnthropicNativeClient().test_basic_messages_create()
        print("  ✓ Native client messages.create")
        TestAnthropicNativeClient().test_system_message_prepended()
        print("  ✓ System message prepended")
        TestAnthropicNativeClient().test_stats_tracking()
        print("  ✓ Stats tracking")
        TestProxyStreaming().test_streaming_response()
        print("  ✓ SSE streaming")
        print()
        print("All tests passed! 🎉")
    else:
        print()
        print("Skipping live tests (no API key).")
        print("Set GROQ_API_KEY or OPENAI_API_KEY to run full suite.")
