"""routesmith quickstart — single-command setup and launch.

Detects provider API keys, generates routesmith.yaml (non-interactively),
starts the proxy server, and prints curl/Python/Anthropic snippets.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml


def _detect_provider() -> tuple[str, str] | None:
    """Check env for API keys. Returns (provider_name, key) or None."""
    for provider, var in [
        ("OpenRouter", "OPENROUTER_API_KEY"),
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
    ]:
        key = os.environ.get(var)
        if key:
            return provider, key
    return None


_DEFAULT_MODELS = [
    {
        "model_id": "openai/gpt-4o-mini",
        "cost_per_1k_input": 0.15,
        "cost_per_1k_output": 0.60,
        "quality_score": 0.85,
    },
    {
        "model_id": "openai/gpt-4o",
        "cost_per_1k_input": 2.50,
        "cost_per_1k_output": 10.00,
        "quality_score": 0.95,
    },
]


def _generate_config(output: str, provider: str | None, yes: bool) -> int:
    """Generate routesmith.yaml."""
    out_path = Path(output)
    if out_path.exists() and not yes:
        print(f"'{output}' exists. Use --yes to overwrite.")
        return 1

    models = _DEFAULT_MODELS
    predictor = "lints"

    config: dict[str, Any] = {
        "predictor_type": predictor,
        "budget": {},
    }

    if provider == "OpenRouter":
        config["openrouter_models"] = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
        ]
        models = []
    else:
        config["models"] = models

    yaml_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    out_path.write_text(yaml_text)
    return 0


def run_quickstart(args: Namespace) -> int:
    """Run routesmith quickstart."""
    port = args.port or 9119
    yes = getattr(args, "yes", False)
    config_path = "routesmith.yaml"

    provider_info = _detect_provider()
    if provider_info is None:
        print("No API key found. Set one of:")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export ANTHROPIC_API_KEY=sk-...")
        return 1

    provider_name, _ = provider_info
    print(f"Detected provider: {provider_name}")

    exit_code = _generate_config(config_path, provider_name, yes)
    if exit_code != 0:
        return exit_code
    print(f"Generated {config_path}")

    print()
    print("To start the proxy and try it out:")
    print(f"  routesmith serve --port {port}")
    print()
    print("Try a completion:")
    print(f'  curl http://localhost:{port}/v1/chat/completions \\')
    print('    -d \'{"model":"auto","messages":[{"role":"user","content":"hi"}]}\' \\')
    print("    -H 'Content-Type: application/json'")
    print()
    print("Python snippet:")
    print("  from openai import OpenAI")
    print(f'  client = OpenAI(base_url="http://localhost:{port}/v1", api_key="dummy")')
    print('  resp = client.chat.completions.create(model="auto", messages=[{"role":"user","content":"hi"}])')
    print()
    print("Anthropic SDK (POST /v1/messages):")
    print(f"  export ANTHROPIC_BASE_URL=http://localhost:{port}")
    print("  export ANTHROPIC_API_KEY=dummy")
    print()

    if not provider_info:
        return 0

    # Start server if RS_MOCK_LITELLM is set (for CI integration test)
    mock_env = os.environ.get("RS_MOCK_LITELLM")
    if mock_env:
        print(f"RS_MOCK_LITELLM is set — starting server on port {port}")
        sys.stdout.flush()
        from routesmith import RouteSmith

        rs = RouteSmith()
        rs.register_model(
            "openai/gpt-4o-mini",
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            quality_score=0.85,
        )
        rs.register_model(
            "openai/gpt-4o",
            cost_per_1k_input=2.50,
            cost_per_1k_output=10.00,
            quality_score=0.95,
        )

        import asyncio

        from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

        server = RouteSmithProxyServer(rs, ServerConfig(host="127.0.0.1", port=port))
        asyncio.run(server.serve_forever())

    return 0
