"""routesmith quickstart — single-command setup and launch.

Detects provider API keys, generates routesmith.yaml (non-interactively),
starts the proxy server, and prints curl/Python/Anthropic snippets.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from routesmith.registry.catalog import build_default_pool, detect_providers


def _generate_config(
    output: str,
    providers: list[str],
    yes: bool,
) -> int:
    """Generate routesmith.yaml."""
    out_path = Path(output)
    if out_path.exists() and not yes:
        print(f"'{output}' exists. Use --yes to overwrite.")
        return 1

    models = build_default_pool(providers)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    config: dict[str, Any] = {
        "catalog": {
            "refreshed_at": now,
            "providers": providers,
        },
        "routing": {
            "intercept": "all",
            "sticky": "auto",
        },
        "predictor_type": "lints",
        "budget": {},
        "models": models,
    }

    yaml_text = yaml.dump(config, default_flow_style=False, sort_keys=False)
    out_path = Path(output)
    out_path.write_text(yaml_text)
    return 0


def run_quickstart(args: Namespace) -> int:
    """Run routesmith quickstart."""
    port = args.port or 9119
    yes = getattr(args, "yes", False)
    config_path = "routesmith.yaml"

    providers = getattr(args, "provider", None) or detect_providers()
    if not providers:
        print("No API key found. Set one of:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export OPENAI_API_KEY=sk-...")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        print("  export GROQ_API_KEY=gsk_...")
        return 1

    provider_names = ", ".join(p.capitalize() for p in providers)
    print(f"Detected provider(s): {provider_names}")

    exit_code = _generate_config(config_path, providers, yes)
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

    if not providers:
        return 0

    mock_env = os.environ.get("RS_MOCK_LITELLM")
    if mock_env:
        print(f"RS_MOCK_LITELLM is set — starting server on port {port}")
        sys.stdout.flush()
        from routesmith import RouteSmith

        models = build_default_pool(providers)
        rs = RouteSmith()
        for m in models:
            rs.register_model(
                m["model_id"],
                cost_per_1k_input=m.get("cost_per_1k_input", 0),
                cost_per_1k_output=m.get("cost_per_1k_output", 0),
                quality_score=m.get("quality_score", 0.5),
            )

        import asyncio

        from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

        server = RouteSmithProxyServer(rs, ServerConfig(host="127.0.0.1", port=port))
        asyncio.run(server.serve_forever())

    return 0
