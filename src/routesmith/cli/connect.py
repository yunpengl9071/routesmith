"""CLI connect command — generate tool-specific config and verify the proxy."""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# Must be resolved at call time, not import time
def _home() -> Path:
    return Path.home()


def _build_url(base: str) -> str:
    return base.rstrip("/")


def _emit_codex(snippets: list[str], url: str, key: str) -> None:
    snippets.append(f"export OPENAI_BASE_URL={url}/v1")
    snippets.append("")
    snippets.append("# Add to ~/.codex/config.yaml:")
    snippets.append("providers:")
    snippets.append("  routesmith:")
    snippets.append(f"    base_url: {url}/v1")
    snippets.append("    wire_api: chat")
    if key:
        snippets.append(f"    api_key: {key}")
    snippets.append("")
    snippets.append("models:")
    snippets.append("  - id: routesmith/auto")
    snippets.append("    provider: routesmith")


def _emit_opencode(snippets: list[str], url: str, key: str) -> None:
    provider = {
        "providers": {
            "routesmith": {
                "base_url": f"{url}/v1",
            }
        }
    }
    if key:
        provider["providers"]["routesmith"]["api_key"] = key
    snippets.append(json.dumps(provider, indent=2))
    snippets.append("")
    snippets.append("# Add the above under the providers key in your opencode.json")


def _emit_openclaw(snippets: list[str], url: str, key: str) -> None:
    import json
    config = {
        "models": {
            "mode": "merge",
            "providers": {
                "routesmith": {
                    "baseUrl": f"{url}/v1",
                    "apiKey": key or "dummy",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "auto",
                            "name": "RouteSmith Auto Router",
                            "contextWindow": 128000,
                            "maxTokens": 8192,
                        }
                    ],
                }
            },
        },
        "agents": {
            "defaults": {
                "models": {
                    "routesmith/auto": {
                        "alias": "routesmith"
                    }
                }
            }
        },
    }
    snippets.append(json.dumps(config, indent=2))


def _emit_pi(snippets: list[str], url: str, key: str) -> None:
    _emit_openclaw(snippets, url, key)


def _emit_hermes(snippets: list[str], url: str, key: str) -> None:
    snippets.append(f"export OPENAI_BASE_URL={url}/v1")
    snippets.append("# Hermes accepts any OpenAI-compatible base_url in its provider config")


def _emit_openai_sdk(snippets: list[str], url: str, key: str) -> None:
    key_placeholder = key or 'os.environ.get("OPENAI_API_KEY", "routesmith")'
    snippets.append("from openai import OpenAI")
    snippets.append("")
    snippets.append(f'client = OpenAI(base_url="{url}/v1", api_key={key_placeholder})')


def _emit_anthropic_sdk(snippets: list[str], url: str, key: str) -> None:
    snippets.append("from anthropic import Anthropic")
    snippets.append("")
    snippets.append(f'client = Anthropic(base_url="{url}")')
    if key:
        snippets.append(f'# Set ANTHROPIC_API_KEY={key} in your environment')


def _emit_claude_code(snippets: list[str], url: str, key: str) -> None:
    key_val = key or "routesmith"
    snippets.append(f"export ANTHROPIC_BASE_URL={url}")
    snippets.append(f"export ANTHROPIC_API_KEY={key_val}")
    snippets.append("")
    snippets.append("# Note: This works for API-key Anthropic accounts.")
    snippets.append("# Subscription (OAuth) Claude Code sessions cannot be rerouted.")
    snippets.append("# For those, use: routesmith run claude")
    snippets.append("")
    snippets.append("# Alternatively, add to ~/.claude/settings.json under the 'env' block:")
    snippets.append(f'#   "env": {{"ANTHROPIC_BASE_URL": "{url}", "ANTHROPIC_API_KEY": "{key_val}"}}')
    snippets.append("")
    snippets.append("---")
    snippets.append("or just use: routesmith run claude")


def _emit_openclaw_like(snippets: list[str], url: str, key: str, label: str) -> None:
    if label == "openclaw":
        snippets.append("# OpenClaw provider config — also available via:")
        snippets.append(f"#   routesmith openclaw-config --host {url}")
    else:
        snippets.append(f"# {label} (OpenClaw-compatible) provider config:")
    snippets.append("")
    _emit_openclaw(snippets, url, key)


EMITTERS = {
    "claude-code": _emit_claude_code,
    "codex": _emit_codex,
    "opencode": _emit_opencode,
    "openclaw": lambda sn, u, k: _emit_openclaw_like(sn, u, k, "openclaw"),
    "pi": lambda sn, u, k: _emit_openclaw_like(sn, u, k, "pi"),
    "hermes": _emit_hermes,
    "openai-sdk": _emit_openai_sdk,
    "anthropic-sdk": _emit_anthropic_sdk,
}

_TOOL_NAMES = sorted(EMITTERS.keys())

_VERIFY_ENDPOINTS = {
    "claude-code": "/v1/messages",
    "anthropic-sdk": "/v1/messages",
}


def run_connect(args: Namespace) -> int:
    tool = args.tool
    url = _build_url(args.url)

    if tool not in EMITTERS:
        print(f"Unknown tool: {tool}", file=sys.stderr)
        print(f"Valid tools: {', '.join(_TOOL_NAMES)}", file=sys.stderr)
        return 2

    if args.verify:
        return _run_verify(tool, url, args)

    if args.apply:
        return _run_apply(tool, url, args)

    return _run_print(tool, url, args)


def _resolve_key(args: Namespace) -> str:
    key = args.proxy_api_key or os.environ.get("ROUTESMITH_API_KEY", "")
    return key


def _run_print(tool: str, url: str, args: Namespace) -> int:
    key = _resolve_key(args)
    snippets: list[str] = []
    emitter = EMITTERS[tool]
    emitter(snippets, url, key)

    header = f"# RouteSmith connect — {tool}"
    if args.proxy_api_key:
        header += " (--proxy-api-key set)"
    print(header)
    print()
    for line in snippets:
        print(line)
    return 0


def _run_apply(tool: str, url: str, args: Namespace) -> int:
    if tool == "codex":
        config_dir = _home() / ".codex"
        config_path = config_dir / "config.yaml"
        if config_path.exists() and not args.yes:
            print(f"File exists: {config_path}", file=sys.stderr)
            print("Use --yes to overwrite", file=sys.stderr)
            return 1
        key = _resolve_key(args)
        config_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "providers:",
            "  routesmith:",
            f"    base_url: {url}/v1",
            "    wire_api: chat",
            "",
            "models:",
            "  - id: routesmith/auto",
            "    provider: routesmith",
        ]
        if key:
            lines.insert(3, f"    api_key: {key}")
        config_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote Codex config to {config_path}")
        return 0

    if tool in ("openclaw", "pi"):
        config_path = Path("routesmith-provider.json")
        if config_path.exists() and not args.yes:
            print(f"File exists: {config_path}", file=sys.stderr)
            print("Use --yes to overwrite", file=sys.stderr)
            return 1
        key = _resolve_key(args)
        sn: list[str] = []
        _emit_openclaw(sn, url, key)
        config_path.write_text("\n".join(sn) + "\n")
        print(f"Wrote provider config to {config_path}")
        return 0

    # For all other tools, print snippet with note
    print(f"--apply is not supported for {tool}; manual placement is required.")
    print()
    _run_print(tool, url, args)
    return 0


def _run_verify(tool: str, url: str, args: Namespace) -> int:
    key = _resolve_key(args)

    # Step 0: check /health is reachable
    try:
        health_body = _http_get(f"{url}/health", key)
    except (URLError, ConnectionError, OSError) as exc:
        print(f"Could not connect to RouteSmith proxy at {url}", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        print(file=sys.stderr)
        print("Start the proxy with:", file=sys.stderr)
        print("  routesmith serve --daemon", file=sys.stderr)
        print("Or:  routesmith run", file=sys.stderr)
        return 1

    # Step 1: verify health response
    if health_body.get("status") != "healthy":
        print(f"Health check failed: {health_body}", file=sys.stderr)
        return 1

    # Step 2: send a minimal completion with a concrete model name
    endpoint = _VERIFY_ENDPOINTS.get(tool, "/v1/chat/completions")
    is_anthropic = endpoint == "/v1/messages"

    try:
        resp = _send_verify_request(url, key, endpoint, is_anthropic)
    except HTTPError as exc:
        print(f"Proxy reached, but the verification request failed ({exc.code}).", file=sys.stderr)
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            message = detail.get("error", {}).get("message", json.dumps(detail))
        except Exception:
            message = exc.reason
        print(f"Error: {message}", file=sys.stderr)
        print(file=sys.stderr)
        print("This is usually a missing/invalid provider API key for the", file=sys.stderr)
        print("model RouteSmith selected — check the proxy's environment and", file=sys.stderr)
        print("`routesmith audit` for the routing decision.", file=sys.stderr)
        return 1
    except (URLError, ConnectionError, OSError) as exc:
        print(f"Could not reach RouteSmith proxy at {url}: {exc}", file=sys.stderr)
        return 1

    # Step 3: assert routesmith_metadata.routed == true
    metadata = resp.get("routesmith_metadata", {})
    if not metadata.get("routed"):
        print("Request was passed through without routing.", file=sys.stderr)
        print("Set routing.intercept: all in routesmith.yaml,", file=sys.stderr)
        print("then restart routesmith serve.", file=sys.stderr)
        return 1

    # Step 4: print success
    model = metadata.get("selected_model", "unknown")
    cost = metadata.get("estimated_cost_usd", 0.0)
    print(f"Verified: proxy at {url} is routing requests.")
    print(f"Routed to: {model}")
    print(f"Estimated cost: ${float(cost):.6f}")
    return 0


def _http_get(url: str, api_key: str = "") -> dict[str, Any]:
    req = Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        result: dict[str, Any] = json.loads(body)
        return result


def _http_post(url: str, data: dict, api_key: str = "") -> dict[str, Any]:
    body_bytes = json.dumps(data).encode("utf-8")
    req = Request(url, data=body_bytes, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        result: dict[str, Any] = json.loads(body)
        return result


def _send_verify_request(url: str, api_key: str, endpoint: str, is_anthropic: bool) -> dict:
    if is_anthropic:
        body = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 50,
        }
        resp = _http_post(f"{url}{endpoint}", body, api_key)
        return resp

    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50,
    }
    resp = _http_post(f"{url}{endpoint}", body, api_key)
    return resp
