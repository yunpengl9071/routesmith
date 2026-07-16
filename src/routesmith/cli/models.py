"""routesmith models — list and refresh the model pool from provider catalogs."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from routesmith.registry.catalog import build_default_pool, detect_providers


def _resolve_config(config_arg: str | None) -> Path | None:
    if config_arg:
        p = Path(config_arg)
        return p if p.exists() else None
    for candidate in [Path("routesmith.yaml"), Path.home() / ".routesmith" / "routesmith.yaml"]:
        if candidate.exists():
            return candidate
    return None


def _provider_from_model_id(model_id: str) -> str:
    if model_id.startswith("openrouter/"):
        return "openrouter"
    if model_id.startswith("groq/"):
        return "groq"
    if model_id.startswith("claude-"):
        return "anthropic"
    if model_id.startswith("gpt-") or model_id.startswith("o"):
        return "openai"
    return "other"


def run_models(args: Namespace) -> int:
    config_path = _resolve_config(getattr(args, "config", None))
    if config_path is None:
        print("No RouteSmith config found.", file=sys.stderr)
        print("Run 'routesmith quickstart' or 'routesmith init' first.", file=sys.stderr)
        return 1

    refresh = getattr(args, "refresh", False)
    output_json = getattr(args, "json", False)

    if refresh:
        providers = getattr(args, "provider", None) or detect_providers()
        if not providers:
            print("No API keys detected. Set one of:", file=sys.stderr)
            print("  ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY", file=sys.stderr)
            return 1

        models = build_default_pool(providers)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        data["catalog"] = {
            "refreshed_at": now,
            "providers": providers,
        }
        data["models"] = models

        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        print(f"Refreshed {len(models)} models from {', '.join(p.capitalize() for p in providers)}")
        print(f"Updated {config_path}")

    else:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        model_entries: list[dict[str, Any]] = data.get("models", [])

        if not model_entries:
            print("No models configured.")
            print("Run 'routesmith models --refresh' to rebuild from provider catalogs.")
            return 0

        if output_json:
            json.dump(model_entries, sys.stdout, indent=2)
            print()
            return 0

        print(f"Models ({len(model_entries)}):")
        print(f"  {'Provider':<14} {'Model ID':<52} {'$/1k in':>10} {'$/1k out':>10} {'Quality':>8}")
        print(f"  {'-'*14} {'-'*52} {'-'*10} {'-'*10} {'-'*8}")
        for m in model_entries:
            provider = _provider_from_model_id(m.get("model_id", ""))
            cost_in = m.get("cost_per_1k_input", 0)
            cost_out = m.get("cost_per_1k_output", 0)
            quality = m.get("quality_score", 0)
            print(f"  {provider:<14} {m.get('model_id', ''):<52} ${cost_in:>8.4f} ${cost_out:>8.4f} {quality:>7.2f}")

    return 0
