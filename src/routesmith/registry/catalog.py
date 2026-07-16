"""Provider-aware model catalog — detect providers, load catalogs, build pools."""
from __future__ import annotations

import json
import os
from importlib import resources

from routesmith.exceptions import NoProviderDetectedError
from routesmith.registry.priors import load_default_priors, lookup_prior

_PROVIDER_ORDER = ["anthropic", "openai", "openrouter", "groq"]
_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
}


def detect_providers() -> list[str]:
    providers = []
    for p in _PROVIDER_ORDER:
        if os.environ.get(_ENV_VARS[p]):
            providers.append(p)
    return providers


def load_catalog(provider: str) -> dict:
    ref = resources.files("routesmith.registry.data").joinpath(f"catalogs/{provider}.json")
    return dict(json.loads(ref.read_text()))


def build_default_pool(providers: list[str] | None = None) -> list[dict]:
    if providers is None:
        providers = detect_providers()
    if not providers:
        raise NoProviderDetectedError(
            "No API keys detected. Set one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY"
        )

    priors = load_default_priors()
    seen: dict[str, dict] = {}

    for provider in providers:
        catalog = load_catalog(provider)
        for entry in catalog.get("models", []):
            if not entry.get("default", False):
                continue
            model_id = entry["model_id"]
            canonical = model_id.split("/")[-1]
            is_direct = not model_id.startswith("openrouter/") and not model_id.startswith("groq/")
            existing = seen.get(canonical)
            if existing:
                existing_is_direct = not existing["model_id"].startswith("openrouter/") and not existing["model_id"].startswith("groq/")
                if is_direct and not existing_is_direct:
                    seen[canonical] = dict(entry)
                elif not is_direct and existing_is_direct:
                    pass
                elif not is_direct and not existing_is_direct:
                    seen[canonical] = dict(entry)
            else:
                seen[canonical] = dict(entry)

    result = list(seen.values())
    for entry in result:
        mid = entry["model_id"]
        if mid not in priors:
            prior = lookup_prior(mid, priors)
            if prior is not None:
                entry["quality_score"] = prior

    return result


def pool_entry_to_yaml_model(entry: dict) -> dict:
    """Translate a catalog/pool entry (registry.catalog schema) to the
    routesmith.yaml on-disk model schema (yaml_loader._parse_model_entry).

    The two schemas disagree on two keys — "model_id" vs "id" and
    "supports_tools" vs "supports_function_calling" — so writing a pool
    entry straight into models: without this translation either crashes
    config load (KeyError: 'id') or silently drops tool-calling capability
    (every model registers with supports_function_calling=False, so no
    model ever qualifies for a request that includes `tools`).
    """
    out = {
        "id": entry["model_id"],
        "cost_per_1k_input": entry.get("cost_per_1k_input", 0.0),
        "cost_per_1k_output": entry.get("cost_per_1k_output", 0.0),
        "quality_score": entry.get("quality_score", 0.8),
    }
    if "context_window" in entry:
        out["context_window"] = entry["context_window"]
    if "latency_p50_ms" in entry:
        out["latency_p50_ms"] = entry["latency_p50_ms"]
    if "supports_tools" in entry:
        out["supports_function_calling"] = entry["supports_tools"]
    if "supports_vision" in entry:
        out["supports_vision"] = entry["supports_vision"]
    return out
