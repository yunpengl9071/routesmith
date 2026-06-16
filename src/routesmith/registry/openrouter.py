# src/routesmith/registry/openrouter.py
"""Fetch and parse model listings from the OpenRouter API.

OpenRouter exposes a public endpoint at https://openrouter.ai/api/v1/models
that returns all available models with their pricing and context windows.
No API key is required to list models; a key is only needed to call them.
"""
from __future__ import annotations

import json
import math
import ssl
import urllib.request
from dataclasses import dataclass
from typing import Any

_MODELS_URL = "https://openrouter.ai/api/v1/models"


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context with verified CA bundle (uses certifi if available)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


@dataclass
class OpenRouterModel:
    """A model available on OpenRouter with its pricing metadata."""

    id: str
    name: str
    cost_per_1k_input: float   # USD per 1 000 input tokens
    cost_per_1k_output: float  # USD per 1 000 output tokens
    context_window: int
    supports_function_calling: bool
    supports_vision: bool
    quality_score: float       # Heuristic: derived from cost ranking


def fetch_models(timeout: int = 10) -> list[OpenRouterModel]:
    """Fetch all available models from OpenRouter.

    Returns a list sorted by cost ascending (cheapest first).
    Models with zero or missing pricing are excluded.

    Raises:
        urllib.error.URLError: If the request fails.
        ValueError: If the response cannot be parsed.
    """
    req = urllib.request.Request(
        _MODELS_URL,
        headers={"User-Agent": "routesmith/0.1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        raw = json.loads(resp.read().decode())

    models = []
    for entry in raw.get("data", []):
        pricing = entry.get("pricing", {})
        try:
            # OpenRouter pricing is per *token*; convert to per *1k tokens*
            input_per_tok = float(pricing.get("prompt", 0) or 0)
            output_per_tok = float(pricing.get("completion", 0) or 0)
        except (TypeError, ValueError):
            continue

        # Skip free / missing-price models
        if input_per_tok <= 0 or output_per_tok <= 0:
            continue

        ctx = entry.get("context_length") or 0
        arch = entry.get("architecture", {}) or {}
        top = entry.get("top_provider", {}) or {}

        supports_vision = "image" in str(arch.get("input_modalities", []))
        supports_fn = bool(top.get("is_supported_in_playground", True))

        models.append(OpenRouterModel(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            cost_per_1k_input=round(input_per_tok * 1000, 6),
            cost_per_1k_output=round(output_per_tok * 1000, 6),
            context_window=int(ctx),
            supports_function_calling=supports_fn,
            supports_vision=supports_vision,
            quality_score=0.0,   # filled in below after sorting
        ))

    if not models:
        return models

    # Sort cheapest first (by average of input+output cost)
    models.sort(key=lambda m: (m.cost_per_1k_input + m.cost_per_1k_output) / 2)

    # Derive quality_score heuristic: cost rank → [0.60, 1.00]
    # More expensive = higher quality proxy (log scale to compress outliers)
    n = len(models)
    for i, m in enumerate(models):
        rank = i + 1  # 1 = cheapest, n = most expensive
        m.quality_score = round(0.60 + 0.40 * math.log1p(rank) / math.log1p(n), 3)

    return models


def model_to_yaml_block(m: OpenRouterModel, indent: int = 2) -> str:
    """Render a model as a YAML list item for routesmith.yaml."""
    pad = " " * indent
    lines = [
        f"  - id: {m.id}",
        f"{pad}  cost_per_1k_input: {m.cost_per_1k_input}",
        f"{pad}  cost_per_1k_output: {m.cost_per_1k_output}",
        f"{pad}  quality_score: {m.quality_score}",
        f"{pad}  context_window: {m.context_window}",
    ]
    if m.supports_vision:
        lines.append(f"{pad}  supports_vision: true")
    return "\n".join(lines)


def fetch_pricing(model_ids: list[str], timeout: int = 10) -> dict[str, OpenRouterModel]:
    """Fetch pricing for a specific set of model IDs.

    Returns a dict keyed by model ID. Models not found on OpenRouter are omitted.
    """
    all_models = fetch_models(timeout=timeout)
    return {m.id: m for m in all_models if m.id in model_ids}
