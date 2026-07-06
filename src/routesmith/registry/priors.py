"""Benchmark-grounded quality priors for cold-start routing."""
from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_default_priors() -> dict[str, float]:
    """Load the packaged prior table. Returns {} if the data file is missing."""
    try:
        ref = resources.files("routesmith.registry.data").joinpath("default_priors.json")
        data: dict[str, Any] = json.loads(ref.read_text())
        return dict(data["priors"])
    except Exception:
        return {}


def lookup_prior(model_id: str, priors: dict[str, float]) -> float | None:
    """Match order: exact id -> id without provider prefix -> unique substring.

    'openrouter/openai/gpt-4o' matches key 'openai/gpt-4o' via suffix.
    Returns None when no unambiguous match exists.
    """
    if model_id in priors:
        return priors[model_id]
    for key, score in priors.items():
        if model_id.endswith("/" + key) or key.endswith("/" + model_id):
            return score
    tail = model_id.rsplit("/", 1)[-1]
    hits = [s for k, s in priors.items() if k.rsplit("/", 1)[-1] == tail]
    return hits[0] if len(hits) == 1 else None
