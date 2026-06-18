"""Auto-discover models from OpenRouter or curated fallback list."""

from typing import Any


def discover_models(
    api_key: str | None = None,
    providers: list[str] | None = None,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    """Discover model metadata from OpenRouter API or fallback curated list."""
    if api_key:
        try:
            import requests
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            return _parse_openrouter_response(resp.json(), providers)
        except Exception:
            pass  # fall through to curated

    return _curated_models(providers)


def _parse_openrouter_response(
    data: dict, providers: list[str] | None
) -> list[dict[str, Any]]:
    """Parse OpenRouter models endpoint response."""
    models = []
    for m in data.get("data", []):
        model_id = m["id"]
        pricing = m.get("pricing", {})
        cost_input = float(pricing.get("prompt", 0)) * 1000
        cost_output = float(pricing.get("completion", 0)) * 1000

        if providers and not any(
            model_id.startswith(f"{p}/") for p in providers
        ):
            continue

        # Seed quality from benchmarks if available
        benchmarks = m.get("benchmarks", {})
        quality = 0.80
        if benchmarks:
            scores = [v for v in benchmarks.values() if isinstance(v, (int, float))]
            if scores:
                quality = round(sum(scores) / len(scores), 2)

        models.append({
            "model_id": model_id,
            "cost_per_1k_input": cost_input,
            "cost_per_1k_output": cost_output,
            "quality_score": quality,
            "context_window": m.get("context_length", 128000),
            "supports_vision": "image" in m.get("architecture", {}).get(
                "modality", ""
            ),
        })

    return models


def _curated_models(
    providers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fallback curated list of well-known models."""
    curated = [
        ("openai/gpt-4o", 0.005, 0.015, 0.95, 128000, True),
        ("openai/gpt-4o-mini", 0.00015, 0.0006, 0.85, 128000, True),
        ("openai/gpt-5.1", 0.008, 0.024, 0.97, 256000, True),
        ("anthropic/claude-sonnet-4.5", 0.003, 0.015, 0.93, 200000, True),
        ("anthropic/claude-haiku-4.5", 0.0008, 0.004, 0.83, 200000, True),
        ("google/gemini-2.5-flash", 0.0, 0.0, 0.82, 1048576, True),
        ("google/gemini-3.1-pro-preview", 0.00125, 0.005, 0.94, 1048576, True),
        ("deepseek/deepseek-v3.2", 0.00027, 0.0011, 0.92, 131072, False),
        ("meta-llama/llama-3.3-70b-instruct", 0.00035, 0.0004, 0.86, 131072, False),
        ("qwen/qwen3-max", 0.002, 0.008, 0.90, 131072, False),
        ("mistralai/mistral-large-2512", 0.002, 0.006, 0.88, 131072, True),
    ]

    models = []
    for (
        model_id, cost_in, cost_out, quality, ctx, vision
    ) in curated:
        if providers and not any(
            model_id.startswith(f"{p}/") for p in providers
        ):
            continue
        models.append({
            "model_id": model_id,
            "cost_per_1k_input": cost_in,
            "cost_per_1k_output": cost_out,
            "quality_score": quality,
            "context_window": ctx,
            "supports_vision": vision,
        })

    return models