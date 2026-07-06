"""Tests for OpenRouter registry integration."""

from __future__ import annotations

import logging

from routesmith.registry.openrouter import fetch_models


def _entry(
    model_id: str,
    input_price: str = "0.000001",
    output_price: str = "0.000002",
    context_length: int = 128000,
    supported_parameters: list[str] | None = None,
    modalities: list[str] | None = None,
) -> dict:
    entry: dict = {
        "id": model_id,
        "name": model_id.replace("/", " ").title(),
        "pricing": {"prompt": input_price, "completion": output_price},
        "context_length": context_length,
        "architecture": {"input_modalities": modalities or []},
        "top_provider": {},
    }
    if supported_parameters is not None:
        entry["supported_parameters"] = supported_parameters
    return entry


class TestToolsSupport:
    def test_tools_in_supported_parameters(self):
        """supported_parameters contains "tools" -> supports_function_calling is True."""
        data = {"data": [_entry("model-a", supported_parameters=["tools", "temperature"])]}
        models = fetch_models(_response_json=data)
        assert len(models) == 1
        assert models[0].supports_function_calling is True

    def test_tools_not_in_supported_parameters(self):
        """supported_parameters without "tools" -> supports_function_calling is False."""
        data = {"data": [_entry("model-b", supported_parameters=["temperature"])]}
        models = fetch_models(_response_json=data)
        assert len(models) == 1
        assert models[0].supports_function_calling is False

    def test_supported_parameters_absent(self):
        """supported_parameters absent -> supports_function_calling defaults to True."""
        data = {"data": [_entry("model-c")]}
        models = fetch_models(_response_json=data)
        assert len(models) == 1
        assert models[0].supports_function_calling is True


class TestFreeModels:
    def test_free_models_skipped_and_counted(self, caplog):
        caplog.set_level(logging.INFO)
        entries = []
        # 3 paid models
        for i in range(3):
            entries.append(_entry(f"paid-{i}"))
        # 2 free models
        for i in range(2):
            entries.append(_entry(f"free-{i}", input_price="0", output_price="0"))
        data = {"data": entries}
        models = fetch_models(_response_json=data)
        assert len(models) == 3
        assert "Skipped 2" in caplog.text

    def test_no_free_models_no_log(self, caplog):
        caplog.set_level(logging.INFO)
        data = {"data": [_entry("paid-0"), _entry("paid-1")]}
        models = fetch_models(_response_json=data)
        assert len(models) == 2
        assert "Skipped" not in caplog.text
