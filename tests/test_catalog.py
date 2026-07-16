"""Tests for provider-aware model catalog (spec R1)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _catalog_path(provider: str) -> Path:
    import routesmith.registry.data
    return Path(routesmith.registry.data.__file__).parent / "catalogs" / f"{provider}.json"


CATALOG_PROVIDERS = ["anthropic", "openai", "openrouter", "groq"]
REQUIRED_KEYS = {
    "model_id", "display_name", "cost_per_1k_input", "cost_per_1k_output",
    "quality_score", "context_window", "latency_p50_ms",
    "supports_tools", "supports_vision", "default",
}


class TestDetectProviders:
    def test_no_keys(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {}, clear=True):
            assert detect_providers() == []

    def test_anthropic_only(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            assert detect_providers() == ["anthropic"]

    def test_openai_only(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            assert detect_providers() == ["openai"]

    def test_openrouter_only(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            assert detect_providers() == ["openrouter"]

    def test_groq_only(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=True):
            assert detect_providers() == ["groq"]

    def test_all_keys(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "OPENAI_API_KEY": "sk-test",
            "OPENROUTER_API_KEY": "sk-or-test",
            "GROQ_API_KEY": "gsk_test",
        }, clear=True):
            assert detect_providers() == ["anthropic", "openai", "openrouter", "groq"]

    def test_partial_keys(self):
        from routesmith.registry.catalog import detect_providers

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "GROQ_API_KEY": "gsk_test",
        }, clear=True):
            assert detect_providers() == ["openai", "groq"]


class TestBuildDefaultPool:
    def test_no_providers_raises(self):
        from routesmith.exceptions import NoProviderDetectedError
        from routesmith.registry.catalog import build_default_pool

        with pytest.raises(NoProviderDetectedError):
            build_default_pool([])

    def test_anthropic_only(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["anthropic"])
        assert len(pool) == 3
        ids = {m["model_id"] for m in pool}
        assert "claude-sonnet-4-5-20251001" in ids
        assert "claude-3-5-haiku-latest" in ids
        assert "claude-opus-4-5-20251001" in ids

    def test_openai_only(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["openai"])
        assert len(pool) == 3
        ids = {m["model_id"] for m in pool}
        assert "gpt-4o-mini" in ids
        assert "gpt-4o" in ids
        assert "o4-mini" in ids

    def test_openrouter_only(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["openrouter"])
        ids = {m["model_id"] for m in pool}
        assert "openrouter/anthropic/claude-sonnet-4-5-20251001" in ids
        assert "openrouter/openai/gpt-4o" in ids
        assert "openrouter/meta-llama/llama-4-maverick" in ids
        assert "openrouter/deepseek/deepseek-chat" in ids

    def test_groq_only(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["groq"])
        assert len(pool) == 2
        ids = {m["model_id"] for m in pool}
        assert "groq/llama-3.3-70b-versatile" in ids
        assert "groq/llama-3.3-70b-specdec" in ids

    def test_anthropic_plus_openrouter_dedupes(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["anthropic", "openrouter"])
        ids = {m["model_id"] for m in pool}
        assert "claude-sonnet-4-5-20251001" in ids
        assert "openrouter/anthropic/claude-sonnet-4-5-20251001" not in ids
        assert "openrouter/openai/gpt-4o" in ids
        assert "openrouter/meta-llama/llama-4-maverick" in ids
        assert "openrouter/deepseek/deepseek-chat" in ids

    def test_openai_plus_openrouter_dedupes(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["openai", "openrouter"])
        ids = {m["model_id"] for m in pool}
        assert "gpt-4o" in ids
        assert "openrouter/openai/gpt-4o" not in ids
        assert "gpt-4o-mini" in ids

    def test_all_providers(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["anthropic", "openai", "openrouter", "groq"])
        ids = {m["model_id"] for m in pool}
        assert "claude-sonnet-4-5-20251001" in ids
        assert "gpt-4o" in ids
        assert "openrouter/meta-llama/llama-4-maverick" in ids
        assert "groq/llama-3.3-70b-versatile" in ids
        assert "openrouter/anthropic/claude-sonnet-4-5-20251001" not in ids
        assert "openrouter/openai/gpt-4o" not in ids

    def test_pool_entries_have_required_keys(self):
        from routesmith.registry.catalog import build_default_pool
        pool = build_default_pool(["anthropic"])
        required = {"model_id", "cost_per_1k_input", "cost_per_1k_output", "quality_score"}
        for entry in pool:
            assert required.issubset(entry.keys()), f"Missing keys in {entry['model_id']}"


class TestCatalogJsonSchema:
    def test_every_catalog_parses(self):
        for provider in CATALOG_PROVIDERS:
            path = _catalog_path(provider)
            assert path.exists(), f"Missing catalog: {path}"
            data = json.loads(path.read_text())
            assert data["provider"] == provider
            assert "generated_at" in data
            assert len(data["models"]) > 0

    def test_every_entry_has_required_keys(self):
        for provider in CATALOG_PROVIDERS:
            path = _catalog_path(provider)
            data = json.loads(path.read_text())
            for entry in data["models"]:
                missing = REQUIRED_KEYS - entry.keys()
                assert not missing, f"{provider}: {entry['model_id']} missing {missing}"

    def test_every_catalog_has_at_least_one_default(self):
        for provider in CATALOG_PROVIDERS:
            path = _catalog_path(provider)
            data = json.loads(path.read_text())
            defaults = [m for m in data["models"] if m.get("default")]
            assert len(defaults) >= 1, f"{provider} has no default models"

    def test_litellm_id_shape(self):
        for provider in CATALOG_PROVIDERS:
            path = _catalog_path(provider)
            data = json.loads(path.read_text())
            for entry in data["models"]:
                mid = entry["model_id"]
                if provider == "openrouter":
                    assert mid.startswith("openrouter/"), f"{mid} should start with openrouter/"
                elif provider == "groq":
                    assert mid.startswith("groq/"), f"{mid} should start with groq/"
                elif provider == "anthropic":
                    assert mid.startswith("claude-"), f"{mid} should start with claude-"
                elif provider == "openai":
                    assert mid.startswith("gpt-") or mid.startswith("o"), f"{mid} should start with gpt- or o"


class TestLoadCatalog:
    def test_load_anthropic(self):
        from routesmith.registry.catalog import load_catalog

        catalog = load_catalog("anthropic")
        assert catalog["provider"] == "anthropic"
        assert len(catalog["models"]) == 3

    def test_load_openai(self):
        from routesmith.registry.catalog import load_catalog

        catalog = load_catalog("openai")
        assert catalog["provider"] == "openai"
        assert len(catalog["models"]) == 3

    def test_load_invalid_provider(self):
        from routesmith.registry.catalog import load_catalog

        with pytest.raises((FileNotFoundError, ModuleNotFoundError)):
            load_catalog("nonexistent")
