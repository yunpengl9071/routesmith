"""Tests for routesmith models CLI."""

from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from unittest.mock import patch

import yaml

# Uses the on-disk model schema (routesmith.yaml.example / yaml_loader._parse_model_entry):
# key "id", not the internal catalog schema key "model_id" (registry.catalog.build_default_pool).
# Conflating the two is exactly the bug this file's round-trip tests guard against — see
# test_models_refresh_output_is_loadable and test_models_refresh_preserves_tool_calling_capability.
SAMPLE_MODELS = [
    {"id": "gpt-4o-mini", "cost_per_1k_input": 0.00015, "cost_per_1k_output": 0.0006, "quality_score": 0.82},
    {"id": "claude-3-5-haiku-latest", "cost_per_1k_input": 0.0008, "cost_per_1k_output": 0.004, "quality_score": 0.80},
    {"id": "groq/llama-3.3-70b-versatile", "cost_per_1k_input": 0.00059, "cost_per_1k_output": 0.00079, "quality_score": 0.90},
    {"id": "openrouter/anthropic/claude-sonnet-4-5-20251001", "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015, "quality_score": 0.92},
]

SAMPLE_CONFIG = {
    "routing": {"intercept": "all", "sticky": "auto"},
    "predictor_type": "lints",
    "models": SAMPLE_MODELS,
}


def _write_config(config_path: str, data: dict | None = None) -> None:
    with open(config_path, "w") as f:
        yaml.dump(data or SAMPLE_CONFIG, f, default_flow_style=False, sort_keys=False)


def test_models_no_config():
    """Without a config file, models prints help and exits 1."""
    from routesmith.cli.models import run_models

    args = Namespace(config="nonexistent.yaml", refresh=False, json=False, provider=None)
    result = run_models(args)
    assert result == 1


def test_models_lists_models():
    """List models from config file."""
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=False, json=False, provider=None)
        result = run_models(args)
        assert result == 0
    finally:
        os.unlink(config_path)


def test_models_list_shows_real_ids(capsys):
    """The printed table shows actual model ids, not blanks.

    Regression test: the display code used to read m["model_id"] while the
    on-disk schema uses "id", so every row printed an empty Model ID column
    against any real (hand-written or generated) config.
    """
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=False, json=False, provider=None)
        run_models(args)
        out = capsys.readouterr().out
        assert "gpt-4o-mini" in out
        assert "claude-3-5-haiku-latest" in out
    finally:
        os.unlink(config_path)


def test_models_json_output():
    """--json flag outputs machine-readable model list."""
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=False, json=True, provider=None)
        result = run_models(args)
        assert result == 0
    finally:
        os.unlink(config_path)


def test_models_no_models():
    """Config without models section prints helpful message."""
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path, {"routing": {"strategy": "direct"}})

    try:
        args = Namespace(config=config_path, refresh=False, json=False, provider=None)
        result = run_models(args)
        assert result == 0
    finally:
        os.unlink(config_path)


def test_models_refresh_no_keys():
    """--refresh without API keys exits 1."""
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=True, json=False, provider=None)
        with patch.dict(os.environ, {}, clear=True):
            result = run_models(args)
        assert result == 1
    finally:
        os.unlink(config_path)


def test_models_refresh_with_provider():
    """--refresh with --provider rebuilds the pool from catalog, using the
    on-disk "id" key (not the internal catalog "model_id" key)."""
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=True, json=False, provider=["openai"])
        result = run_models(args)
        assert result == 0

        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "catalog" in data
        assert data["catalog"]["providers"] == ["openai"]
        assert len(data["models"]) == 3
        ids = {m["id"] for m in data["models"]}
        assert "gpt-4o-mini" in ids
        # The bug this guards against: writing "model_id" instead of "id"
        # produces a config that crashes on load (KeyError: 'id').
        assert all("model_id" not in m for m in data["models"])
    finally:
        os.unlink(config_path)


def test_models_refresh_writes_catalog_block():
    """--refresh writes catalog.refreshed_at and catalog.providers."""
    from routesmith.cli.models import run_models

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=True, json=False, provider=["anthropic"])
        result = run_models(args)
        assert result == 0

        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert "refreshed_at" in data["catalog"]
        assert data["catalog"]["providers"] == ["anthropic"]
    finally:
        os.unlink(config_path)


def test_models_refresh_output_is_loadable():
    """The config `models --refresh` writes must actually load via
    load_config_file() and register models — the real end-to-end contract.

    This is the regression test for the bug where `routesmith quickstart`
    (and `models --refresh`, and `init --provider`) wrote catalog-schema
    dicts (key "model_id") straight into routesmith.yaml, which
    yaml_loader._parse_model_entry can't parse (it requires "id") — so
    `routesmith serve` crashed with "Error loading config: 'id'" on any
    config these commands generated.
    """
    from routesmith.cli.models import run_models
    from routesmith.cli.yaml_loader import load_config_file

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=True, json=False, provider=["anthropic"])
        assert run_models(args) == 0

        # Must not raise — this is what routesmith serve does with the file.
        _, models = load_config_file(config_path)
        assert len(models) == 3
        model_ids = {m["model_id"] for m in models}
        assert "claude-sonnet-4-5-20251001" in model_ids or any(
            "claude" in mid for mid in model_ids
        )
    finally:
        os.unlink(config_path)


def test_models_refresh_preserves_tool_calling_capability():
    """Anthropic catalog models are tool-capable; that must survive the
    catalog -> YAML -> load_config_file round trip as
    supports_function_calling=True (the on-disk key), not be silently
    dropped because the catalog wrote "supports_tools" instead.
    """
    from routesmith.cli.models import run_models
    from routesmith.cli.yaml_loader import load_config_file

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name
        _write_config(config_path)

    try:
        args = Namespace(config=config_path, refresh=True, json=False, provider=["anthropic"])
        assert run_models(args) == 0

        _, models = load_config_file(config_path)
        assert all(m.get("supports_function_calling", True) is True for m in models)
    finally:
        os.unlink(config_path)


def test_provider_from_id():
    """_provider_from_id correctly classifies known prefixes."""
    from routesmith.cli.models import _provider_from_id

    assert _provider_from_id("openrouter/anthropic/claude-sonnet") == "openrouter"
    assert _provider_from_id("groq/llama-3.3-70b") == "groq"
    assert _provider_from_id("claude-3-5-haiku-latest") == "anthropic"
    assert _provider_from_id("gpt-4o-mini") == "openai"
    assert _provider_from_id("o4-mini") == "openai"
    assert _provider_from_id("custom-model") == "other"
