"""Tests for routesmith models CLI."""

from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from unittest.mock import patch

import yaml

SAMPLE_MODELS = [
    {"model_id": "gpt-4o-mini", "cost_per_1k_input": 0.00015, "cost_per_1k_output": 0.0006, "quality_score": 0.82},
    {"model_id": "claude-3-5-haiku-latest", "cost_per_1k_input": 0.0008, "cost_per_1k_output": 0.004, "quality_score": 0.80},
    {"model_id": "groq/llama-3.3-70b-versatile", "cost_per_1k_input": 0.00059, "cost_per_1k_output": 0.00079, "quality_score": 0.90},
    {"model_id": "openrouter/anthropic/claude-sonnet-4-5-20251001", "cost_per_1k_input": 0.003, "cost_per_1k_output": 0.015, "quality_score": 0.92},
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
    from routesmith.cli import models as models_module
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
    """--refresh with --provider rebuilds the pool from catalog."""
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
        ids = {m["model_id"] for m in data["models"]}
        assert "gpt-4o-mini" in ids
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


def test_provider_from_model_id():
    """_provider_from_model_id correctly classifies known prefixes."""
    from routesmith.cli.models import _provider_from_model_id

    assert _provider_from_model_id("openrouter/anthropic/claude-sonnet") == "openrouter"
    assert _provider_from_model_id("groq/llama-3.3-70b") == "groq"
    assert _provider_from_model_id("claude-3-5-haiku-latest") == "anthropic"
    assert _provider_from_model_id("gpt-4o-mini") == "openai"
    assert _provider_from_model_id("o4-mini") == "openai"
    assert _provider_from_model_id("custom-model") == "other"
