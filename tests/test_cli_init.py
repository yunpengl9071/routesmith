"""Tests for `routesmith init --provider` (non-interactive path).

The interactive OpenRouter-picker path (no --provider) is exercised via
routesmith.registry.openrouter and is not duplicated here.
"""

from __future__ import annotations

import os
import tempfile
from argparse import Namespace

import pytest


def _run_init(config_path: str, providers: list[str]) -> int:
    from routesmith.cli.init import run_init

    args = Namespace(output=config_path, force=True, provider=providers)
    return run_init(args)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "groq"])
def test_init_provider_output_is_loadable(provider):
    """`routesmith init --provider X` must write a config that
    `routesmith serve` can actually load.

    Regression test: this path wrote build_default_pool()'s catalog-schema
    dicts (key "model_id") straight into routesmith.yaml, same bug as
    quickstart and `models --refresh` — see test_cli_quickstart.py's
    test_quickstart_output_is_loadable for the full failure mode.
    """
    from routesmith.cli.yaml_loader import load_config_file

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name

    try:
        assert _run_init(config_path, [provider]) == 0

        _, models = load_config_file(config_path)
        assert len(models) > 0
        for m in models:
            assert m["model_id"]
            assert m.get("supports_function_calling", True) is True
    finally:
        os.unlink(config_path)


def test_init_provider_writes_intercept_all():
    """Provider-driven init writes the just-works routing defaults."""
    import yaml

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name

    try:
        assert _run_init(config_path, ["openai"]) == 0
        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["routing"]["intercept"] == "all"
        assert data["routing"]["sticky"] == "auto"
        assert "refreshed_at" in data["catalog"]
    finally:
        os.unlink(config_path)
