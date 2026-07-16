"""Tests for routesmith quickstart CLI."""
from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from unittest.mock import patch

import pytest


def test_quickstart_no_key_prints_help():
    """Without any API key, quickstart prints help and exits 1."""
    from routesmith.cli.quickstart import run_quickstart

    args = Namespace(port=9119, yes=False, provider=None)
    with patch.dict(os.environ, {}, clear=True):
        result = run_quickstart(args)
    assert result == 1


def test_quickstart_detects_anthropic():
    """Quickstart detects ANTHROPIC_API_KEY and generates anthropic pool."""
    from routesmith.cli.quickstart import run_quickstart

    args = Namespace(port=9119, yes=True, provider=None)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        result = run_quickstart(args)
    assert result == 0


def test_quickstart_detects_openai():
    """Quickstart detects OPENAI_API_KEY and generates openai pool."""
    from routesmith.cli.quickstart import run_quickstart

    args = Namespace(port=9119, yes=True, provider=None)
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        result = run_quickstart(args)
    assert result == 0


def test_quickstart_generates_config():
    """Quickstart --yes generates a valid YAML config file with catalog stamp."""
    import os
    import tempfile

    from routesmith.cli.quickstart import _generate_config
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name

    try:
        result = _generate_config(config_path, ["openai"], yes=True)
        assert result == 0

        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert "catalog" in config
        assert "refreshed_at" in config["catalog"]
        assert config["catalog"]["providers"] == ["openai"]
        assert config["routing"]["intercept"] == "all"
        assert config["routing"]["sticky"] == "auto"
        assert len(config["models"]) > 0
    finally:
        os.unlink(config_path)


def test_quickstart_generates_config_without_overwrite():
    """Without --yes, quickstart refuses to overwrite existing config."""
    from routesmith.cli.quickstart import _generate_config

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("existing: true\n")
        config_path = f.name

    try:
        result = _generate_config(config_path, ["openai"], yes=False)
        assert result == 1
    finally:
        os.unlink(config_path)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter", "groq"])
def test_quickstart_output_is_loadable(provider):
    """The config `routesmith quickstart` writes must actually load via
    load_config_file() and register tool-capable models — the real
    end-to-end contract, not just "has a models key".

    Regression test: _generate_config used to write build_default_pool()'s
    catalog-schema dicts (key "model_id", "supports_tools") straight into
    routesmith.yaml. yaml_loader._parse_model_entry requires key "id" and
    crashes otherwise ("Error loading config: 'id'") — so `routesmith
    quickstart && routesmith serve`, the flagship one-command setup this
    whole CLI exists for, was broken for every provider on every fresh
    install of 0.9.0/0.9.1.
    """
    import os
    import tempfile

    from routesmith.cli.quickstart import _generate_config
    from routesmith.cli.yaml_loader import load_config_file

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name

    try:
        assert _generate_config(config_path, [provider], yes=True) == 0

        # Must not raise — this is exactly what `routesmith serve` does.
        _, models = load_config_file(config_path)
        assert len(models) > 0
        for m in models:
            assert m["model_id"]
            assert m["cost_per_1k_input"] >= 0
            # Every packaged catalog's defaults are tool-capable; that must
            # survive the round trip, not silently reset to a default.
            assert m.get("supports_function_calling", True) is True
    finally:
        os.unlink(config_path)
