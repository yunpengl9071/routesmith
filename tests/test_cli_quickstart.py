"""Tests for routesmith quickstart CLI."""

from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from unittest.mock import patch


def test_quickstart_no_key_prints_help():
    """Without any API key, quickstart prints help and exits 1."""
    from routesmith.cli.quickstart import run_quickstart

    args = Namespace(port=9119, yes=False)
    with patch.dict(os.environ, {}, clear=True):
        result = run_quickstart(args)
    assert result == 1


def test_quickstart_detects_openrouter():
    """Quickstart detects OPENROUTER_API_KEY."""
    from routesmith.cli.quickstart import _detect_provider

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
        info = _detect_provider()
        assert info is not None
        assert info[0] == "OpenRouter"


def test_quickstart_detects_openai():
    """Quickstart detects OPENAI_API_KEY."""
    from routesmith.cli.quickstart import _detect_provider

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        info = _detect_provider()
        assert info is not None
        assert info[0] == "OpenAI"


def test_quickstart_detects_anthropic():
    """Quickstart detects ANTHROPIC_API_KEY."""
    from routesmith.cli.quickstart import _detect_provider

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
        info = _detect_provider()
        assert info is not None
        assert info[0] == "Anthropic"


def test_quickstart_generates_config():
    """Quickstart --yes generates a valid YAML config file."""
    from routesmith.cli.quickstart import _generate_config

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        config_path = f.name

    try:
        result = _generate_config(config_path, "OpenRouter", yes=True)
        assert result == 0

        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert "predictor_type" in config
        assert "budget" in config
    finally:
        os.unlink(config_path)


def test_quickstart_generates_config_without_overwrite():
    """Without --yes, quickstart refuses to overwrite existing config."""
    from routesmith.cli.quickstart import _generate_config

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("existing: true\n")
        config_path = f.name

    try:
        result = _generate_config(config_path, "OpenRouter", yes=False)
        assert result == 1
    finally:
        os.unlink(config_path)
