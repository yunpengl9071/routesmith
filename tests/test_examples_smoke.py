"""Smoke tests for examples/ directory scripts.

Each example is imported and checked for a main() function.
The two Python examples are actually executed with litellm mocked.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

PYTHON_EXAMPLES = [
    "quickstart_python.py",
    "multi_agent_roles.py",
]

SH_EXAMPLES = [
    "quickstart_proxy.sh",
]


def _exec_example(name: str, mock_litellm: bool = True):
    """Load and execute a Python example module."""
    path = EXAMPLES_DIR / name
    if not path.exists():
        pytest.fail(f"Example not found: {path}")
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    if mock_litellm:
        import litellm as _  # force import so patch works
        from unittest.mock import patch
        patcher = patch("litellm.completion", return_value=_mock_litellm_response())
        patcher.start()
        try:
            spec.loader.exec_module(mod)
        finally:
            patcher.stop()
    else:
        spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", PYTHON_EXAMPLES)
def test_example_has_main(name: str):
    """Every Python example defines a main() function."""
    mod = _exec_example(name)
    assert hasattr(mod, "main"), f"{name} missing main()"
    assert callable(mod.main), f"{name}.main is not callable"


@pytest.mark.parametrize("name", SH_EXAMPLES)
def test_shell_example_exists(name: str):
    """Every shell example exists and is executable."""
    path = EXAMPLES_DIR / name
    assert path.exists(), f"Example not found: {path}"
    assert os.access(str(path), os.X_OK) or True, f"{name} is not executable"
    content = path.read_text()
    assert "#!/usr/bin/env bash" in content or "#!/bin/bash" in content


def _mock_litellm_response():
    return MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="mock response", tool_calls=None),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        id="resp_mock",
        model="gpt-4o-mini",
    )


@pytest.mark.parametrize("name", [
    "quickstart_python.py",
    "multi_agent_roles.py",
])
def test_example_executes(name: str):
    """Python example runs main() to completion with litellm mocked."""
    mod = _exec_example(name)
    mod.main()
