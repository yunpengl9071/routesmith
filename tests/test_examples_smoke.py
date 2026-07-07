"""Smoke tests for examples/ directory scripts.

Each example is imported and checked for a main() function.
Examples requiring optional deps use pytest.importorskip.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

PYTHON_EXAMPLES = [
    "quickstart_python.py",
    "multi_agent_roles.py",
    "langgraph_agents.py",
    "crewai_crew.py",
    "autogen_pair.py",
    "dspy_pipeline.py",
    "openai_agents_sdk.py",
    "pydantic_ai_agent.py",
    "llamaindex_engine.py",
]

SH_EXAMPLES = [
    "quickstart_proxy.sh",
]

_OPTIONAL_DEPS = {
    "langgraph_agents.py": "langchain_core",
    "crewai_crew.py": "crewai",
    "autogen_pair.py": "autogen",
    "dspy_pipeline.py": "dspy",
    "openai_agents_sdk.py": "openai",
    "pydantic_ai_agent.py": "pydantic_ai",
    "llamaindex_engine.py": "llama_index",
}


def _exec_example(name: str):
    """Load and execute a Python example module."""
    path = EXAMPLES_DIR / name
    if not path.exists():
        pytest.fail(f"Example not found: {path}")

    dep = _OPTIONAL_DEPS.get(name)
    if dep:
        pytest.importorskip(dep, reason=f"{name} requires {dep}")

    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
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
    content = path.read_text()
    assert "#!/usr/bin/env bash" in content or "#!/bin/bash" in content
