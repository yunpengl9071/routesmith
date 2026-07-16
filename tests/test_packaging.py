"""Packaging metadata guards.

The distribution is named routesmith-llm, but the import package is
routesmith and PyPI's `routesmith` name belongs to an unrelated project.
Any dependency spec that references the bare name therefore installs a
stranger's package into the user's environment — which is exactly what the
0.8.0–0.9.2 `all` extra did via its self-referential
"routesmith[proxy,...]" entry. These tests parse pyproject.toml directly so
that class of bug fails CI instead of shipping.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 without tomli
    import pytest

    pytest.skip("tomllib unavailable on this interpreter", allow_module_level=True)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _dep_name(spec: str) -> str:
    """Extract the distribution name from a PEP 508 requirement string."""
    return re.split(r"[\[><=!~; ]", spec.strip(), maxsplit=1)[0].lower()


def _load() -> dict:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def _all_requirement_specs(data: dict) -> list[str]:
    project = data["project"]
    specs = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    specs.extend(data.get("tool", {}).get("uv", {}).get("dev-dependencies", []))
    return specs


class TestPackagingMetadata:
    def test_no_dependency_on_bare_routesmith(self):
        """No requirement may resolve to PyPI's unrelated `routesmith` package."""
        data = _load()
        offenders = [
            spec for spec in _all_requirement_specs(data)
            if _dep_name(spec) == "routesmith"
        ]
        assert offenders == [], (
            f"pyproject.toml references the bare 'routesmith' PyPI name "
            f"(an unrelated third-party package): {offenders} — "
            f"self-references must use 'routesmith-llm'"
        )

    def test_all_extra_is_self_referential_to_correct_name(self):
        """The `all` extra must reference this distribution by its real name."""
        data = _load()
        all_extra = data["project"]["optional-dependencies"]["all"]
        assert len(all_extra) == 1
        assert _dep_name(all_extra[0]) == "routesmith-llm"

    def test_all_extra_covers_every_user_facing_extra(self):
        """`all` should include every extra except dev-only ones."""
        data = _load()
        extras = set(data["project"]["optional-dependencies"])
        spec = data["project"]["optional-dependencies"]["all"][0]
        match = re.search(r"\[(.*?)\]", spec)
        assert match is not None, f"'all' extra has no extras bracket: {spec}"
        included = set(match.group(1).split(","))
        dev_only = {"dev", "all", "langchain-agents"}
        missing = extras - included - dev_only
        assert missing == set(), (
            f"extras missing from the 'all' extra: {missing} — "
            f"add them or list them as dev-only in this test"
        )

    def test_project_urls_point_at_real_repository(self):
        """PyPI sidebar links must not point at the nonexistent routesmith org."""
        data = _load()
        for name, url in data["project"].get("urls", {}).items():
            assert "github.com/routesmith/" not in url, (
                f"project.urls[{name}] points at the wrong GitHub org: {url}"
            )

    def test_distribution_name_unchanged(self):
        data = _load()
        assert data["project"]["name"] == "routesmith-llm"
