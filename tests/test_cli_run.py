"""Tests for routesmith run CLI (P5b-lifecycle, R7.1)."""

from __future__ import annotations

import io
import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_run_args(
    command: list[str],
    host: str = "127.0.0.1",
    port: int = 9119,
    config: str | None = None,
    family: str | None = None,
    api_key: str = "",
) -> Namespace:
    return Namespace(
        command=command,
        host=host,
        port=port,
        config=config,
        family=family,
        api_key=api_key,
    )


def _mock_health_ok() -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value.read.return_value = b'{"status":"healthy"}'
    mock_resp.__enter__.return_value.status = 200
    return mock_resp


class TestRunConfigResolution:
    """Config resolution: --config > ./routesmith.yaml > ~/.routesmith/routesmith.yaml."""

    def test_no_config_exits_1_with_quickstart_hint(self):
        from routesmith.cli.run import run_run

        args = _make_run_args(["echo", "hi"])
        with patch("routesmith.cli.run._resolve_config", return_value=None):
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    rc = run_run(args)

        assert rc == 1
        errmsg = stderr.getvalue()
        assert "No RouteSmith config found" in errmsg
        assert "routesmith quickstart" in errmsg

    def test_resolve_config_explicit(self):
        from routesmith.cli.run import _resolve_config

        with patch("pathlib.Path.exists", return_value=True):
            result = _resolve_config("/some/path.yaml")
            assert result is not None
            assert str(result) == "/some/path.yaml"

    def test_resolve_config_explicit_not_found(self):
        from routesmith.cli.run import _resolve_config

        with patch("pathlib.Path.exists", return_value=False):
            result = _resolve_config("/nonexistent.yaml")
            assert result is None

    def test_resolve_config_fallback_cwd(self):
        from routesmith.cli.run import _resolve_config

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.side_effect = [True, False]
            result = _resolve_config(None)
            assert result is not None
            assert result.name == "routesmith.yaml"

    def test_resolve_config_fallback_home(self):
        from routesmith.cli.run import _resolve_config

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.side_effect = [False, True]
            with patch("pathlib.Path.home", return_value=Path("/fake/home")):
                result = _resolve_config(None)
                assert result is not None
                assert ".routesmith" in str(result)

    def test_resolve_config_none_found(self):
        from routesmith.cli.run import _resolve_config

        with patch("pathlib.Path.exists", return_value=False):
            result = _resolve_config(None)
            assert result is None


class TestRunFamilyDetection:
    """Family detection by command basename (R7.1)."""

    def test_claude_is_anthropic(self):
        from routesmith.cli.run import _detect_family

        assert _detect_family("claude", None) == "anthropic"
        assert _detect_family("/usr/bin/claude", None) == "anthropic"
        assert _detect_family("claude-code", None) == "anthropic"

    def test_codex_is_openai(self):
        from routesmith.cli.run import _detect_family

        assert _detect_family("codex", None) == "openai"
        assert _detect_family("opencode", None) == "openai"
        assert _detect_family("python", None) == "openai"

    def test_family_override(self):
        from routesmith.cli.run import _detect_family

        assert _detect_family("claude", "openai") == "openai"
        assert _detect_family("codex", "anthropic") == "anthropic"


class TestRunEnvInjection:
    """Correct env injected per family, parent env unmodified."""

    @patch("routesmith.cli.run.os.execvpe")
    @patch("routesmith.cli.run.urlopen")
    def test_anthropic_family_injects_vars(self, mock_urlopen, mock_execvpe):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        args = _make_run_args(["claude", "--version"])

        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            rc = run_run(args)

        assert rc == 0
        assert mock_execvpe.called
        _call_env = mock_execvpe.call_args[0][2]
        assert "ANTHROPIC_BASE_URL" in _call_env
        assert "ANTHROPIC_API_KEY" in _call_env
        assert _call_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9119"
        assert _call_env["ANTHROPIC_API_KEY"] == "routesmith"

    @patch("routesmith.cli.run.os.execvpe")
    @patch("routesmith.cli.run.urlopen")
    def test_openai_family_injects_vars(self, mock_urlopen, mock_execvpe):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        args = _make_run_args(["codex", "--flag"])

        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            rc = run_run(args)

        assert rc == 0
        assert mock_execvpe.called
        _call_env = mock_execvpe.call_args[0][2]
        assert "OPENAI_BASE_URL" in _call_env
        assert "OPENAI_API_KEY" in _call_env
        assert _call_env["OPENAI_BASE_URL"] == "http://127.0.0.1:9119/v1"
        assert _call_env["OPENAI_API_KEY"] == "routesmith"

    @patch("routesmith.cli.run.os.execvpe")
    @patch("routesmith.cli.run.urlopen")
    def test_family_override_works(self, mock_urlopen, mock_execvpe):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        args = _make_run_args(["claude"], family="openai")

        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            rc = run_run(args)

        assert rc == 0
        _call_env = mock_execvpe.call_args[0][2]
        assert "OPENAI_BASE_URL" in _call_env
        assert "OPENAI_API_KEY" in _call_env

    @patch("routesmith.cli.run.os.execvpe")
    @patch("routesmith.cli.run.urlopen")
    def test_parent_env_unmodified(self, mock_urlopen, mock_execvpe):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        _original_openai = os.environ.get("OPENAI_BASE_URL", "__NOT_SET__")

        args = _make_run_args(["codex"])
        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            with patch.dict(os.environ, {"EXISTING_VAR": "yes"}, clear=True):
                rc = run_run(args)

        assert rc == 0
        _call_env = mock_execvpe.call_args[0][2]
        assert "EXISTING_VAR" in _call_env
        assert _call_env["EXISTING_VAR"] == "yes"
        assert os.environ.get("OPENAI_BASE_URL") is None

    @patch("routesmith.cli.run.urlopen")
    def test_env_always_overrides_api_key(self, mock_urlopen):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        args = _make_run_args(["codex"])

        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            with patch("routesmith.cli.run.os.execvpe") as mock_execvpe:
                with patch.dict(os.environ, {"OPENAI_API_KEY": "real-key"}, clear=True):
                    rc = run_run(args)

        assert rc == 0
        _call_env = mock_execvpe.call_args[0][2]
        assert _call_env["OPENAI_API_KEY"] == "routesmith"


class TestRunExitCode:
    """Exit code propagation via os.execvpe (mock)."""

    @patch("routesmith.cli.run.os.execvpe")
    @patch("routesmith.cli.run.urlopen")
    def test_exec_called_with_correct_args(self, mock_urlopen, mock_execvpe):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        args = _make_run_args(["opencode", "arg1", "arg2"])

        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            rc = run_run(args)

        assert rc == 0
        mock_execvpe.assert_called_once()
        call_path, call_args, call_env = mock_execvpe.call_args[0]
        assert call_path == "opencode"
        assert call_args == ["opencode", "arg1", "arg2"]


class TestNoCommand:
    """Empty command list."""

    @patch("routesmith.cli.run.urlopen")
    def test_no_command(self, mock_urlopen):
        from routesmith.cli.run import run_run

        mock_urlopen.return_value = _mock_health_ok()
        args = _make_run_args([])
        args.command = []

        with patch("routesmith.cli.run._resolve_config", return_value=Path("/fake/config.yaml")):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                rc = run_run(args)

        assert rc == 1
        assert "No command specified" in stderr.getvalue()
