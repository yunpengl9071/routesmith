"""Tests for routesmith connect CLI (P5)."""

from __future__ import annotations

import io
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_args(
    tool: str,
    url: str = "http://localhost:9119",
    apply: bool = False,
    verify: bool = False,
    yes: bool = False,
    proxy_api_key: str = "",
) -> Namespace:
    return Namespace(
        tool=tool,
        url=url,
        apply=apply,
        verify=verify,
        yes=yes,
        proxy_api_key=proxy_api_key,
    )


class TestConnectSnippetEmission:
    """All eight tools emit --url and never print real key values (spec §6)."""

    def _check_snippet(self, tool: str, url: str) -> None:
        from routesmith.cli.connect import run_connect

        args = _make_args(tool, url=url)
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = run_connect(args)
        assert rc == 0, f"{tool}: exit code {rc}"
        output = stdout.getvalue()
        assert url in output, f"{tool}: --url {url} not in output"
        assert "sk-" not in output, f"{tool}: real key pattern in output"

    def test_claude_code_snippet(self):
        self._check_snippet("claude-code", "http://localhost:9119")

    def test_codex_snippet(self):
        self._check_snippet("codex", "http://localhost:9119")

    def test_opencode_snippet(self):
        self._check_snippet("opencode", "http://localhost:9119")

    def test_openclaw_snippet(self):
        self._check_snippet("openclaw", "http://localhost:9119")

    def test_pi_snippet(self):
        self._check_snippet("pi", "http://localhost:9119")

    def test_hermes_snippet(self):
        self._check_snippet("hermes", "http://localhost:9119")

    def test_openai_sdk_snippet(self):
        self._check_snippet("openai-sdk", "http://localhost:9119")

    def test_anthropic_sdk_snippet(self):
        self._check_snippet("anthropic-sdk", "http://localhost:9119")

    def test_custom_url_appears(self):
        self._check_snippet("opencode", "http://my-proxy:9999")

    def test_unknown_tool_exits_2(self):
        from routesmith.cli.connect import run_connect

        args = _make_args("nonexistent")
        with patch("sys.stdout", new_callable=io.StringIO):
            with patch("sys.stderr", new_callable=io.StringIO):
                rc = run_connect(args)
        assert rc == 2


class TestConnectApply:
    """--apply for codex and openclaw/pi under monkeypatched HOME."""

    def test_codex_apply_writes_file(self):
        from routesmith.cli.connect import run_connect

        with tempfile.TemporaryDirectory() as tmp_home:
            with patch("pathlib.Path.home", return_value=Path(tmp_home)):
                args = _make_args("codex", apply=True, yes=True)
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    rc = run_connect(args)
                    output = stdout.getvalue()
                    assert rc == 0
                    assert "Wrote Codex config" in output

                    config_path = Path(tmp_home) / ".codex" / "config.yaml"
                    assert config_path.exists()
                    content = config_path.read_text()
                    assert "base_url:" in content
                    assert "wire_api: chat" in content

    def test_codex_apply_refuses_overwrite(self):
        from routesmith.cli.connect import run_connect

        with tempfile.TemporaryDirectory() as tmp_home:
            codex_dir = Path(tmp_home) / ".codex"
            codex_dir.mkdir(parents=True)
            (codex_dir / "config.yaml").write_text("existing: true\n")

            with patch("pathlib.Path.home", return_value=Path(tmp_home)):
                args = _make_args("codex", apply=True, yes=False)
                with patch("sys.stdout", new_callable=io.StringIO):
                    with patch("sys.stderr", new_callable=io.StringIO):
                        rc = run_connect(args)
                        assert rc == 1

    def test_codex_apply_yes_overwrites(self):
        from routesmith.cli.connect import run_connect

        with tempfile.TemporaryDirectory() as tmp_home:
            codex_dir = Path(tmp_home) / ".codex"
            codex_dir.mkdir(parents=True)
            orig = codex_dir / "config.yaml"
            orig.write_text("existing: true\n")

            with patch("pathlib.Path.home", return_value=Path(tmp_home)):
                args = _make_args("codex", apply=True, yes=True)
                with patch("sys.stdout", new_callable=io.StringIO):
                    rc = run_connect(args)
                    assert rc == 0
                    content = orig.read_text()
                    assert "wire_api: chat" in content
                    assert "existing:" not in content

    def test_openclaw_apply_writes_file(self):
        from routesmith.cli.connect import run_connect

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                args = _make_args("openclaw", apply=True, yes=True)
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    rc = run_connect(args)
                assert rc == 0
                output = stdout.getvalue()
                assert "Wrote provider config" in output

                config_path = Path(tmpdir) / "routesmith-provider.json"
                assert config_path.exists()
                data = json.loads(config_path.read_text())
                assert "models" in data
            finally:
                os.chdir(orig_cwd)

    def test_openclaw_apply_refuses_overwrite(self):
        from routesmith.cli.connect import run_connect

        with tempfile.TemporaryDirectory() as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                Path(tmpdir, "routesmith-provider.json").write_text("{}")
                args = _make_args("openclaw", apply=True, yes=False)
                with patch("sys.stdout", new_callable=io.StringIO):
                    with patch("sys.stderr", new_callable=io.StringIO):
                        rc = run_connect(args)
                assert rc == 1
            finally:
                os.chdir(orig_cwd)

    def test_other_tools_apply_prints_note(self):
        from routesmith.cli.connect import run_connect

        args = _make_args("hermes", apply=True, yes=True)
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            rc = run_connect(args)
        assert rc == 0
        output = stdout.getvalue()
        assert "manual placement is required" in output.lower()


class TestConnectVerify:
    """--verify against mocked proxy responses."""

    def _mock_urlopen_json(self, status: int = 200, data: dict | None = None):
        """Return a mock urlopen that returns JSON data."""
        body = json.dumps(data or {}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value.read.return_value = body
        mock_resp.__enter__.return_value.status = status
        return mock_resp

    def test_verify_unreachable_proxy(self):
        from routesmith.cli.connect import run_connect

        args = _make_args("opencode", verify=True)
        with patch("sys.stdout", new_callable=io.StringIO):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with patch(
                    "routesmith.cli.connect._http_get",
                    side_effect=ConnectionError("refused"),
                ):
                    rc = run_connect(args)
        assert rc == 1
        err = stderr.getvalue()
        assert "Could not connect" in err
        assert "routesmith serve" in err

    def test_verify_health_fails(self):
        from routesmith.cli.connect import run_connect

        health_resp = {"status": "not_healthy"}

        args = _make_args("opencode", verify=True)

        with patch("sys.stdout", new_callable=io.StringIO):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with patch(
                    "routesmith.cli.connect._http_get",
                    return_value=health_resp,
                ):
                    rc = run_connect(args)
        assert rc == 1
        err = stderr.getvalue()
        assert "Health check failed" in err

    def test_verify_downstream_failure_prints_clean_error(self):
        """A 500 from the completion request (e.g. the selected model's
        provider key is missing/invalid downstream) must print a clean,
        actionable error and exit 1 — not dump an unhandled urllib
        traceback to the user's terminal.

        Regression test: --verify's whole purpose is to give a clear
        pass/fail signal for a freshly-configured proxy; a raw traceback
        on the single most common first-run failure (bad/missing provider
        key) defeats that purpose.
        """
        from urllib.error import HTTPError

        from routesmith.cli.connect import run_connect

        health_resp = {"status": "healthy", "registered_models": 2}
        error_body = json.dumps({
            "error": {
                "message": "litellm.InternalServerError: Missing credentials for OPENAI_API_KEY",
                "type": "invalid_request_error",
            }
        }).encode("utf-8")

        def _raise_http_error(*a, **kw):
            raise HTTPError(
                "http://localhost:9119/v1/chat/completions", 500, "Internal Server Error",
                {}, io.BytesIO(error_body),
            )

        args = _make_args("opencode", verify=True)

        with patch("sys.stdout", new_callable=io.StringIO):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with patch(
                    "routesmith.cli.connect._http_get",
                    return_value=health_resp,
                ):
                    with patch(
                        "routesmith.cli.connect._send_verify_request",
                        side_effect=_raise_http_error,
                    ):
                        rc = run_connect(args)
        assert rc == 1
        err = stderr.getvalue()
        assert "500" in err
        assert "Missing credentials" in err

    def test_verify_routed_success(self):
        from routesmith.cli.connect import run_connect

        health_resp = {"status": "healthy", "registered_models": 2}
        completion_resp = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "hello"}}],
            "routesmith_metadata": {
                "routed": True,
                "selected_model": "gpt-4o-mini",
                "requested_model": "gpt-4o",
                "passthrough_reason": None,
                "estimated_cost_usd": 0.00015,
            },
        }

        args = _make_args("opencode", verify=True)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with patch(
                "routesmith.cli.connect._http_get",
                return_value=health_resp,
            ):
                with patch(
                    "routesmith.cli.connect._http_post",
                    return_value=completion_resp,
                ):
                    rc = run_connect(args)
        assert rc == 0
        output = stdout.getvalue()
        assert "Verified:" in output
        assert "gpt-4o-mini" in output
        assert "0.000150" in output

    def test_verify_not_routed_exits_1(self):
        from routesmith.cli.connect import run_connect

        health_resp = {"status": "healthy", "registered_models": 2}
        completion_resp = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "hello"}}],
            "routesmith_metadata": {
                "routed": False,
                "selected_model": "gpt-4o",
                "requested_model": "gpt-4o",
                "passthrough_reason": "intercept_auto",
                "estimated_cost_usd": 0.0,
            },
        }

        args = _make_args("opencode", verify=True)

        with patch("sys.stdout", new_callable=io.StringIO):
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with patch(
                    "routesmith.cli.connect._http_get",
                    return_value=health_resp,
                ):
                    with patch(
                        "routesmith.cli.connect._http_post",
                        return_value=completion_resp,
                    ):
                        rc = run_connect(args)
        assert rc == 1
        err = stderr.getvalue()
        assert "routing.intercept: all" in err

    def test_verify_anthropic_endpoint(self):
        from routesmith.cli.connect import run_connect

        health_resp = {"status": "healthy", "registered_models": 2}
        completion_resp = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hello"}],
            "routesmith_metadata": {
                "routed": True,
                "selected_model": "claude-haiku-4-5",
                "requested_model": "claude-sonnet-4-5",
                "passthrough_reason": None,
                "estimated_cost_usd": 0.001,
            },
        }

        args = _make_args("claude-code", verify=True)

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with patch(
                "routesmith.cli.connect._http_get",
                return_value=health_resp,
            ):
                with patch(
                    "routesmith.cli.connect._http_post",
                    return_value=completion_resp,
                ):
                    rc = run_connect(args)
        assert rc == 0
        output = stdout.getvalue()
        assert "Verified:" in output
        assert "claude-haiku-4-5" in output

    def test_verify_with_proxy_api_key(self):
        from routesmith.cli.connect import run_connect

        health_resp = {"status": "healthy", "registered_models": 2}
        completion_resp = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "hello"}}],
            "routesmith_metadata": {
                "routed": True,
                "selected_model": "gpt-4o-mini",
                "requested_model": "gpt-4o",
                "passthrough_reason": None,
            },
        }

        args = _make_args("opencode", verify=True, proxy_api_key="test-key-123")

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            with patch(
                "routesmith.cli.connect._http_get",
                return_value=health_resp,
            ):
                with patch(
                    "routesmith.cli.connect._http_post",
                    return_value=completion_resp,
                ):
                    rc = run_connect(args)
        assert rc == 0
        output = stdout.getvalue()
        assert "Verified:" in output

    def test_verify_routesmith_api_key_env(self):
        from routesmith.cli.connect import run_connect

        health_resp = {"status": "healthy", "registered_models": 2}
        completion_resp = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "hello"}}],
            "routesmith_metadata": {
                "routed": True,
                "selected_model": "gpt-4o-mini",
                "requested_model": "gpt-4o",
                "passthrough_reason": None,
            },
        }

        args = _make_args("opencode", verify=True, proxy_api_key="")

        with patch("sys.stdout", new_callable=io.StringIO):
            with patch.dict(os.environ, {"ROUTESMITH_API_KEY": "env-key"}):
                with patch(
                    "routesmith.cli.connect._http_get",
                    return_value=health_resp,
                ):
                    with patch(
                        "routesmith.cli.connect._http_post",
                        return_value=completion_resp,
                    ):
                        rc = run_connect(args)
        assert rc == 0


class TestConnectPathHomeResolution:
    """Path.home() must be resolved at call time, not import time."""

    def test_home_is_not_resolved_at_import(self):
        from pathlib import Path

        import routesmith.cli.connect as connect_mod

        assert callable(connect_mod._home)
        with patch("pathlib.Path.home", return_value=Path("/fake/home")):
            result = connect_mod._home()
        assert result == Path("/fake/home")
