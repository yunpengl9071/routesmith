"""Tests for daemon lifecycle (serve --daemon, status, down — R7.2)."""

from __future__ import annotations

import io
import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_pidfile_dir() -> Path:
    """Create a temporary ~/.routesmith directory."""
    tmp = Path(tempfile.mkdtemp())
    routesmith_dir = tmp / ".routesmith"
    routesmith_dir.mkdir(parents=True, exist_ok=True)
    return tmp


def _write_pidfile(home_dir: Path, pid: int) -> Path:
    pidfile = home_dir / ".routesmith" / "routesmith.pid"
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(pid) + "\n")
    return pidfile


def _make_serve_args(
    daemon: bool = False,
    host: str = "127.0.0.1",
    port: int = 9119,
    config: str = "routesmith.yaml",
    log_level: str = "INFO",
    api_key: str = "",
) -> Namespace:
    return Namespace(
        daemon=daemon,
        host=host,
        port=port,
        config=config,
        log_level=log_level,
        api_key=api_key,
    )


def _make_status_args(host: str = "127.0.0.1", port: int = 9119) -> Namespace:
    return Namespace(host=host, port=port)


def _make_down_args() -> Namespace:
    return Namespace()


class TestDaemonLifecycle:
    """serve --daemon writes pidfile; status/down manage lifecycle."""

    @patch("routesmith.cli.run.os.fork")
    @patch("routesmith.cli.run._run_server_daemon")
    @patch("routesmith.cli.run._daemon_pidfile")
    @patch("routesmith.cli.run._daemon_logfile")
    @patch("routesmith.cli.run._redirect_stdio")
    def test_daemonize_writes_pidfile(
        self,
        mock_redirect,
        mock_logfile,
        mock_pidfile,
        mock_serve,
        mock_fork,
    ):
        from routesmith.cli.run import _daemonize

        pidfile = Path("/tmp/fake/routesmith.pid")
        logfile = Path("/tmp/fake/routesmith.log")

        mock_fork.side_effect = [0, 0]
        mock_pidfile.return_value = pidfile
        mock_logfile.return_value = logfile

        with patch.object(Path, "write_text") as mock_write:
            _daemonize(pidfile, logfile, exit_parent=True)
            mock_write.assert_called_once()
            written = mock_write.call_args[0][0]
            assert written.strip().isdigit()

    @patch("routesmith.cli.run.os.fork")
    @patch("routesmith.cli.run._run_server_daemon")
    @patch("routesmith.cli.run._redirect_stdio")
    def test_daemonize_parent_exits(self, mock_redirect, mock_serve, mock_fork):
        from routesmith.cli.run import _daemonize

        mock_fork.return_value = 12345

        with patch("routesmith.cli.run.os._exit") as mock_exit:
            _daemonize(Path("/tmp/pid"), Path("/tmp/log"), exit_parent=True)
            mock_exit.assert_called_once_with(0)

    @patch("routesmith.cli.run.os.fork")
    @patch("routesmith.cli.run._run_server_daemon")
    @patch("routesmith.cli.run._redirect_stdio")
    def test_daemonize_parent_returns(self, mock_redirect, mock_serve, mock_fork):
        from routesmith.cli.run import _daemonize

        mock_fork.return_value = 12345

        with patch("routesmith.cli.run.os._exit") as mock_exit:
            _daemonize(Path("/tmp/pid"), Path("/tmp/log"), exit_parent=False)
            mock_exit.assert_not_called()

    @patch("routesmith.cli.run.urlopen")
    def test_status_reports_not_running_no_pidfile(self, mock_urlopen):
        from routesmith.cli.run import run_status

        with tempfile.TemporaryDirectory() as tmp:
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    rc = run_status(_make_status_args())

        assert rc == 0
        output = stdout.getvalue()
        assert "not running" in output.lower()
        assert "pidfile" in output.lower()

    @patch("routesmith.cli.run.urlopen")
    def test_status_reports_running(self, mock_urlopen):
        from routesmith.cli.run import run_status

        mock_health_resp = MagicMock()
        mock_health_resp.__enter__.return_value.read.return_value = (
            b'{"status":"healthy","registered_models":3}'
        )

        mock_stats_resp = MagicMock()
        mock_stats_resp.__enter__.return_value.read.return_value = json.dumps(
            {"registered_models": 3, "routed_requests": 10, "passthrough_requests": 2, "total_cost_usd": 0.05}
        ).encode()

        mock_urlopen.side_effect = [mock_health_resp, mock_stats_resp]

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / ".routesmith" / "routesmith.pid"
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text("999999\n")

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("routesmith.cli.run._is_pid_alive", return_value=True):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        rc = run_status(_make_status_args())

        assert rc == 0
        output = stdout.getvalue()
        assert "RUNNING" in output

    def test_status_stale_pid_dead_process(self):
        from routesmith.cli.run import run_status

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = _write_pidfile(Path(tmp), 1)
            assert pidfile.exists()

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("routesmith.cli.run._is_pid_alive", return_value=False):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        rc = run_status(_make_status_args())

            assert rc == 0
            output = stdout.getvalue()
            assert "not running" in output.lower()
            assert "Cleaned up stale" in output
            assert not pidfile.exists()

    def test_down_no_pidfile(self):
        from routesmith.cli.run import run_down

        with tempfile.TemporaryDirectory() as tmp:
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    rc = run_down(_make_down_args())

        assert rc == 0
        output = stdout.getvalue()
        assert "not running" in output.lower()

    def test_down_already_dead(self):
        from routesmith.cli.run import run_down

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = _write_pidfile(Path(tmp), 1)
            assert pidfile.exists()

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("routesmith.cli.run._is_pid_alive", return_value=False):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        rc = run_down(_make_down_args())

            assert rc == 0
            output = stdout.getvalue()
            assert "already dead" in output.lower()
            assert not pidfile.exists()

    @patch("routesmith.cli.run.os.kill")
    def test_down_sends_sigterm(self, mock_kill):
        from routesmith.cli.run import run_down

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = _write_pidfile(Path(tmp), 999999)
            assert pidfile.exists()

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("routesmith.cli.run._is_pid_alive") as mock_alive:
                    mock_alive.side_effect = [True] + [False] * 12
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        rc = run_down(_make_down_args())

            assert rc == 0
            output = stdout.getvalue()
            assert "terminated" in output.lower()
            assert not pidfile.exists()

            mock_kill.assert_called_once_with(999999, 15)

    def test_status_cleanup_stale_with_unresponsive_health(self):
        from routesmith.cli.run import run_status

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = _write_pidfile(Path(tmp), 999999)

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("routesmith.cli.run._is_pid_alive", return_value=True):
                    with patch("routesmith.cli.run._check_health", return_value=None):
                        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                            rc = run_status(_make_status_args())

            assert rc == 0
            output = stdout.getvalue()
            assert "not running" in output.lower()
            assert "unresponsive" in output.lower()
            assert "Cleaned up stale" in output
            assert not pidfile.exists()

    @patch("routesmith.cli.run.urlopen")
    def test_status_shows_counters(self, mock_urlopen):
        from routesmith.cli.run import run_status

        mock_health_resp = MagicMock()
        mock_health_resp.__enter__.return_value.read.return_value = (
            b'{"status":"healthy"}'
        )

        mock_stats_resp = MagicMock()
        mock_stats_resp.__enter__.return_value.read.return_value = json.dumps(
            {"registered_models": 5, "routed_requests": 42, "passthrough_requests": 3, "total_cost_usd": 0.15}
        ).encode()

        mock_urlopen.side_effect = [mock_health_resp, mock_stats_resp]

        with tempfile.TemporaryDirectory() as tmp:
            _write_pidfile(Path(tmp), 999999)

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                with patch("routesmith.cli.run._is_pid_alive", return_value=True):
                    with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                        rc = run_status(_make_status_args())

        assert rc == 0
        output = stdout.getvalue()
        assert "5 models" in output
        assert "42" in output
        assert "0.15" in output


class TestDaemonServeCommand:
    """serve --daemon integration with run.py daemon lifecycle."""

    @patch("routesmith.cli.run._daemonize")
    def test_serve_daemon_calls_daemonize(self, mock_daemonize):
        from routesmith.cli.serve import run_serve
        from routesmith.config import RouteSmithConfig

        config = RouteSmithConfig()
        with patch("routesmith.cli.run._daemon_pidfile", return_value=Path("/tmp/pid")):
            with patch("routesmith.cli.run._daemon_logfile", return_value=Path("/tmp/log")):
                with patch("routesmith.cli.yaml_loader.load_config_file") as mock_load:
                    mock_load.return_value = (config, [])
                    with patch("routesmith.RouteSmith") as mock_rs:
                        mock_rs_instance = MagicMock()
                        mock_rs_instance.registry.__len__.return_value = 0
                        mock_rs.return_value = mock_rs_instance
                        with patch("routesmith.proxy.server.RouteSmithProxyServer"):
                            with patch("sys.stdout", new_callable=io.StringIO):
                                args = _make_serve_args(daemon=True)
                                rc = run_serve(args)

        assert rc == 1
        mock_daemonize.assert_called_once()


class TestReadPid:
    """_read_pid / _is_pid_alive helpers."""

    def test_read_pid_nonexistent(self):
        from routesmith.cli.run import _read_pid

        with tempfile.TemporaryDirectory() as tmp:
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                assert _read_pid() is None

    def test_read_pid_valid(self):
        from routesmith.cli.run import _read_pid

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / ".routesmith" / "routesmith.pid"
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text("42\n")
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                assert _read_pid() == 42

    def test_read_pid_invalid(self):
        from routesmith.cli.run import _read_pid

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / ".routesmith" / "routesmith.pid"
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text("not_a_number\n")
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                assert _read_pid() is None

    def test_is_pid_alive_true(self):
        from routesmith.cli.run import _is_pid_alive

        assert _is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_false(self):
        from routesmith.cli.run import _is_pid_alive

        assert _is_pid_alive(999999999) is False

    def test_cleanup_pidfile(self):
        from routesmith.cli.run import _cleanup_pidfile

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / ".routesmith" / "routesmith.pid"
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text("42\n")
            assert pidfile.exists()

            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                _cleanup_pidfile()
                assert not pidfile.exists()

    def test_cleanup_nonexistent_no_error(self):
        from routesmith.cli.run import _cleanup_pidfile

        with tempfile.TemporaryDirectory() as tmp:
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                _cleanup_pidfile()


class TestRoutesmithDir:
    """~/.routesmith/ directory management."""

    def test_routesmith_dir_creates(self):
        from routesmith.cli.run import _routesmith_dir

        with tempfile.TemporaryDirectory() as tmp:
            with patch("routesmith.cli.run._home", return_value=Path(tmp)):
                d = _routesmith_dir()
                assert d.exists()
                assert d.name == ".routesmith"

    def test_daemon_pidfile_path(self):
        from routesmith.cli.run import _daemon_pidfile

        with patch("routesmith.cli.run._home", return_value=Path("/tmp/home")):
            pidfile = _daemon_pidfile()
            assert ".routesmith" in str(pidfile)
            assert pidfile.name == "routesmith.pid"

    def test_daemon_logfile_path(self):
        from routesmith.cli.run import _daemon_logfile

        with patch("routesmith.cli.run._home", return_value=Path("/tmp/home")):
            logfile = _daemon_logfile()
            assert ".routesmith" in str(logfile)
            assert logfile.name == "routesmith.log"
