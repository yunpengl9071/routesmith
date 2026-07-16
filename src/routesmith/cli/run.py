"""CLI run command — proxy lifecycle management (R7)."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

ANTHROPIC_COMMANDS = frozenset({"claude", "claude-code"})


def _home() -> Path:
    return Path.home()


def _routesmith_dir() -> Path:
    d = _home() / ".routesmith"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _daemon_pidfile() -> Path:
    return _routesmith_dir() / "routesmith.pid"


def _daemon_logfile() -> Path:
    return _routesmith_dir() / "routesmith.log"


def _resolve_config(config_path_arg: str | None) -> Path | None:
    if config_path_arg:
        p = Path(config_path_arg)
        if p.exists():
            return p
        return None
    for candidate in [Path("routesmith.yaml"), _home() / ".routesmith" / "routesmith.yaml"]:
        if candidate.exists():
            return candidate
    return None


def _detect_family(command: str, family_override: str | None) -> str:
    if family_override:
        return family_override
    cmd = os.path.basename(command)
    if cmd in ANTHROPIC_COMMANDS:
        return "anthropic"
    return "openai"


def _read_pid() -> int | None:
    pidfile = _daemon_pidfile()
    if not pidfile.exists():
        return None
    try:
        return int(pidfile.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_health(url: str, timeout: int = 5) -> dict[str, Any] | None:
    try:
        req = Request(f"{url}/health", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            result: dict[str, Any] = json.loads(body)
            return result
    except (URLError, ConnectionError, OSError, json.JSONDecodeError):
        return None


def _cleanup_pidfile() -> None:
    pidfile = _daemon_pidfile()
    if pidfile.exists():
        pidfile.unlink()


def _redirect_stdio(logfile: Path) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    fd_null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(fd_null, 0)
    os.close(fd_null)
    fd_log = os.open(str(logfile), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd_log, 1)
    os.dup2(fd_log, 2)
    if fd_log > 2:
        os.close(fd_log)


def _daemonize(pidfile: Path, logfile: Path, exit_parent: bool = True) -> None:
    pid = os.fork()
    if pid > 0:
        if exit_parent:
            os._exit(0)
        return

    os.setsid()
    os.umask(0)

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    _redirect_stdio(logfile)
    pidfile.write_text(str(os.getpid()) + "\n")


def _run_server_daemon(
    config_path: Path,
    host: str,
    port: int,
    api_key: str,
    log_level: str,
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("routesmith.daemon")

    from routesmith import RouteSmith
    from routesmith.cli.yaml_loader import load_config_file
    from routesmith.proxy.server import RouteSmithProxyServer, ServerConfig

    logger.info(f"Daemon starting, config={config_path}")
    try:
        routesmith_config, models = load_config_file(config_path)
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        return

    rs = RouteSmith(config=routesmith_config)
    for m in models:
        mid = m.pop("model_id")
        rs.register_model(mid, **m)

    if len(rs.registry) == 0:
        logger.error("No models registered")
        return

    sc = ServerConfig(host=host, port=port, api_key=api_key or None)
    server = RouteSmithProxyServer(rs, sc)
    logger.info(f"Daemon ready on http://{host}:{port}, {len(rs.registry)} models")

    import asyncio
    try:
        asyncio.run(server.serve_forever())
    except Exception as e:
        logger.error(f"Server error: {e}")


def _start_daemon(
    config_path: Path,
    host: str,
    port: int,
    api_key: str,
    log_level: str,
    exit_parent: bool = True,
) -> None:
    pidfile = _daemon_pidfile()
    logfile = _daemon_logfile()
    _daemonize(pidfile, logfile, exit_parent=exit_parent)
    _run_server_daemon(config_path, host, port, api_key, log_level)


def _wait_for_health(url: str, max_retries: int = 30, delay: float = 0.5) -> bool:
    for _ in range(max_retries):
        if _check_health(url) is not None:
            return True
        time.sleep(delay)
    return False


def run_run(args: Namespace) -> int:
    config_path = _resolve_config(getattr(args, "config", None))
    if config_path is None:
        print("No RouteSmith config found.", file=sys.stderr)
        print(file=sys.stderr)
        print("To get started, run:", file=sys.stderr)
        print("  routesmith quickstart", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}"
    api_key = getattr(args, "api_key", "") or os.environ.get("ROUTESMITH_API_KEY", "")

    health = _check_health(url)
    if health is None:
        _start_daemon(config_path, args.host, args.port, api_key, "INFO", exit_parent=False)
        if not _wait_for_health(url):
            print("Failed to start RouteSmith proxy daemon.", file=sys.stderr)
            return 1

    cmd_list = args.command
    if not cmd_list:
        print("No command specified.", file=sys.stderr)
        return 1

    family = _detect_family(cmd_list[0], args.family)

    env = os.environ.copy()
    if family == "anthropic":
        env["ANTHROPIC_BASE_URL"] = url
        env["ANTHROPIC_API_KEY"] = "routesmith"
    else:
        env["OPENAI_BASE_URL"] = f"{url}/v1"
        env["OPENAI_API_KEY"] = "routesmith"

    if sys.platform == "win32":
        import subprocess
        proc = subprocess.Popen(cmd_list, env=env)
        return proc.wait()
    else:
        os.execvpe(cmd_list[0], cmd_list, env)
        return 0


def run_status(args: Namespace) -> int:
    pidfile = _daemon_pidfile()
    url = f"http://{args.host}:{args.port}"
    pid = _read_pid()

    if pid is None:
        print("RouteSmith proxy is not running.")
        print(f"  pidfile: {pidfile} (not found)")
        return 0

    alive = _is_pid_alive(pid)
    health = _check_health(url) if alive else None

    if not alive or health is None:
        reason = f"pid {pid} is dead" if not alive else "/health unresponsive"
        print(f"RouteSmith proxy is not running ({reason}).")
        _cleanup_pidfile()
        print("Cleaned up stale pidfile.")
        return 0

    print("RouteSmith proxy is RUNNING")
    print(f"  PID:     {pid}")
    print(f"  URL:     {url}")

    config_candidate = pidfile.parent / "routesmith.yaml"
    if config_candidate.exists():
        print(f"  Config:  {config_candidate}")

    try:
        req = Request(f"{url}/v1/stats", method="GET")
        with urlopen(req, timeout=5) as resp:
            stats: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        print(f"  Pool:    {stats.get('registered_models', '?')} models")
        print(f"  Routed:  {stats.get('routed_requests', 0)}")
        print(f"  Passed:  {stats.get('passthrough_requests', 0)}")
        print(f"  Cost:    ${float(stats.get('total_cost_usd', 0)):.6f}")
    except Exception:
        pass

    return 0


def run_down(args: Namespace) -> int:
    pid = _read_pid()

    if pid is None:
        print("RouteSmith proxy is not running.")
        return 0

    if not _is_pid_alive(pid):
        print(f"Process {pid} is already dead. Cleaning up.")
        _cleanup_pidfile()
        return 0

    print(f"Sending SIGTERM to {pid}...")
    try:
        os.kill(pid, 15)
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for _ in range(10):
        if not _is_pid_alive(pid):
            break
        time.sleep(0.5)

    if _is_pid_alive(pid):
        print(f"Process {pid} did not terminate.", file=sys.stderr)
        return 1

    print(f"Process {pid} terminated.")
    _cleanup_pidfile()
    return 0
