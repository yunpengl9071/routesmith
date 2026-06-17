"""Structured JSON logging for RouteSmith."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

_EXTRA_KEYS = {"model_id", "request_id", "cost_usd", "routing_latency_ms"}


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON with keys for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "location": f"{record.pathname}:{record.lineno}",
        }
        for key in _EXTRA_KEYS:
            obj[key] = getattr(record, key, None)
        if record.exc_info and record.exc_info[1]:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj)


def setup_logger(
    name: str = "routesmith",
    level: str | int = "INFO",
    json_format: bool = True,
) -> logging.Logger:
    """
    Set up a RouteSmith logger.

    Args:
        name: Logger name.
        level: Log level string or int.
        json_format: If True, use JSON. Otherwise plain text.

    Returns:
        Configured logger (singleton per name).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = (
        _LOG_LEVELS.get(str(level).upper(), logging.INFO)
        if isinstance(level, str)
        else int(level)
    )
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(
        JsonFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)

    return logger


class RouteSmithLogger:
    """Adapter that sets extra fields on log records before emitting."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, extra: dict[str, Any] | None = None) -> None:
        extra = extra or {}
        kwargs: dict[str, Any] = {"extra": {}}
        for key in _EXTRA_KEYS:
            if key in extra:
                kwargs["extra"][key] = extra[key]
        self._logger.log(level, msg, **kwargs)

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, extra)
