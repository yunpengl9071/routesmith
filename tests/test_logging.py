"""Real use case: JSON logs must be parseable at scale."""
import json
import logging

from routesmith.utils.logging import JsonFormatter, RouteSmithLogger, setup_logger


class TestJsonFormatter:
    """JSON logs must be valid JSON with required fields."""

    def test_format_produces_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="routesmith", level=logging.INFO,
            pathname="test.py", lineno=1,
            msg="routing decision", args=(),
            exc_info=None,
        )
        record.model_id = "gpt-4o-mini"
        record.request_id = "abc123def4567890"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "routesmith"
        assert parsed["model_id"] == "gpt-4o-mini"
        assert parsed["request_id"] == "abc123def4567890"

    def test_format_includes_timestamp(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="rs", level=logging.WARN,
            pathname="x.py", lineno=2,
            msg="test", args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert "T" in parsed["timestamp"]

    def test_format_handles_missing_extra_fields(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="rs", level=logging.DEBUG,
            pathname="x.py", lineno=3,
            msg="cache miss", args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "level" in parsed
        assert parsed.get("model_id") is None

    def test_format_includes_exception_info(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="rs", level=logging.ERROR,
                pathname="x.py", lineno=4,
                msg="failed", args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "test error" in parsed["exception"]

    def test_cost_fields_serialized_when_present(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="rs", level=logging.INFO,
            pathname="x.py", lineno=5,
            msg="cost tracked", args=(),
            exc_info=None,
        )
        record.cost_usd = 0.0023
        record.routing_latency_ms = 2.1
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["cost_usd"] == 0.0023
        assert parsed["routing_latency_ms"] == 2.1


class TestSetupLogger:
    """Real use case: logger setup works in production and test."""

    def test_setup_logger_returns_logger(self):
        logger = setup_logger("routesmith.test", level="INFO")
        assert logger.name == "routesmith.test"
        assert logger.level == logging.INFO

    def test_setup_logger_json_format(self):
        logger = setup_logger("routesmith.json_test", level="INFO", json_format=True)
        assert len(logger.handlers) >= 1
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_setup_logger_plain_format(self):
        logger = setup_logger("routesmith.plain", level="INFO", json_format=False)
        handler = logger.handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_setup_logger_no_duplicate_handlers(self):
        logger = setup_logger("routesmith.no_dup", level="INFO")
        initial_count = len(logger.handlers)
        logger2 = setup_logger("routesmith.no_dup", level="INFO")
        assert logger2 is logger
        assert len(logger.handlers) == initial_count


class TestRouteSmithLogger:
    """RouteSmithLogger wraps standard logger with extra field support."""

    def test_info_logs_message(self, caplog):
        base = setup_logger("rs.adapter_test", level="INFO", json_format=False)
        log = RouteSmithLogger(base)
        caplog.set_level(logging.INFO)
        log.info("test message", model_id="gpt-4o", request_id="abc")
        assert "test message" in caplog.text

    def test_error_logs_with_extra(self, caplog):
        base = setup_logger("rs.err_test", level="ERROR", json_format=False)
        log = RouteSmithLogger(base)
        caplog.set_level(logging.ERROR)
        log.error("something broke", model_id="claude", cost_usd=0.05)
        assert "something broke" in caplog.text
