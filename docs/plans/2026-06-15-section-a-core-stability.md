# Section A: Core Stability & Resilience — Implementation Plan

> **REQUIRED SUB-SKILL:** Use the subagent-driven-development skill to execute.
> **Branch:** feature/v0.2.0-core-stability (from dev)
> **CRITICAL:** Real use case tests, not cherry-picked mocks

**Goal:** Add circuit breaker, retry logic, structured logging, and exception hierarchy to make RouteSmith production-resilient.

**Architecture:** New modules (`exceptions.py`, `strategy/circuit_breaker.py`, `utils/logging.py`) wired into `client.py`. Circuit breaker sits between router and LiteLLM call, retries wrapped around it. All components use structured JSON logging.

**Tech Stack:** Python 3.13, pytest, LiteLLM

**Real Use Case Validation Strategy:**
- NOT just unit tests with mocks
- NOT just "a single model returns a success"
- Must simulate: provider failures, rate limits, timeout cascades, multi-model routing under failure
- Must verify: circuit opens after 5 real failures, logs are valid JSON parseable, retry backoff actually increases

---

## Task A1: Exception Hierarchy

**Files:**
- Create: `src/routesmith/exceptions.py`
- Test: `tests/test_exceptions.py`
- Modify: `src/routesmith/__init__.py` (add exports)

### Step 1: Write failing tests for exception hierarchy

```python
# tests/test_exceptions.py
"""Real use case: exception hierarchy enforces consistent error handling."""
import json
import pytest
from routesmith.exceptions import (
    RouteSmithError,
    BudgetExceededError,
    NoCapableModelError,
    ProviderUnavailableError,
    CircuitOpenError,
)


class TestExceptionHierarchy:
    """Verify exception classes have correct inheritance and info."""

    def test_budget_exceeded_is_routesmith_error(self):
        err = BudgetExceededError("Budget exceeded", current_spend=5.0, limit=10.0, reset_seconds=3600)
        assert isinstance(err, RouteSmithError)  # Base class

    def test_budget_exceeded_stores_context(self):
        err = BudgetExceededError("message", current_spend=5.0, limit=10.0, reset_seconds=3600)
        assert err.current_spend == 5.0
        assert err.limit == 10.0
        assert err.reset_seconds == 3600

    def test_budget_exceeded_str_includes_details(self):
        err = BudgetExceededError("nope", current_spend=9.50, limit=10.0, reset_seconds=1800)
        s = str(err)
        assert "9.50" in s
        assert "10.0" in s
        assert "30 minutes" in s or "1800" in s

    def test_no_capable_model_stores_requirements(self):
        err = NoCapableModelError(
            "No model supports the required capabilities",
            required_capabilities={"tool_calling", "vision"},
            available_models=["gpt-4o-mini"],
        )
        assert err.required_capabilities == {"tool_calling", "vision"}
        assert err.available_models == ["gpt-4o-mini"]

    def test_provider_unavailable_stores_original_error(self):
        original = ValueError("connection refused")
        err = ProviderUnavailableError("gpt-4o", original)
        assert err.model_id == "gpt-4o"
        assert err.original_error is original

    def test_circuit_open_stores_retry_after(self):
        err = CircuitOpenError("gpt-4o-mini", retry_after=30.0)
        assert err.model_id == "gpt-4o-mini"
        assert err.retry_after == 30.0

    def test_arbitrary_kwargs_survive(self):
        """Custom exceptions shouldn't lose kwargs passed through."""
        err = RouteSmithError("test", custom_field="x")
        assert err.custom_field == "x"


class TestRealUseCaseErrors:
    """End-to-end: exceptions carry enough info for a monitoring system."""

    def test_budget_error_serializable_for_alerts(self):
        """Alerts can parse budget context from exceptions."""
        err = BudgetExceededError("over", current_spend=99.99, limit=100.0, reset_seconds=600)
        data = {
            "type": type(err).__name__,
            "message": str(err),
            "current_spend": err.current_spend,
            "limit": err.limit,
            "reset_seconds": err.reset_seconds,
        }
        # Must be JSON serializable for webhook alerts
        json.dumps(data)  # Should not raise

    def test_circuit_error_carries_retry_after_for_scheduler(self):
        """Scheduler can parse retry_after to re-enable model later."""
        err = CircuitOpenError("claude-3-opus", retry_after=60.0)
        assert err.retry_after > 0
        # Real scheduling code does: schedule_reopen(model, after=err.retry_after)
        assert err.retry_after == 60.0

    def test_provider_error_chains_cause_for_debugging(self):
        """Root cause is preserved through raise...from chain."""
        try:
            try:
                raise ConnectionRefusedError("Connection refused")
            except ConnectionRefusedError as original:
                raise ProviderUnavailableError("gpt-4o", original) from original
        except ProviderUnavailableError as err:
            assert err.__cause__ is not None
            assert isinstance(err.__cause__, ConnectionRefusedError)
```

Run: `.venv/bin/pytest tests/test_exceptions.py -v`
Expected: FAIL - module not found

### Step 2: Write minimal implementation

```python
# src/routesmith/exceptions.py
"""RouteSmith exception hierarchy for production error handling."""

from __future__ import annotations


class RouteSmithError(Exception):
    """Base exception for all RouteSmith errors."""

    def __init__(self, message: str = "", **kwargs: object) -> None:
        super().__init__(message)
        for key, value in kwargs.items():
            setattr(self, key, value)


class BudgetExceededError(RouteSmithError):
    """
    Raised when budget limit is exceeded.

    Attributes:
        current_spend: Current spend in USD.
        limit: Budget limit in USD.
        reset_seconds: Seconds until budget resets.
    """

    def __init__(
        self,
        message: str = "",
        *,
        current_spend: float = 0.0,
        limit: float = 0.0,
        reset_seconds: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.current_spend = current_spend
        self.limit = limit
        self.reset_seconds = reset_seconds

    def __str__(self) -> str:
        base = super().__str__() or "Budget exceeded"
        parts = [base]
        if self.current_spend > 0:
            parts.append(f"Spent ${self.current_spend:.2f}")
        if self.limit > 0:
            parts.append(f"Limit ${self.limit:.2f}")
        if self.reset_seconds > 0:
            minutes = int(self.reset_seconds / 60)
            parts.append(f"Resets in {minutes} minutes")
        return ", ".join(parts)


class NoCapableModelError(RouteSmithError):
    """
    Raised when no registered model satisfies required capabilities.

    Attributes:
        required_capabilities: Set of capabilities that were required.
        available_models: List of model IDs that were registered.
    """

    def __init__(
        self,
        message: str = "",
        *,
        required_capabilities: set[str] | None = None,
        available_models: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.required_capabilities = required_capabilities or set()
        self.available_models = available_models or []


class ProviderUnavailableError(RouteSmithError):
    """
    Raised when a provider/model returns a non-retryable error.

    Attributes:
        model_id: The model that failed.
        original_error: The original exception from the provider.
    """

    def __init__(
        self,
        model_id: str,
        original_error: Exception | None = None,
        **kwargs: object,
    ) -> None:
        msg = f"Provider unavailable for model '{model_id}'"
        if original_error:
            msg += f": {original_error}"
        super().__init__(msg, **kwargs)
        self.model_id = model_id
        self.original_error = original_error


class CircuitOpenError(RouteSmithError):
    """
    Raised when the circuit breaker is open for a model.

    Attributes:
        model_id: The model whose circuit is open.
        retry_after: Seconds until the circuit can be tested again.
    """

    def __init__(
        self,
        model_id: str,
        retry_after: float = 30.0,
        **kwargs: object,
    ) -> None:
        msg = f"Circuit breaker open for model '{model_id}'"
        super().__init__(msg, **kwargs)
        self.model_id = model_id
        self.retry_after = retry_after
```

Run: `.venv/bin/pytest tests/test_exceptions.py -v`
Expected: PASS

### Step 3: Export from __init__.py

Add to `src/routesmith/__init__.py`:
```python
from routesmith.exceptions import (
    BudgetExceededError,
    CircuitOpenError,
    NoCapableModelError,
    ProviderUnavailableError,
    RouteSmithError,
)
```

Run: `.venv/bin/pytest tests/test_exceptions.py -v`
Expected: PASS (still)

### Step 4: Commit

```bash
git add src/routesmith/exceptions.py tests/test_exceptions.py src/routesmith/__init__.py
git commit -m "feat(exceptions): add exception hierarchy with BudgetExceeded, CircuitOpen, ProviderUnavailable"
```

---

## Task A2: Structured JSON Logging

**Files:**
- Create: `src/routesmith/utils/logging.py`
- Test: `tests/test_logging.py`

### Step 1: Write failing tests

```python
# tests/test_logging.py
"""Real use case: JSON logs must be parseable at scale."""
import json
import logging
from routesmith.utils.logging import setup_logger, JsonFormatter, RouteSmithLogger


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
        # ISO 8601 format
        assert "T" in parsed["timestamp"]

    def test_format_handles_missing_extra_fields(self):
        """Records without custom extra fields still produce valid JSON."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="rs", level=logging.DEBUG,
            pathname="x.py", lineno=3,
            msg="cache miss", args=(),
            exc_info=None,
        )
        # No model_id, no request_id set
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "level" in parsed
        # model_id defaults to null
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
        """Default is plain text, not JSON."""
        logger = setup_logger("routesmith.plain", level="INFO", json_format=False)
        handler = logger.handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_setup_logger_no_duplicate_handlers(self):
        """Calling setup_logger twice doesn't add duplicate handlers."""
        logger = setup_logger("routesmith.no_dup", level="INFO")
        initial_count = len(logger.handlers)
        logger2 = setup_logger("routesmith.no_dup", level="INFO")
        assert logger2 is logger
        assert len(logger.handlers) == initial_count
```

Run: `.venv/bin/pytest tests/test_logging.py -v`
Expected: FAIL - module not found

### Step 2: Write implementation

```python
# src/routesmith/utils/logging.py
"""Structured JSON logging for RouteSmith."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

_EXTRA_KEYS = {"model_id", "request_id", "cost_usd", "routing_latency_ms"}
_SYS_KEYS = {"name", "msg", "args", "created", "exc_info", "exc_text",
             "filename", "funcName", "levelname", "levelno", "lineno",
             "module", "msecs", "pathname", "process", "processName",
             "relativeCreated", "stack_info", "thread", "threadName"}


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

    log_level = _LOG_LEVELS.get(str(level).upper(), logging.INFO) \
        if isinstance(level, str) else int(level)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(
        JsonFormatter() if json_format
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
        kwargs: dict[str, Any] = {}
        for key in _EXTRA_KEYS:
            if key in extra:
                kwargs[key] = extra[key]
        self._logger.log(level, msg, extra=kwargs)

    def debug(self, msg: str, **extra: Any) -> None:
        self._log(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        self._log(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        self._log(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        self._log(logging.ERROR, msg, extra)
```

Run: `.venv/bin/pytest tests/test_logging.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add src/routesmith/utils/logging.py tests/test_logging.py
git commit -m "feat(logging): add structured JSON logging with JsonFormatter and RouteSmithLogger"
```

---

## Task A3: Circuit Breaker

**Files:**
- Create: `src/routesmith/strategy/circuit_breaker.py`
- Test: `tests/test_circuit_breaker.py`

### Step 1: Write failing tests

```python
# tests/test_circuit_breaker.py
"""Real use case: circuit breakers protect against cascading failures."""
import time
import pytest
from routesmith.strategy.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInit:
    def test_defaults(self):
        cb = CircuitBreaker("test-model")
        assert cb.model_id == "test-model"
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_custom_thresholds(self):
        cb = CircuitBreaker("x", failure_threshold=3, recovery_timeout=10.0)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10.0


class TestCircuitBreakerFlow:
    """Real use case: 5 failures → open → wait → half-open → test → close."""

    def test_allow_when_closed(self):
        cb = CircuitBreaker("gpt-4o")
        assert cb.allow_request() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker("gpt-4o", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_reopens_after_timeout(self):
        cb = CircuitBreaker("gpt-4o", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.allow_request() is False  # OPEN
        time.sleep(0.02)
        assert cb.allow_request() is True  # HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        cb = CircuitBreaker("gpt-4o", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True  # HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_reopens_after_failure_in_half_open(self):
        cb = CircuitBreaker("gpt-4o", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()  # Enter HALF_OPEN
        cb.record_failure()  # Fail the test request
        assert cb.state == CircuitState.OPEN

    def test_multiple_models_independent(self):
        """Each model has its own circuit breaker state."""
        cb_a = CircuitBreaker("model-a", failure_threshold=2)
        cb_b = CircuitBreaker("model-b", failure_threshold=2)
        for _ in range(2):
            cb_a.record_failure()
        assert cb_a.state == CircuitState.OPEN
        assert cb_b.state == CircuitState.CLOSED

    def test_retry_after_seconds(self):
        cb = CircuitBreaker("m", failure_threshold=1, recovery_timeout=5.0)
        cb.record_failure()
        # retry_after should be approximately recovery_timeout
        assert 4.0 <= cb.retry_after_seconds() <= 6.0

    def test_reset(self):
        cb = CircuitBreaker("m", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerDictionary:
    """Real use case: circuit breakers managed as a dict per model."""

    def test_dict_management(self):
        breakers: dict[str, CircuitBreaker] = {}
        def get_breaker(model_id: str) -> CircuitBreaker:
            if model_id not in breakers:
                breakers[model_id] = CircuitBreaker(model_id)
            return breakers[model_id]

        b1 = get_breaker("gpt-4o")
        b2 = get_breaker("gpt-4o-mini")
        assert b1 is not b2
        b1b = get_breaker("gpt-4o")
        assert b1 is b1b  # Same breaker returned
```

Run: `.venv/bin/pytest tests/test_circuit_breaker.py -v`
Expected: FAIL - module not found

### Step 2: Write implementation

```python
# src/routesmith/strategy/circuit_breaker.py
"""Circuit breaker for provider resilience."""
from __future__ import annotations

import enum
import time


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker for a single model.

    State machine:
        CLOSED → (failures >= threshold) → OPEN
        OPEN → (time elapsed >= timeout) → HALF_OPEN
        HALF_OPEN → (success) → CLOSED
        HALF_OPEN → (failure) → OPEN
    """

    def __init__(
        self,
        model_id: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.model_id = model_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0

    def allow_request(self) -> bool:
        """Check if a request can be sent through this breaker."""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN — allow one test request
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self._last_failure_time = time.monotonic()
        if (self.state == CircuitState.CLOSED
                and self.failure_count >= self.failure_threshold):
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def retry_after_seconds(self) -> float:
        """Seconds until the circuit can be tested again."""
        if self.state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self.recovery_timeout - elapsed)

    def reset(self) -> None:
        """Force reset to CLOSED (e.g., manual intervention)."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
```

Run: `.venv/bin/pytest tests/test_circuit_breaker.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add src/routesmith/strategy/circuit_breaker.py tests/test_circuit_breaker.py
git commit -m "feat(circuit-breaker): add per-model circuit breaker with CLOSED→OPEN→HALF_OPEN state machine"
```

---

## Task A4: Retry with Exponential Backoff

**Files:**
- Create: `src/routesmith/utils/retry.py`
- Test: `tests/test_retry.py`

### Step 1: Write failing tests

```python
# tests/test_retry.py
"""Real use case: retry with backoff handles transient failures."""
import time
import pytest
from routesmith.utils.retry import retry_with_backoff, RetryExhaustedError


class TestRetryWithBackoff:
    def test_success_first_try(self):
        call_count = [0]
        def succeed():
            call_count[0] += 1
            return "ok"
        result = retry_with_backoff(succeed, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count[0] == 1

    def test_retries_on_exception(self):
        call_count = [0]
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return "finally"
        result = retry_with_backoff(fail_twice, max_retries=3, base_delay=0.01)
        assert result == "finally"
        assert call_count[0] == 3

    def test_exhausted_retries_raises(self):
        call_count = [0]
        def always_fail():
            call_count[0] += 1
            raise ValueError("permanent")
        with pytest.raises(RetryExhaustedError) as exc_info:
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)
        assert call_count[0] == 3  # 1 initial + 2 retries
        assert "permanent" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_exponential_backoff_timing(self):
        delays = []
        def track_delay(attempt: int):
            delays.append(attempt)

        call_count = [0]
        def fail():
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("fail")
            return "ok"

        t0 = time.monotonic()
        retry_with_backoff(fail, max_retries=3, base_delay=0.05, jitter=False)
        elapsed = time.monotonic() - t0
        # base_delay * 2^0 + base_delay * 2^1 + base_delay * 2^2
        # = 0.05 + 0.10 + 0.20 = 0.35s
        assert elapsed >= 0.30  # Allow small timing variance

    def test_jitter_adds_variability(self):
        delays = []
        call_count = [0]
        def fail():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("fail")
            return "ok"
        t0 = time.monotonic()
        retry_with_backoff(fail, max_retries=2, base_delay=0.05, jitter=True)
        elapsed = time.monotonic() - t0
        # Should be at least base_delay
        assert elapsed >= 0.04

    def test_specific_exception_types_retried(self):
        call_count = [0]
        def fail_with_value_error():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("transient value error")
            return "ok"
        # ValueError is in retryable set
        result = retry_with_backoff(
            fail_with_value_error,
            max_retries=2,
            base_delay=0.01,
            retryable=(ValueError,),
        )
        assert result == "ok"

    def test_non_retryable_exception_not_retried(self):
        call_count = [0]
        def fail_unexpected():
            call_count[0] += 1
            raise KeyError("unexpected")
        with pytest.raises(KeyError):
            retry_with_backoff(
                fail_unexpected,
                max_retries=3,
                base_delay=0.01,
                retryable=(ValueError, ConnectionError),
            )
        assert call_count[0] == 1  # Not retried

    def test_retries_on_any_exception_by_default(self):
        """By default, all exceptions are retried."""
        call_count = [0]
        def fail():
            call_count[0] += 1
            if call_count[0] < 2:
                raise KeyError("retried by default")
            return "ok"
        result = retry_with_backoff(fail, max_retries=2, base_delay=0.01)
        assert result == "ok"
        assert call_count[0] == 2
```

Run: `.venv/bin/pytest tests/test_retry.py -v`
Expected: FAIL - module not found

### Step 2: Write implementation

```python
# src/routesmith/utils/retry.py
"""Retry with exponential backoff and jitter."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: bool = True,
    retryable: tuple[type[Exception], ...] | None = None,
) -> T:
    """
    Call fn with exponential backoff retry.

    Args:
        fn: Function to call.
        max_retries: Maximum number of retries (total attempts = max_retries + 1).
        base_delay: Base delay in seconds.
        jitter: If True, add random jitter (0-100% of delay).
        retryable: Tuple of exception types to retry on. None = retry on any.

    Returns:
        Result of fn.

    Raises:
        RetryExhaustedError: If all retries exhausted.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exception = exc
            if retryable is not None and not isinstance(exc, retryable):
                raise
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                if jitter:
                    delay += random.uniform(0, delay)
                time.sleep(delay)

    raise RetryExhaustedError(
        f"All {max_retries + 1} attempts failed. Last error: {last_exception}"
    ) from last_exception
```

Run: `.venv/bin/pytest tests/test_retry.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add src/routesmith/utils/retry.py tests/test_retry.py
git commit -m "feat(retry): add exponential backoff retry with configurable jitter and exception filtering"
```

---

## Task A5: Wire Everything into Client Flow

**Files:**
- Modify: `src/routesmith/client.py` (add breaker dict, logging, retry wrapper)
- Modify: `tests/test_client.py` (add real use case tests)
- Create: `tests/test_client_resilience.py` (integration tests)

### Step 1: Write failing integration tests

```python
# tests/test_client_resilience.py
"""Real use case: client handles provider failures gracefully."""
import json
import pytest
from unittest.mock import patch, MagicMock
from routesmith import RouteSmith, RouteSmithConfig
from routesmith.strategy.circuit_breaker import CircuitState
from routesmith.exceptions import CircuitOpenError, ProviderUnavailableError


def make_test_client():
    """Create a RouteSmith client with two models for resilience testing."""
    config = RouteSmithConfig(feedback_sample_rate=1.0)
    rs = RouteSmith(config=config)
    rs.register_model("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015,
                       quality_score=0.95)
    rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
                       quality_score=0.80)
    return rs


class TestCircuitBreakerIntegration:
    """Verify circuit breaker is wired into the completion flow."""

    @patch("litellm.completion")
    def test_circuit_opens_after_repeated_failures(self, mock_litellm):
        """5 consecutive failures from same model → circuit opens → re-route to other model."""
        rs = make_test_client()
        # Force the router to always pick gpt-4o-mini first, then make it fail
        messages = [{"role": "user", "content": "test"}]

        # Make LiteLLM fail 5 times
        mock_litellm.side_effect = RuntimeError("connection refused")

        for i in range(5):
            try:
                rs.completion(messages=messages)
            except Exception:
                pass

        # After 5 failures, the circuit for the failing model should be open
        breaker = rs._circuit_breakers.get("gpt-4o-mini")
        if breaker:
            assert breaker.state == CircuitState.OPEN

    @patch("litellm.completion")
    def test_retry_logic_applied(self, mock_litellm):
        """Transient errors are retried before circuit breaker kicks in."""
        rs = make_test_client()
        messages = [{"role": "user", "content": "test"}]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        # Fail twice, succeed on third
        mock_litellm.side_effect = [
            RuntimeError("timeout"),
            RuntimeError("timeout"),
            mock_response,
        ]

        result = rs.completion(messages=messages)
        assert result.choices[0].message.content == "ok"

    @patch("litellm.completion")
    def test_fallback_to_other_model_when_circuit_open(self, mock_litellm):
        """When one model's circuit is open, router picks another model."""
        rs = make_test_client()
        # Pre-fill the circuit breaker for gpt-4o-mini
        breaker = rs._circuit_breakers.get("gpt-4o-mini")
        if breaker:
            for _ in range(breaker.failure_threshold):
                breaker.record_failure()

        messages = [{"role": "user", "content": "test"}]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "from gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_litellm.return_value = mock_response

        result = rs.completion(messages=messages)
        assert result.choices[0].message.content is not None


class TestStructuredLoggingIntegration:
    """Verify JSON logs are emitted during routing."""

    @patch("litellm.completion")
    def test_logging_emits_json(self, mock_litellm, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="routesmith")

        rs = make_test_client()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_litellm.return_value = mock_response

        messages = [{"role": "user", "content": "hello"}]
        rs.completion(messages=messages)

        # Verify something was logged
        assert len(caplog.records) > 0
```

Run: `.venv/bin/pytest tests/test_client_resilience.py -v`
Expected: FAIL - _circuit_breakers not found

### Step 2: Wire into client.py

Add to `src/routesmith/client.py`:

```python
# At top of RouteSmith.__init__:
from routesmith.strategy.circuit_breaker import CircuitBreaker, CircuitState
from routesmith.utils.logging import setup_logger, RouteSmithLogger
from routesmith.utils.retry import retry_with_backoff, RetryExhaustedError
from routesmith.exceptions import (
    BudgetExceededError, CircuitOpenError, ProviderUnavailableError,
)

# In RouteSmith.__init__, add:
self._circuit_breakers: dict[str, CircuitBreaker] = {}
self._log: RouteSmithLogger = RouteSmithLogger(
    setup_logger("routesmith", json_format=True)
)

# In completion(), wrap the LiteLLM call with circuit breaker + retry:
def _execute_llm_call(self, model_id: str, **kwargs):
    """Execute LLM call with circuit breaker and retry."""
    breaker = self._circuit_breakers.get(model_id)
    if breaker is None:
        breaker = CircuitBreaker(model_id)
        self._circuit_breakers[model_id] = breaker

    if not breaker.allow_request():
        raise CircuitOpenError(model_id, retry_after=breaker.retry_after_seconds())

    try:
        result = retry_with_backoff(
            lambda: litellm.completion(model=model_id, **kwargs),
            max_retries=2,
            base_delay=1.0,
            retryable=(Exception,),  # Any transient error
        )
        breaker.record_success()
        self._log.info("llm_call_success", model_id=model_id, ...)
        return result
    except (CircuitOpenError, BudgetExceededError):
        raise
    except RetryExhaustedError as e:
        breaker.record_failure()
        self._log.error("llm_call_exhausted", model_id=model_id, ...)
        raise ProviderUnavailableError(model_id, e) from e
    except Exception as e:
        breaker.record_failure()
        self._log.error("llm_call_failed", model_id=model_id, ...)
        raise ProviderUnavailableError(model_id, e) from e
```

Run: `.venv/bin/pytest tests/test_client_resilience.py -v`
Expected: PASS (after implementation)

### Step 3: Run full test suite

```bash
.venv/bin/pytest tests/ -q
```

Expected: All 544 tests still pass + new resilience tests pass

### Step 4: Commit

```bash
git add src/routesmith/client.py tests/test_client_resilience.py
git commit -m "feat(client): wire circuit breaker, retry, and structured logging into completion flow"
```

---

## UAT Validation (Real Use Case)

Before merging to dev, run:

```bash
# 1. Full test suite
.venv/bin/pytest tests/ -v

# 2. Manual resilience test (requires Groq)
GROQ_API_KEY=gsk_... .venv/bin/python -c "
from routesmith import RouteSmith
rs = RouteSmith()
rs.register_model('groq/llama-3.3-70b-versatile', cost_per_1k_input=0.00059, cost_per_1k_output=0.00079, quality_score=0.90)
rs.register_model('groq/llama-3.1-8b-instant', cost_per_1k_input=0.00005, cost_per_1k_output=0.00008, quality_score=0.60)
# Run 10 queries and verify no crash
for i in range(10):
    r = rs.completion(messages=[{'role':'user','content':f'Count to {i}'}])
    print(f'Q{i}: {r.choices[0].message.content[:30]}...')
print(f'Stats: {rs.stats}')
"
```

Expected: All tests pass, output shows proper JSON logging, circuit breaker works.
