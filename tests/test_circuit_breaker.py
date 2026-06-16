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
        assert cb.allow_request() is False
        time.sleep(0.02)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        cb = CircuitBreaker("gpt-4o", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_reopens_after_failure_in_half_open(self):
        cb = CircuitBreaker("gpt-4o", failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_multiple_models_independent(self):
        cb_a = CircuitBreaker("model-a", failure_threshold=2)
        cb_b = CircuitBreaker("model-b", failure_threshold=2)
        for _ in range(2):
            cb_a.record_failure()
        assert cb_a.state == CircuitState.OPEN
        assert cb_b.state == CircuitState.CLOSED

    def test_retry_after_seconds(self):
        cb = CircuitBreaker("m", failure_threshold=1, recovery_timeout=5.0)
        cb.record_failure()
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
        assert b1 is b1b