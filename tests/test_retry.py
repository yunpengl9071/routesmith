"""Real use case: retry with backoff handles transient failures."""
import time

import pytest

from routesmith.utils.retry import RetryExhaustedError, retry_with_backoff


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
        assert call_count[0] == 3
        assert "permanent" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_exponential_backoff_timing(self):
        call_count = [0]

        def fail():
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RuntimeError("fail")
            return "ok"

        t0 = time.monotonic()
        retry_with_backoff(fail, max_retries=3, base_delay=0.05, jitter=False)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.30

    def test_jitter_adds_variability(self):
        call_count = [0]

        def fail():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("fail")
            return "ok"

        t0 = time.monotonic()
        retry_with_backoff(fail, max_retries=2, base_delay=0.05, jitter=True)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.04

    def test_specific_exception_types_retried(self):
        call_count = [0]

        def fail_with_value_error():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("transient value error")
            return "ok"

        result = retry_with_backoff(
            fail_with_value_error, max_retries=2, base_delay=0.01,
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
                fail_unexpected, max_retries=3, base_delay=0.01,
                retryable=(ValueError, ConnectionError),
            )
        assert call_count[0] == 1

    def test_retries_on_any_exception_by_default(self):
        call_count = [0]

        def fail():
            call_count[0] += 1
            if call_count[0] < 2:
                raise KeyError("retried by default")
            return "ok"

        result = retry_with_backoff(fail, max_retries=2, base_delay=0.01)
        assert result == "ok"
        assert call_count[0] == 2
