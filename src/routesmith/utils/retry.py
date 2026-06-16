"""Retry with exponential backoff and jitter."""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""


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
                delay = base_delay * (2**attempt)
                if jitter:
                    delay += random.uniform(0, delay)
                time.sleep(delay)

    raise RetryExhaustedError(
        f"All {max_retries + 1} attempts failed. Last error: {last_exception}"
    ) from last_exception