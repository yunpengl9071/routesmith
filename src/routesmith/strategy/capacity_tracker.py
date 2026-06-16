"""Track provisioned model capacity and utilization."""
from __future__ import annotations

import threading
import time


class CapacityTracker:
    """
    Track per-model request capacity for provisioned throughput.

    Maintains a rolling window of request counts to enforce
    requests-per-minute (RPM) limits. Thread-safe for concurrent
    completion calls.

    Attributes:
        max_rpm: Maximum requests per minute (0 = unlimited).
        window_seconds: Rolling window size in seconds.
    """

    def __init__(self, max_rpm: int = 0, window_seconds: float = 60.0) -> None:
        self.max_rpm = max_rpm
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._timestamps: list[float] = []
        self._total_requests = 0
        self._overflow_count = 0

    def record_request(self) -> None:
        """Record a request being routed through this capacity tracker."""
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            self._total_requests += 1
            self._prune(now)
            if self.max_rpm > 0 and len(self._timestamps) > self.max_rpm:
                self._overflow_count += 1

    def available(self) -> bool:
        """Check if capacity is available for a new request."""
        if self.max_rpm == 0:
            return True
        with self._lock:
            self._prune(time.monotonic())
            if len(self._timestamps) >= self.max_rpm:
                self._overflow_count += 1
                return False
            return True

    @property
    def current_utilization(self) -> float:
        """Current utilization as a fraction (0.0 - 1.0)."""
        if self.max_rpm == 0:
            return 0.0
        with self._lock:
            self._prune(time.monotonic())
            return min(len(self._timestamps) / self.max_rpm, 1.0)

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def overflow_count(self) -> int:
        return self._overflow_count

    def stats(self) -> dict:
        """Return capacity stats as a dictionary."""
        return {
            "max_rpm": self.max_rpm,
            "total_requests": self._total_requests,
            "overflow_count": self._overflow_count,
            "utilization": self.current_utilization,
            "current_window_count": len(self._timestamps),
        }

    def _prune(self, now: float) -> None:
        """Remove timestamps outside the rolling window."""
        cutoff = now - self.window_seconds
        self._timestamps = [ts for ts in self._timestamps if ts > cutoff]