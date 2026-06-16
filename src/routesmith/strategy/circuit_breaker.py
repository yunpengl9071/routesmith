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
        if (
            self.state == CircuitState.CLOSED
            and self.failure_count >= self.failure_threshold
        ):
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
