"""Rolling-window budget enforcement."""
from __future__ import annotations

import threading
import time
from collections import deque

WINDOWS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}


class BudgetExceededError(RuntimeError):
    """Raised pre-flight when a spend window is exhausted."""

    def __init__(self, window: str, limit: float, spent: float) -> None:
        self.window = window
        self.limit = limit
        self.spent = spent
        super().__init__(
            f"Budget exceeded: spent ${spent:.4f} of ${limit:.4f} in the last {window}"
        )


class BudgetTracker:
    """Tracks spend in rolling minute/hour/day windows. Thread-safe."""

    def __init__(self, budget_config) -> None:
        self._config = budget_config
        self._events: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOWS["day"]
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def spent(self, window: str, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            cutoff = now - WINDOWS[window]
            return sum(c for ts, c in self._events if ts >= cutoff)

    def check(self, now: float | None = None) -> None:
        """Raise BudgetExceededError if any configured window is exhausted."""
        now = time.time() if now is None else now
        limits = {
            "minute": self._config.max_cost_per_minute,
            "hour": self._config.max_cost_per_hour,
            "day": self._config.max_cost_per_day,
        }
        for window, limit in limits.items():
            if limit is not None:
                s = self.spent(window, now)
                if s >= limit:
                    raise BudgetExceededError(window, limit, s)

    def record(self, cost: float, now: float | None = None) -> None:
        if cost <= 0:
            return
        now = time.time() if now is None else now
        with self._lock:
            self._events.append((now, cost))
            self._prune(now)
