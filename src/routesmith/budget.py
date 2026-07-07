"""Rolling-window budget enforcement."""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

WINDOWS = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}

# How long we wait between budget-poll cycles when queueing
_POLL_INTERVAL = 0.5
_MAX_QUEUE_WAIT = 3600.0


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

    def seconds_until_available(self, now: float | None = None) -> float:
        """Return seconds until the earliest budget window is available.

        Returns 0 if already within budget.
        """
        now = time.time() if now is None else now
        limits = {
            "minute": self._config.max_cost_per_minute,
            "hour": self._config.max_cost_per_hour,
            "day": self._config.max_cost_per_day,
        }
        max_wait = 0.0
        with self._lock:
            self._prune(now)
            for window, limit in limits.items():
                if limit is None:
                    continue
                spent = sum(c for ts, c in self._events if ts >= now - WINDOWS[window])
                if spent < limit:
                    continue
                # Find oldest event in this window — that's when we'll be under budget
                cutoff = now - WINDOWS[window]
                oldest_in_window = now
                for ts, _ in self._events:
                    if ts >= cutoff:
                        oldest_in_window = min(oldest_in_window, ts)
                wait = oldest_in_window + WINDOWS[window] - now + 0.1  # small buffer
                max_wait = max(max_wait, wait)
        return max_wait

    def wait_until_available(self, now: float | None = None) -> float:
        """Block until a budget slot opens. Returns total seconds waited."""
        now = time.time() if now is None else now
        deadline = now + _MAX_QUEUE_WAIT
        total_waited = 0.0
        while True:
            wait = self.seconds_until_available(now)
            if wait <= 0:
                return total_waited
            if time.time() + wait > deadline:
                raise BudgetExceededError(
                    "queued", _MAX_QUEUE_WAIT, total_waited,
                )
            time.sleep(min(_POLL_INTERVAL, wait))
            now = time.time()
            total_waited += _POLL_INTERVAL

    async def await_until_available(self, now: float | None = None) -> float:
        """Async wait until a budget slot opens. Returns total seconds waited."""
        now = time.time() if now is None else now
        deadline = now + _MAX_QUEUE_WAIT
        total_waited = 0.0
        while True:
            wait = self.seconds_until_available(now)
            if wait <= 0:
                return total_waited
            if time.time() + wait > deadline:
                raise BudgetExceededError(
                    "queued", _MAX_QUEUE_WAIT, total_waited,
                )
            await asyncio.sleep(min(_POLL_INTERVAL, wait))
            now = time.time()
            total_waited += _POLL_INTERVAL
