"""Tests for BudgetTracker and rolling-window budget enforcement."""

import time
from unittest.mock import patch

import pytest

from routesmith.budget import BudgetExceededError, BudgetTracker
from routesmith.config import BudgetConfig
from tests.helpers import fake_response, make_rs


class TestBudgetTracker:
    """Tests for BudgetTracker core logic."""

    def test_tracker_spent_windows(self):
        """Record costs at different times, verify per-window spend."""
        tracker = BudgetTracker(BudgetConfig())
        tracker.record(0.5, now=0)
        tracker.record(0.3, now=100)
        tracker.record(0.2, now=4000)

        # At now=4010, only the 4000 event is within the last minute
        assert tracker.spent("minute", now=4010) == pytest.approx(0.2)

        # All three are within the last day
        assert tracker.spent("day", now=4010) == pytest.approx(1.0)

    def test_check_raises_when_day_exhausted(self):
        """Raises BudgetExceededError when daily limit is reached."""
        tracker = BudgetTracker(BudgetConfig(max_cost_per_day=1.0))
        tracker.record(1.0, now=0)

        with pytest.raises(BudgetExceededError) as excinfo:
            tracker.check(now=10)
        assert excinfo.value.window == "day"
        assert excinfo.value.limit == 1.0
        assert excinfo.value.spent == pytest.approx(1.0)

    def test_check_noop_when_unconfigured(self):
        """No error raised when all limits are None."""
        tracker = BudgetTracker(BudgetConfig())
        tracker.record(100.0, now=0)
        # Should not raise despite high spend (no limits configured)
        tracker.check(now=10)

    def test_check_passes_when_window_expired(self):
        """Spend outside the window doesn't trigger."""
        tracker = BudgetTracker(BudgetConfig(max_cost_per_day=1.0))
        tracker.record(1.0, now=0)
        # At now=86401, the record from t=0 is pruned (outside 1 day window)
        tracker.check(now=86401)


class TestBudgetWiredInClient:
    """Tests for budget enforcement through RouteSmith."""

    def test_completion_blocked_when_over_budget(self):
        """Rolling-window budget check blocks completion when exhausted."""
        budget = BudgetConfig(max_cost_per_day=1.0)
        rs = make_rs(budget=budget)

        now = time.time()
        rs._budget.record(1.0, now=now)

        with patch("routesmith.client.litellm.completion", return_value=fake_response()):
            with pytest.raises(BudgetExceededError) as excinfo:
                rs.completion(messages=[{"role": "user", "content": "Hi"}])
        assert "Budget exceeded" in str(excinfo.value)

    def test_completion_records_spend(self):
        """Successful completion records spend in tracker."""
        rs = make_rs()
        with patch("routesmith.client.litellm.completion", return_value=fake_response()):
            rs.completion(messages=[{"role": "user", "content": "Hi"}])
        assert rs._budget.spent("day") > 0

    def test_max_cost_per_request_filters_models(self):
        """max_cost_per_request filters out expensive models."""
        budget = BudgetConfig(max_cost_per_request=0.001)
        rs = make_rs(budget=budget)
        with patch("routesmith.client.litellm.completion", return_value=fake_response()):
            resp = rs.completion(messages=[{"role": "user", "content": "Hi"}])
        # gpt-4o (0.02/1k) should be filtered; gpt-4o-mini (0.00075/1k) should be selected
        assert resp.model == "gpt-4o-mini"
