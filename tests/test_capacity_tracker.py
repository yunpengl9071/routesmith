"""Tests for CapacityTracker."""
from routesmith.strategy.capacity_tracker import CapacityTracker


class TestCapacityTracker:
    def test_initial_state(self):
        tracker = CapacityTracker(max_rpm=10)
        assert tracker.available()
        assert tracker.current_utilization == 0.0
        assert tracker.total_requests == 0
        assert tracker.overflow_count == 0

    def test_available_when_under_limit(self):
        tracker = CapacityTracker(max_rpm=10)
        for _ in range(9):
            tracker.record_request()
        assert tracker.available()
        assert tracker.total_requests == 9
        assert tracker.overflow_count == 0

    def test_unavailable_when_at_limit(self):
        tracker = CapacityTracker(max_rpm=10)
        for _ in range(10):
            tracker.record_request()
        assert not tracker.available()
        assert tracker.total_requests == 10

    def test_overflow_count_increments(self):
        tracker = CapacityTracker(max_rpm=10)
        for _ in range(15):
            tracker.record_request()
        assert tracker.total_requests == 15
        assert tracker.overflow_count == 5

    def test_utilization_percentage(self):
        tracker = CapacityTracker(max_rpm=10)
        for _ in range(7):
            tracker.record_request()
        assert tracker.current_utilization == 0.7

    def test_max_rpm_zero_means_unlimited(self):
        tracker = CapacityTracker(max_rpm=0)
        for _ in range(1000):
            tracker.record_request()
        assert tracker.available()
        assert tracker.total_requests == 1000

    def test_stats_dict(self):
        tracker = CapacityTracker(max_rpm=10)
        for _ in range(3):
            tracker.record_request()
        stats = tracker.stats()
        assert stats["total_requests"] == 3
        assert stats["overflow_count"] == 0
        assert stats["utilization"] == 0.3
        assert "max_rpm" in stats

    def test_thread_safety_basic(self):
        """Basic concurrency test — records should not be lost."""
        import threading

        tracker = CapacityTracker(max_rpm=1000)
        threads = []
        for _ in range(10):
            t = threading.Thread(target=lambda: [tracker.record_request() for _ in range(100)])
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert tracker.total_requests == 1000

    def test_mark_overflow_increments_count(self):
        """mark_overflow() explicitly records overflow events."""
        tracker = CapacityTracker(max_rpm=5)
        assert tracker.overflow_count == 0
        tracker.mark_overflow()
        assert tracker.overflow_count == 1
        tracker.mark_overflow()
        tracker.mark_overflow()
        assert tracker.overflow_count == 3

    def test_available_does_not_side_effect_overflow(self):
        """available() is a pure query — no overflow side effects."""
        tracker = CapacityTracker(max_rpm=3)
        for _ in range(3):
            tracker.record_request()
        assert not tracker.available()
        assert tracker.overflow_count == 0  # Not incremented
        tracker.available()
        tracker.available()
        assert tracker.overflow_count == 0  # Still not incremented
