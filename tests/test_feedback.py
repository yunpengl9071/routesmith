"""Tests for feedback collection system: signals, storage, and integration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from routesmith import RouteSmith, RouteSmithConfig
from routesmith.feedback.collector import FeedbackCollector, FeedbackRecord
from routesmith.feedback.signals import QualitySignal, SignalExtractor
from routesmith.feedback.storage import FeedbackStorage
from routesmith.registry.models import ModelRegistry


# ---------------------------------------------------------------------------
# QualitySignal dataclass
# ---------------------------------------------------------------------------

class TestQualitySignal:
    def test_creation(self):
        sig = QualitySignal(
            signal_type="implicit",
            signal_name="error_detected",
            signal_value=1.0,
            raw_value="stop",
        )
        assert sig.signal_type == "implicit"
        assert sig.signal_name == "error_detected"
        assert sig.signal_value == 1.0
        assert sig.raw_value == "stop"
        assert isinstance(sig.timestamp, float)


# ---------------------------------------------------------------------------
# SignalExtractor
# ---------------------------------------------------------------------------

def _make_response(
    content: str = "Hello!",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> MagicMock:
    """Build a mock LLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


class TestSignalExtractor:
    def test_extract_returns_six_signals(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(), "model-a", 100.0)
        assert len(signals) == 6
        names = {s.signal_name for s in signals}
        assert names == {
            "error_detected",
            "refusal_detected",
            "empty_response",
            "truncated_response",
            "response_length_anomaly",
            "latency_anomaly",
        }

    def test_error_detected(self):
        ext = SignalExtractor()
        signals = ext.extract(
            _make_response(finish_reason="error"), "m", 100.0
        )
        error_sig = next(s for s in signals if s.signal_name == "error_detected")
        assert error_sig.signal_value == 0.0

    def test_no_error(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(), "m", 100.0)
        error_sig = next(s for s in signals if s.signal_name == "error_detected")
        assert error_sig.signal_value == 1.0

    def test_refusal_detected(self):
        ext = SignalExtractor()
        resp = _make_response(content="I'm sorry, I cannot help with that request.")
        signals = ext.extract(resp, "m", 100.0)
        refusal_sig = next(s for s in signals if s.signal_name == "refusal_detected")
        assert refusal_sig.signal_value == 0.0

    def test_no_refusal(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(content="Sure, here's the answer."), "m", 100.0)
        refusal_sig = next(s for s in signals if s.signal_name == "refusal_detected")
        assert refusal_sig.signal_value == 1.0

    def test_empty_response(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(content=""), "m", 100.0)
        empty_sig = next(s for s in signals if s.signal_name == "empty_response")
        assert empty_sig.signal_value == 0.0

    def test_non_empty_response(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(content="hello"), "m", 100.0)
        empty_sig = next(s for s in signals if s.signal_name == "empty_response")
        assert empty_sig.signal_value == 1.0

    def test_truncated_response(self):
        ext = SignalExtractor()
        signals = ext.extract(
            _make_response(finish_reason="length"), "m", 100.0
        )
        trunc_sig = next(s for s in signals if s.signal_name == "truncated_response")
        assert trunc_sig.signal_value == 0.0

    def test_not_truncated(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(), "m", 100.0)
        trunc_sig = next(s for s in signals if s.signal_name == "truncated_response")
        assert trunc_sig.signal_value == 1.0

    def test_latency_anomaly_no_baseline(self):
        ext = SignalExtractor()
        signals = ext.extract(_make_response(), "m", 5000.0)
        lat_sig = next(s for s in signals if s.signal_name == "latency_anomaly")
        # No baseline → no anomaly
        assert lat_sig.signal_value == 1.0

    def test_latency_anomaly_over_p95(self):
        ext = SignalExtractor(model_latency_p95={"m": 1000.0})
        signals = ext.extract(_make_response(), "m", 2000.0)
        lat_sig = next(s for s in signals if s.signal_name == "latency_anomaly")
        assert lat_sig.signal_value < 1.0

    def test_latency_ok_under_p95(self):
        ext = SignalExtractor(model_latency_p95={"m": 2000.0})
        signals = ext.extract(_make_response(), "m", 500.0)
        lat_sig = next(s for s in signals if s.signal_name == "latency_anomaly")
        assert lat_sig.signal_value == 1.0

    def test_length_anomaly_needs_history(self):
        ext = SignalExtractor()
        # First few calls: not enough data
        for _ in range(4):
            signals = ext.extract(_make_response(completion_tokens=20), "m", 100.0)
        len_sig = next(s for s in signals if s.signal_name == "response_length_anomaly")
        assert len_sig.signal_value == 1.0  # Not enough data yet

    def test_length_anomaly_detected(self):
        ext = SignalExtractor()
        # Build up history with ~20 tokens
        for _ in range(10):
            ext.extract(_make_response(completion_tokens=20), "m", 100.0)
        # Now send a very anomalous response (200 tokens, avg is ~20)
        signals = ext.extract(_make_response(completion_tokens=200), "m", 100.0)
        len_sig = next(s for s in signals if s.signal_name == "response_length_anomaly")
        assert len_sig.signal_value == 0.5  # Anomalous


# ---------------------------------------------------------------------------
# FeedbackStorage
# ---------------------------------------------------------------------------

class TestFeedbackStorage:
    def test_store_and_get_record(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record(
            request_id="req1",
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            latency_ms=100.0,
        )
        record = storage.get_record("req1")
        assert record is not None
        assert record["request_id"] == "req1"
        assert record["model_id"] == "gpt-4o"
        assert record["messages"] == [{"role": "user", "content": "hi"}]
        assert record["latency_ms"] == 100.0
        assert record["quality_score"] is None

    def test_get_record_not_found(self):
        storage = FeedbackStorage(":memory:")
        assert storage.get_record("nonexistent") is None

    def test_update_record(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record(
            request_id="req1",
            model_id="m",
            messages=[],
            latency_ms=50.0,
        )
        updated = storage.update_record("req1", quality_score=0.9, user_feedback="good")
        assert updated is True
        record = storage.get_record("req1")
        assert record["quality_score"] == 0.9
        assert record["user_feedback"] == "good"

    def test_update_record_not_found(self):
        storage = FeedbackStorage(":memory:")
        assert storage.update_record("nope", quality_score=0.5) is False

    def test_store_and_query_signals(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record("req1", "m", [], 50.0)
        storage.store_signal("req1", "implicit", "error_detected", 1.0, raw_value="stop")
        storage.store_signal("req1", "implicit", "empty_response", 0.0, raw_value=0)

        # Verify via direct query
        conn = storage._get_conn()
        rows = conn.execute(
            "SELECT * FROM outcome_signals WHERE request_id = ?", ("req1",)
        ).fetchall()
        assert len(rows) == 2

    def test_get_training_data(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record("req1", "m1", [{"role": "user", "content": "a"}], 50.0, quality_score=0.8)
        storage.store_record("req2", "m2", [{"role": "user", "content": "b"}], 60.0)  # No quality
        storage.store_record("req3", "m1", [{"role": "user", "content": "c"}], 70.0, quality_score=0.6)

        data = storage.get_training_data()
        assert len(data) == 2
        # Should be ordered by created_at DESC
        assert data[0]["request_id"] == "req3"

    def test_get_training_data_with_filters(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record("req1", "m1", [], 50.0, quality_score=0.8)
        storage.store_record("req2", "m2", [], 60.0, quality_score=0.3)

        data = storage.get_training_data(min_quality=0.5)
        assert len(data) == 1
        assert data[0]["request_id"] == "req1"

        data = storage.get_training_data(model_id="m2")
        assert len(data) == 1
        assert data[0]["model_id"] == "m2"

    def test_get_model_stats(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record("r1", "m1", [], 100.0, quality_score=0.8)
        storage.store_record("r2", "m1", [], 200.0, quality_score=0.6)
        storage.store_record("r3", "m2", [], 150.0)

        stats = storage.get_model_stats()
        assert stats["m1"]["count"] == 2
        assert stats["m1"]["avg_latency_ms"] == 150.0
        assert stats["m1"]["avg_quality"] == 0.7
        assert stats["m2"]["count"] == 1
        assert stats["m2"]["avg_quality"] is None

    def test_close(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record("r1", "m", [], 50.0)
        storage.close()
        assert storage._conn is None


# ---------------------------------------------------------------------------
# FeedbackCollector with storage & signals
# ---------------------------------------------------------------------------

class TestFeedbackCollectorIntegration:
    def _make_config(self, storage_path: str | None = None) -> RouteSmithConfig:
        return RouteSmithConfig(
            feedback_enabled=True,
            feedback_sample_rate=1.0,  # Always sample
            feedback_storage_path=storage_path,
        )

    def _make_registry(self) -> ModelRegistry:
        reg = ModelRegistry()
        reg.register("model-a", 0.001, 0.002, quality_score=0.9, latency_p99_ms=2000.0)
        return reg

    def test_record_with_request_id(self):
        config = self._make_config()
        collector = FeedbackCollector(config)
        record = collector.record(
            request_id="abc123",
            messages=[{"role": "user", "content": "hi"}],
            model="model-a",
            response=_make_response(),
            latency_ms=100.0,
        )
        assert record is not None
        assert record.request_id == "abc123"
        assert collector.get_record_by_id("abc123") is record

    def test_record_outcome_in_memory(self):
        config = self._make_config()
        collector = FeedbackCollector(config)
        collector.record(
            request_id="req1",
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            response=_make_response(),
            latency_ms=50.0,
        )
        found = collector.record_outcome("req1", success=True, feedback="great")
        assert found is True
        record = collector.get_record_by_id("req1")
        assert record.quality_score == 1.0
        assert record.user_feedback == "great"

    def test_record_outcome_not_found(self):
        config = self._make_config()
        collector = FeedbackCollector(config)
        assert collector.record_outcome("nope", score=0.5) is False

    def test_record_persists_to_storage(self):
        config = self._make_config(storage_path=":memory:")
        collector = FeedbackCollector(config)
        collector.record(
            request_id="req1",
            messages=[{"role": "user", "content": "hello"}],
            model="m",
            response=_make_response(),
            latency_ms=75.0,
        )
        # Check it's in storage
        stored = collector._storage.get_record("req1")
        assert stored is not None
        assert stored["model_id"] == "m"

    def test_signals_persisted_to_storage(self):
        config = self._make_config(storage_path=":memory:")
        registry = self._make_registry()
        collector = FeedbackCollector(config, registry=registry)
        collector.record(
            request_id="req1",
            messages=[{"role": "user", "content": "hi"}],
            model="model-a",
            response=_make_response(),
            latency_ms=100.0,
        )
        # Check signals were stored
        conn = collector._storage._get_conn()
        rows = conn.execute(
            "SELECT * FROM outcome_signals WHERE request_id = ?", ("req1",)
        ).fetchall()
        assert len(rows) == 6  # All 6 implicit signals

    def test_record_outcome_updates_storage(self):
        config = self._make_config(storage_path=":memory:")
        collector = FeedbackCollector(config)
        collector.record(
            request_id="req1",
            messages=[{"role": "user", "content": "hi"}],
            model="m",
            response=_make_response(),
            latency_ms=50.0,
        )
        collector.record_outcome("req1", score=0.85, feedback="good")

        stored = collector._storage.get_record("req1")
        assert stored["quality_score"] == 0.85
        assert stored["user_feedback"] == "good"

        # Check explicit signal was stored
        conn = collector._storage._get_conn()
        rows = conn.execute(
            "SELECT * FROM outcome_signals WHERE request_id = ? AND signal_type = 'explicit'",
            ("req1",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["signal_value"] == 0.85

    def test_eviction_cleans_request_index(self):
        config = self._make_config()
        collector = FeedbackCollector(config, max_records=3)
        for i in range(5):
            collector.record(
                request_id=f"req{i}",
                messages=[],
                model="m",
                response=_make_response(),
                latency_ms=10.0,
            )
        # Only last 3 should be in memory
        assert len(collector) == 3
        assert collector.get_record_by_id("req0") is None
        assert collector.get_record_by_id("req1") is None
        assert collector.get_record_by_id("req4") is not None


# ---------------------------------------------------------------------------
# RouteSmith client integration
# ---------------------------------------------------------------------------

class TestRouteSmithFeedbackIntegration:
    @pytest.fixture
    def client(self):
        config = RouteSmithConfig(
            feedback_enabled=True,
            feedback_sample_rate=1.0,
            feedback_storage_path=":memory:",
        )
        rs = RouteSmith(config=config)
        rs.register_model(
            "test-model",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            quality_score=0.85,
        )
        return rs

    @patch("routesmith.client.litellm")
    def test_completion_generates_request_id(self, mock_litellm, client):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "hi"}],
            include_metadata=True,
        )

        # request_id should be in metadata
        assert "request_id" in response.routesmith_metadata
        rid = response.routesmith_metadata["request_id"]
        assert len(rid) == 16

        # Should be on response object
        assert hasattr(response, "_routesmith_request_id")
        assert response._routesmith_request_id == rid

        # Should be in last_routing_metadata
        assert client.last_routing_metadata.request_id == rid

    @patch("routesmith.client.litellm")
    def test_record_outcome_updates_predictor(self, mock_litellm, client):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "hello"}],
        )
        rid = response._routesmith_request_id

        # Record outcome
        old_prior = client.router.predictor._ema_priors.get("test-model", 0.5)
        found = client.record_outcome(rid, score=0.95)
        assert found is True

        # Predictor should have been updated
        new_prior = client.router.predictor._ema_priors.get("test-model")
        assert new_prior != old_prior

    @patch("routesmith.client.litellm")
    def test_full_flow_completion_to_storage(self, mock_litellm, client):
        """Integration: completion → signals → record_outcome → storage → predictor."""
        mock_response = _make_response(content="Here's the answer!", completion_tokens=30)
        mock_litellm.completion.return_value = mock_response

        response = client.completion(
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        rid = response._routesmith_request_id

        # Verify record in storage
        stored = client.feedback._storage.get_record(rid)
        assert stored is not None
        assert stored["model_id"] == "test-model"

        # Record user outcome
        client.record_outcome(rid, score=0.9, feedback="correct answer")

        # Verify updated in storage
        stored = client.feedback._storage.get_record(rid)
        assert stored["quality_score"] == 0.9
        assert stored["user_feedback"] == "correct answer"

        # Verify training data available
        training = client.feedback._storage.get_training_data()
        assert len(training) >= 1
        assert any(r["quality_score"] == 0.9 for r in training)


class TestFeedbackConfigPreservation:
    def test_with_cache_preserves_storage_path(self):
        config = RouteSmithConfig(feedback_storage_path="/tmp/test.db")
        new_config = config.with_cache(enabled=True)
        assert new_config.feedback_storage_path == "/tmp/test.db"

    def test_with_budget_preserves_storage_path(self):
        config = RouteSmithConfig(feedback_storage_path="/tmp/test.db")
        new_config = config.with_budget(quality_threshold=0.9)
        assert new_config.feedback_storage_path == "/tmp/test.db"

    def test_default_storage_path_is_none(self):
        config = RouteSmithConfig()
        assert config.feedback_storage_path is None
