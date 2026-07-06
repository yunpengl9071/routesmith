"""Tests for implicit signal feedback (P1.3)."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from routesmith import RouteSmith
from routesmith.config import RouteSmithConfig
from routesmith.feedback.signals import (
    IMPLICIT_QUALITY,
    QualitySignal,
    implicit_quality,
)


def _make_mock_response(content: str = "Hello!", finish_reason: str = "stop") -> MagicMock:
    """Build a minimal litellm ModelResponse-like object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    # Attach a minimal usage object so cost tracking doesn't crash.
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


@pytest.fixture
def rs() -> RouteSmith:
    """RouteSmith with one registered model and feedback enabled."""
    cfg = RouteSmithConfig(feedback_sample_rate=1.0)
    rs = RouteSmith(config=cfg)
    rs.register_model(
        "test-model",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        quality_score=0.8,
    )
    return rs


@pytest.fixture
def rs_no_implicit() -> RouteSmith:
    """RouteSmith with implicit feedback disabled."""
    cfg = RouteSmithConfig(feedback_sample_rate=1.0, implicit_feedback_enabled=False)
    rs = RouteSmith(config=cfg)
    rs.register_model(
        "test-model",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        quality_score=0.8,
    )
    return rs


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestImplicitQualityPure:
    def test_empty_list_returns_none(self) -> None:
        assert implicit_quality([]) is None

    def test_clean_signals_returns_none(self) -> None:
        signals = [
            QualitySignal("implicit", "refusal_detected", 1.0),
            QualitySignal("implicit", "error_detected", 1.0),
            QualitySignal("implicit", "truncated_response", 1.0),
            QualitySignal("implicit", "empty_response", 1.0),
        ]
        assert implicit_quality(signals) is None

    def test_refusal_triggered(self) -> None:
        signals = [
            QualitySignal("implicit", "refusal_detected", 0.0),
        ]
        assert implicit_quality(signals) == 0.05

    def test_error_triggered(self) -> None:
        signals = [
            QualitySignal("implicit", "error_detected", 0.0),
        ]
        assert implicit_quality(signals) == 0.05

    def test_empty_triggered(self) -> None:
        signals = [
            QualitySignal("implicit", "empty_response", 0.0),
        ]
        assert implicit_quality(signals) == 0.05

    def test_truncation_triggered(self) -> None:
        signals = [
            QualitySignal("implicit", "truncated_response", 0.0),
        ]
        assert implicit_quality(signals) == 0.40

    def test_multiple_negative_returns_worst(self) -> None:
        signals = [
            QualitySignal("implicit", "truncated_response", 0.0),
            QualitySignal("implicit", "refusal_detected", 0.0),
        ]
        # truncation is 0.40, refusal is 0.05, min is 0.05
        assert implicit_quality(signals) == 0.05

    def test_signal_value_boundary_not_triggered(self) -> None:
        signals = [
            QualitySignal("implicit", "refusal_detected", 0.5),
        ]
        # 0.5 is not < 0.5, so not triggered
        assert implicit_quality(signals) is None

    def test_excluded_signals_are_ignored(self) -> None:
        signals = [
            QualitySignal("implicit", "response_length_anomaly", 0.0),
            QualitySignal("implicit", "latency_anomaly", 0.0),
        ]
        assert implicit_quality(signals) is None

    def test_mixed_triggered_and_clean(self) -> None:
        signals = [
            QualitySignal("implicit", "refusal_detected", 0.0),
            QualitySignal("implicit", "response_length_anomaly", 0.5),
        ]
        assert implicit_quality(signals) == 0.05


# ---------------------------------------------------------------------------
# Integration tests (mocked litellm)
# ---------------------------------------------------------------------------


class TestImplicitLearning:
    def test_refusal_triggers_negative_update(self, rs: RouteSmith) -> None:
        """Refusal content should cause predictor.update with quality <= 0.05."""
        spy_calls: list[tuple] = []
        original_update = rs.router.predictor.update

        def spy(*args: object, **kwargs: object) -> None:
            spy_calls.append((args, kwargs))
            return original_update(*args, **kwargs)

        rs.router.predictor.update = spy

        mock_resp = _make_mock_response(
            content="I'm sorry, I can't help with that.",
        )

        with patch("litellm.completion", return_value=mock_resp):
            rs.completion(messages=[{"role": "user", "content": "do something bad"}])

        assert len(spy_calls) == 1
        args, kwargs = spy_calls[0]
        # actual_quality should be 0.05 (IMPLICIT_QUALITY["refusal_detected"])
        assert kwargs.get("actual_quality", args[2] if len(args) >= 3 else None) <= 0.05

    def test_truncation_triggers_mild_negative(self, rs: RouteSmith) -> None:
        """finish_reason='length' should trigger update with 0.40."""
        spy_calls: list[tuple] = []
        original_update = rs.router.predictor.update

        def spy(*args: object, **kwargs: object) -> None:
            spy_calls.append((args, kwargs))
            return original_update(*args, **kwargs)

        rs.router.predictor.update = spy

        mock_resp = _make_mock_response(
            content="Some partial response...", finish_reason="length",
        )

        with patch("litellm.completion", return_value=mock_resp):
            rs.completion(messages=[{"role": "user", "content": "write a long essay"}])

        assert len(spy_calls) == 1
        args, kwargs = spy_calls[0]
        actual_q = kwargs.get("actual_quality", args[2] if len(args) >= 3 else None)
        assert actual_q == 0.40

    def test_clean_response_no_implicit_update(self, rs: RouteSmith) -> None:
        """Normal response should NOT call predictor.update."""
        spy_calls: list[tuple] = []
        original_update = rs.router.predictor.update

        def spy(*args: object, **kwargs: object) -> None:
            spy_calls.append((args, kwargs))
            return original_update(*args, **kwargs)

        rs.router.predictor.update = spy

        mock_resp = _make_mock_response(content="Everything is fine.")

        with patch("litellm.completion", return_value=mock_resp):
            rs.completion(messages=[{"role": "user", "content": "hello"}])

        # No implicit updates for clean responses
        for call_args, call_kwargs in spy_calls:
            actual_q = call_kwargs.get("actual_quality", call_args[2] if len(call_args) >= 3 else None)
            # The predictor may get called during registration (add_arm),
            # but NOT with an implicit quality value
            if actual_q is not None and actual_q < 1.0 and actual_q > 0:
                # This should not happen for a clean response
                assert False, f"Unexpected predictor.update called with actual_quality={actual_q}"

    def test_implicit_disabled_no_update(self, rs_no_implicit: RouteSmith) -> None:
        """When implicit_feedback_enabled=False, refusal should NOT trigger update."""
        spy_calls: list[tuple] = []
        original_update = rs_no_implicit.router.predictor.update

        def spy(*args: object, **kwargs: object) -> None:
            spy_calls.append((args, kwargs))
            return original_update(*args, **kwargs)

        rs_no_implicit.router.predictor.update = spy

        mock_resp = _make_mock_response(
            content="I'm sorry, I can't help with that.",
        )

        with patch("litellm.completion", return_value=mock_resp):
            rs_no_implicit.completion(messages=[{"role": "user", "content": "do something bad"}])

        # Check no update was called with implicit quality values
        for call_args, call_kwargs in spy_calls:
            actual_q = call_kwargs.get("actual_quality", call_args[2] if len(call_args) >= 3 else None)
            if actual_q is not None and actual_q <= 0.05:
                assert False, "predictor.update was called with an implicit quality even though implicit_feedback_enabled=False"

    def test_async_refusal_triggers_negative_update(self, rs: RouteSmith) -> None:
        """Async path: refusal content should trigger predictor.update."""
        spy_calls: list[tuple] = []
        original_update = rs.router.predictor.update

        def spy(*args: object, **kwargs: object) -> None:
            spy_calls.append((args, kwargs))
            return original_update(*args, **kwargs)

        rs.router.predictor.update = spy

        mock_resp = _make_mock_response(
            content="I'm sorry, I can't help with that.",
        )

        with patch("litellm.acompletion", return_value=mock_resp):
            import asyncio
            asyncio.run(
                rs.acompletion(messages=[{"role": "user", "content": "do something bad"}])
            )

        assert len(spy_calls) >= 1
        args, kwargs = spy_calls[0]
        actual_q = kwargs.get("actual_quality", args[2] if len(args) >= 3 else None)
        assert actual_q <= 0.05
