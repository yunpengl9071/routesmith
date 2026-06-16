"""Tests for feature extraction."""

import math
import time

import pytest

from routesmith.config import RouteContext
from routesmith.predictor.features import FeatureExtractor, FeatureVector
from routesmith.registry.models import ModelRegistry


def _make_registry():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    return reg


class TestContextFeatures:
    def test_no_context_gives_35_features_zeros_for_new_dims(self):
        extractor = FeatureExtractor(_make_registry())
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=None)
        assert len(fv.features) == 35
        assert all(f == 0.0 for f in fv.features[27:])

    def test_context_with_explicit_role_sets_confidence_1(self):
        extractor = FeatureExtractor(_make_registry())
        ctx = RouteContext(agent_role="research", turn_index=2)
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=ctx)
        assert len(fv.features) == 35
        assert fv.features[32] == pytest.approx(1.0)  # agent_role_confidence
        assert fv.features[34] == pytest.approx(1.0)  # has_agent_context

    def test_turn_index_normalized(self):
        extractor = FeatureExtractor(_make_registry())
        ctx = RouteContext(turn_index=10)
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=ctx)
        assert fv.features[27] == pytest.approx(0.5)  # 10/20

    def test_turn_index_capped_at_one(self):
        extractor = FeatureExtractor(_make_registry())
        ctx = RouteContext(turn_index=100)
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=ctx)
        assert fv.features[27] == pytest.approx(1.0)

    def test_feature_names_length(self):
        extractor = FeatureExtractor(_make_registry())
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o")
        assert len(fv.feature_names) == 35

    def test_backward_compat_no_context_arg(self):
        """extract() with no context kwarg still returns 35 features with zeros."""
        extractor = FeatureExtractor(_make_registry())
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o")
        assert len(fv.features) == 35


@pytest.fixture
def registry():
    reg = ModelRegistry()
    # register() passes extra kwargs to metadata, not dataclass fields.
    # To set supports_vision etc., we modify the ModelConfig directly.
    cfg = reg.register(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
        latency_p50_ms=800.0,
        context_window=128000,
    )
    cfg.supports_vision = True

    reg.register(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
        latency_p50_ms=300.0,
        context_window=128000,
    )
    return reg


@pytest.fixture
def extractor(registry):
    return FeatureExtractor(registry)


class TestFeatureVector:
    def test_feature_count(self, extractor):
        """Extract produces exactly 35 features (27 base + 8 context)."""
        msgs = [{"role": "user", "content": "Hello"}]
        fv = extractor.extract(msgs, "gpt-4o")
        assert len(fv.features) == 35
        assert len(fv.feature_names) == 35

    def test_feature_names_match_length(self, extractor):
        """feature_names length always matches features length."""
        fv = extractor.extract([], "gpt-4o")
        assert len(fv.features) == len(fv.feature_names)


class TestMessageFeatures:
    def test_single_user_message(self, extractor):
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        fv = extractor.extract(msgs, "gpt-4o")
        # msg_count = 1
        assert fv.features[0] == 1.0
        # user_msg_count = 1
        assert fv.features[4] == 1.0
        # system_msg_present = 0
        assert fv.features[5] == 0.0
        # question_mark_count >= 1
        assert fv.features[7] >= 1.0

    def test_multi_turn_conversation(self, extractor):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        fv = extractor.extract(msgs, "gpt-4o")
        # msg_count = 4
        assert fv.features[0] == 4.0
        # user_msg_count = 2
        assert fv.features[4] == 2.0
        # system_msg_present = 1
        assert fv.features[5] == 1.0
        # last_msg_length = len("How are you?")
        assert fv.features[6] == float(len("How are you?"))

    def test_empty_messages(self, extractor):
        fv = extractor.extract([], "gpt-4o")
        # All message features should be 0
        for i in range(11):
            assert fv.features[i] == 0.0

    def test_word_count_and_avg_word_length(self, extractor):
        msgs = [{"role": "user", "content": "hello world"}]
        fv = extractor.extract(msgs, "gpt-4o")
        # word_count = 2
        assert fv.features[8] == 2.0
        # avg_word_length = (5 + 5) / 2 = 5.0
        assert fv.features[9] == 5.0


class TestModelFeatures:
    def test_known_model(self, extractor):
        msgs = [{"role": "user", "content": "Hi"}]
        fv = extractor.extract(msgs, "gpt-4o")
        # Model features start at index 17 (after 17 message features)
        # cost_per_1k_input
        assert fv.features[17] == 0.005
        # cost_per_1k_output
        assert fv.features[18] == 0.015
        # quality_prior
        assert fv.features[19] == 0.95
        # latency_p50_ms
        assert fv.features[20] == 800.0
        # context_window_log
        assert abs(fv.features[21] - math.log(128000)) < 0.01
        # supports_function_calling
        assert fv.features[22] == 1.0
        # supports_vision
        assert fv.features[23] == 1.0
        # supports_json_mode
        assert fv.features[24] == 1.0

    def test_known_model_no_vision(self, extractor):
        fv = extractor.extract([{"role": "user", "content": "Hi"}], "gpt-4o-mini")
        assert fv.features[23] == 0.0  # no vision (index 23)

    def test_unknown_model(self, extractor):
        fv = extractor.extract([{"role": "user", "content": "Hi"}], "unknown-model")
        # Should return defaults without crashing (35 features: 27 base + 8 context)
        assert len(fv.features) == 35
        # quality_prior default = 0.5 (at index 19 in model features)
        assert fv.features[19] == 0.5


class TestPerformance:
    def test_extraction_speed(self, extractor):
        """Feature extraction should be fast (<1ms)."""
        msgs = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Explain quantum computing in detail."},
        ]
        start = time.perf_counter()
        for _ in range(100):
            extractor.extract(msgs, "gpt-4o")
        elapsed = (time.perf_counter() - start) / 100
        # Should be well under 1ms per extraction
        assert elapsed < 0.001, f"Feature extraction took {elapsed*1000:.2f}ms"
