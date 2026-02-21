"""Tests for feature extraction."""

import math
import time

import pytest

from routesmith.predictor.features import FeatureExtractor, FeatureVector
from routesmith.registry.models import ModelRegistry


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
        """Extract produces exactly 19 features."""
        msgs = [{"role": "user", "content": "Hello"}]
        fv = extractor.extract(msgs, "gpt-4o")
        assert len(fv.features) == 19
        assert len(fv.feature_names) == 19

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
        # cost_per_1k_input (index 11 after tools_present at 10)
        assert fv.features[11] == 0.005
        # cost_per_1k_output
        assert fv.features[12] == 0.015
        # quality_prior
        assert fv.features[13] == 0.95
        # latency_p50_ms
        assert fv.features[14] == 800.0
        # context_window_log
        assert abs(fv.features[15] - math.log(128000)) < 0.01
        # supports_function_calling
        assert fv.features[16] == 1.0
        # supports_vision
        assert fv.features[17] == 1.0
        # supports_json_mode
        assert fv.features[18] == 1.0

    def test_known_model_no_vision(self, extractor):
        fv = extractor.extract([{"role": "user", "content": "Hi"}], "gpt-4o-mini")
        assert fv.features[17] == 0.0  # no vision

    def test_unknown_model(self, extractor):
        fv = extractor.extract([{"role": "user", "content": "Hi"}], "unknown-model")
        # Should return defaults without crashing
        assert len(fv.features) == 19
        # quality_prior default = 0.5
        assert fv.features[13] == 0.5


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
