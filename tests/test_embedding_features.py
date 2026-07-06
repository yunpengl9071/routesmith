"""Tests for optional embedding features (P2.2) in FeatureExtractor.

Uses a FakeEncoder to avoid loading real sentence-transformers in CI.
"""

from __future__ import annotations

import hashlib

import numpy as np

from routesmith.predictor.features import FeatureExtractor
from routesmith.registry.models import ModelRegistry


class FakeEncoder:
    """Deterministic fake encoder that produces reproducible 384-d vectors."""

    def encode(self, text, normalize_embeddings=True):
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:4], "little"))
        return rng.normal(0, 1, 384).reshape(1, -1)


def _make_registry():
    reg = ModelRegistry()
    reg.register(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.9,
    )
    reg.register(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.7,
    )
    return reg


class TestDim:
    def test_dim_35_when_enabled(self):
        ext = FeatureExtractor(
            _make_registry(),
            use_embeddings=True,
            _encoder_override=FakeEncoder(),
        )
        assert ext.dim == 35
        fv = ext.extract([{"role": "user", "content": "hello"}], "gpt-4o")
        assert len(fv.features) == 35
        assert len(fv.feature_names) == 35

    def test_dim_27_when_disabled(self):
        ext = FeatureExtractor(_make_registry())
        assert ext.dim == 27
        fv = ext.extract([{"role": "user", "content": "hello"}], "gpt-4o")
        assert len(fv.features) == 27
        assert len(fv.feature_names) == 27

    def test_extract_message_and_context_returns_17_plus_8(self):
        ext = FeatureExtractor(
            _make_registry(),
            use_embeddings=True,
            _encoder_override=FakeEncoder(),
        )
        msg_feats, sem_feats = ext.extract_message_and_context(
            [{"role": "user", "content": "hello"}]
        )
        assert len(msg_feats) == 17
        assert len(sem_feats) == 8

    def test_extract_for_model_35_features(self):
        ext = FeatureExtractor(
            _make_registry(),
            use_embeddings=True,
            _encoder_override=FakeEncoder(),
        )
        msg_feats, sem_feats = ext.extract_message_and_context(
            [{"role": "user", "content": "hello"}]
        )
        fv = ext.extract_for_model(msg_feats, sem_feats, "gpt-4o")
        assert len(fv.features) == 35


class TestSemanticDims:
    def test_differ_across_topics(self):
        """Different topics produce different sem blocks; same text is identical."""
        ext = FeatureExtractor(
            _make_registry(),
            use_embeddings=True,
            _encoder_override=FakeEncoder(),
            normalize=False,
        )
        fv1 = ext.extract(
            [{"role": "user", "content": "solve this integral"}], "gpt-4o"
        )
        fv2 = ext.extract(
            [{"role": "user", "content": "write a poem"}], "gpt-4o"
        )
        fv3 = ext.extract(
            [{"role": "user", "content": "solve this integral"}], "gpt-4o"
        )

        sem1 = fv1.features[27:]
        sem2 = fv2.features[27:]
        sem3 = fv3.features[27:]

        assert sem1 != sem2, "Different topics should yield different sem vectors"
        assert sem1 == sem3, "Same text should yield identical sem vectors (cache)"

    def test_feature_names_and_scales(self):
        ext = FeatureExtractor(
            _make_registry(),
            use_embeddings=True,
            _encoder_override=FakeEncoder(),
        )
        assert ext.dim == 35
        for i in range(8):
            assert ext._feature_names[27 + i] == f"sem_{i}"
        assert len(ext._scales) == 35
        for i in range(27, 35):
            assert ext._scales[i] == 1.0


class TestEmbeddingCallCount:
    def test_embedding_once_per_predict(self):
        """Encoder.encode() is called exactly once in a multi-candidate predict."""
        encoder = FakeEncoder()
        call_count = [0]
        orig_encode = encoder.encode

        def counting_encode(text, normalize_embeddings=True):
            call_count[0] += 1
            return orig_encode(text, normalize_embeddings)

        encoder.encode = counting_encode

        from routesmith.predictor.lints import LinTSPredictor

        extractor = FeatureExtractor(
            _make_registry(),
            use_embeddings=True,
            _encoder_override=encoder,
        )
        pred = LinTSPredictor(_make_registry(), extractor=extractor)

        msgs = [{"role": "user", "content": "solve this integral"}]
        model_ids = ["gpt-4o", "gpt-4o-mini"]
        pred.predict(msgs, model_ids)

        assert call_count[0] == 1, f"Expected 1 encode call, got {call_count[0]}"


class TestStateColdStart:
    def test_cold_start_on_dim_change(self):
        """Save 27-dim state, load into 35-dim predictor -> cold start."""
        from routesmith.predictor.lints import LinTSPredictor

        reg = _make_registry()
        pred_27 = LinTSPredictor(reg)
        pred_27.update(
            [{"role": "user", "content": "hello"}], "gpt-4o", actual_quality=0.9
        )
        blob = pred_27.serialize_state()

        extractor_35 = FeatureExtractor(
            reg,
            use_embeddings=True,
            _encoder_override=FakeEncoder(),
        )
        pred_35 = LinTSPredictor(reg, extractor=extractor_35)
        assert pred_35._router.d == 35

        pred_35.load_state(blob)
        assert pred_35._router._t == 0, "Should cold-start on dim mismatch"


class TestMissingDependency:
    def test_falls_back_gracefully(self):
        """Force ImportError -> dim 27, no crash."""
        orig_import = __builtins__["__import__"]

        def broken_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated missing dependency")
            return orig_import(name, *args, **kwargs)

        __builtins__["__import__"] = broken_import
        try:
            ext = FeatureExtractor(_make_registry(), use_embeddings=True)
            assert ext.dim == 27
            fv = ext.extract(
                [{"role": "user", "content": "hello"}], "gpt-4o"
            )
            assert len(fv.features) == 27
        finally:
            __builtins__["__import__"] = orig_import
