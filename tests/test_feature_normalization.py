"""Tests for per-feature normalization in FeatureExtractor."""

from routesmith.predictor.features import (
    FEATURE_SCALES,
    FEATURE_VERSION,
    FeatureExtractor,
    _normalize,
)
from routesmith.registry.models import ModelRegistry


def _make_registry():
    reg = ModelRegistry()
    reg.register("test-model", 0.001, 0.002, quality_score=0.85)
    return reg


def test_scales_length_matches_feature_names():
    assert len(FEATURE_SCALES) == len(FeatureExtractor.ALL_FEATURE_NAMES) == 27


def test_feature_version_defined():
    assert isinstance(FEATURE_VERSION, int)
    assert FEATURE_VERSION >= 2


def test_all_features_bounded():
    """Adversarial input produces every normalized feature in [0, 1.5]."""
    reg = _make_registry()
    ext = FeatureExtractor(reg, normalize=True)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "?" * 10},
    ] * 15
    messages.append({"role": "user", "content": "a" * 8000})
    fv = ext.extract(messages, "test-model")
    for val in fv.features:
        assert 0.0 <= val <= 1.5, f"Feature {val} out of [0, 1.5]"


def test_normalize_can_be_disabled():
    """normalize=False produces raw (unscaled) values."""
    reg = _make_registry()
    ext = FeatureExtractor(reg, normalize=False)
    messages = [{"role": "user", "content": "hello world"}]
    fv = ext.extract(messages, "test-model")
    raw_msg_count = fv.features[0]
    assert raw_msg_count == 1.0
    assert raw_msg_count * FEATURE_SCALES[0] > 0


def test_type_scores_unchanged_by_normalization():
    """Indices 11-16 (already 0-1) are unchanged for short messages."""
    reg = _make_registry()
    ext = FeatureExtractor(reg, normalize=True)
    messages = [{"role": "user", "content": "hello world"}]
    fv = ext.extract(messages, "test-model")
    for i in range(11, 17):
        val = fv.features[i]
        assert 0.0 <= val <= 1.0


def test_normalize_applied_via_extract_for_model():
    """extract_for_model also normalizes when normalize=True."""
    reg = _make_registry()
    ext = FeatureExtractor(reg, normalize=True)
    messages = [{"role": "user", "content": "solve this integral"}]
    msg_feats, ctx_feats = ext.extract_message_and_context(messages)
    fv = ext.extract_for_model(msg_feats, ctx_feats, "test-model")
    for val in fv.features:
        assert 0.0 <= val <= 1.5


def test_state_cold_start_on_feature_version_mismatch_lints():
    """load_state with stale feature_version skips load (cold start)."""
    from routesmith.predictor.lints import LinTSPredictor

    reg = _make_registry()
    pred = LinTSPredictor(reg)
    state = pred.serialize_state()
    import json
    decoded = json.loads(state.decode())
    decoded["feature_version"] = 1
    stale_state = json.dumps(decoded).encode()
    old_total = pred._total_updates
    pred.update(
        [{"role": "user", "content": "hello"}],
        "test-model",
        actual_quality=0.9,
    )
    assert pred._total_updates > old_total
    pred.load_state(stale_state)
    assert pred._total_updates > old_total
