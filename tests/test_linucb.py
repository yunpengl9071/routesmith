"""Tests for LinUCB predictor."""

from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.registry.models import ModelRegistry


def _make_linucb():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    reg.register("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.7)
    return LinUCBPredictor(registry=reg)


class TestLinUCBArmLifecycle:
    def test_arm_lazy_init_on_predict(self):
        """Arms are lazily initialized on first predict call."""
        p = _make_linucb()
        msgs = [{"role": "user", "content": "test"}]
        results = p.predict(msgs, ["gpt-4o", "gpt-4o-mini"])
        assert len(results) == 2
        assert all(r.model_id in ("gpt-4o", "gpt-4o-mini") for r in results)
        # Arms should be auto-created
        assert "gpt-4o" in p._arms
        assert "gpt-4o-mini" in p._arms

    def test_unknown_model_predicted_with_defaults(self):
        """Models not in the registry get neutral defaults without error."""
        p = _make_linucb()
        msgs = [{"role": "user", "content": "test"}]
        results = p.predict(msgs, ["unknown-model"])
        assert len(results) == 1
        assert results[0].model_id == "unknown-model"

    def test_update_increments_count(self):
        """update() increments the arm's update count."""
        p = _make_linucb()
        msgs = [{"role": "user", "content": "hi"}]
        p.predict(msgs, ["gpt-4o"])  # lazy init
        count_before = p._arms.get("gpt-4o", {}).get("count", 0)
        p.update(msgs, "gpt-4o", 0.8)
        assert p._arms.get("gpt-4o", {}).get("count", 0) == count_before + 1

    def test_update_on_new_model_auto_inits(self):
        """update() on a model not yet seen auto-initializes the arm."""
        p = _make_linucb()
        msgs = [{"role": "user", "content": "hi"}]
        p.update(msgs, "gpt-4o", 0.85)
        assert "gpt-4o" in p._arms
        assert p._arms["gpt-4o"]["count"] == 1

    def test_diagnostics_returns_state(self):
        """diagnostics() returns structured info."""
        p = _make_linucb()
        msgs = [{"role": "user", "content": "test"}]
        p.predict(msgs, ["gpt-4o"])
        p.update(msgs, "gpt-4o", 0.85)
        diag = p.diagnostics()
        assert diag["type"] == "linucb"
        assert "gpt-4o" in diag["arms"]

    def test_get_arm_weights(self):
        """get_arm_weights returns learned feature weights."""
        p = _make_linucb()
        msgs = [{"role": "user", "content": "test"}]
        p.update(msgs, "gpt-4o", 0.9)
        weights = p.get_arm_weights("gpt-4o")
        assert weights is not None
        assert len(weights) == 27  # 27 features

    def test_get_arm_weights_unknown_returns_none(self):
        """get_arm_weights returns None for unknown model."""
        p = _make_linucb()
        assert p.get_arm_weights("nonexistent") is None
