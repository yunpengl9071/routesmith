import json

from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.registry.models import ModelRegistry


def _make_linucb():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    reg.register("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.7)
    return LinUCBPredictor(registry=reg)


class TestLinUCBArmLifecycle:
    def test_new_arm_predicted_after_add(self):
        p = _make_linucb()
        p.add_arm("claude-haiku")
        msgs = [{"role": "user", "content": "test"}]
        results = p.predict(msgs, ["gpt-4o", "claude-haiku"])
        assert any(r.model_id == "claude-haiku" for r in results)

    def test_add_existing_arm_preserves_state(self):
        p = _make_linucb()
        msgs = [{"role": "user", "content": "hi"}]
        p.update(msgs, "gpt-4o", 0.8)
        count_before = p._arms.get("gpt-4o", {}).get("count", 0)
        p.add_arm("gpt-4o")
        assert p._arms.get("gpt-4o", {}).get("count", 0) == count_before

    def test_remove_arm(self):
        p = _make_linucb()
        # Trigger lazy init
        p.predict([{"role": "user", "content": "hi"}], ["gpt-4o-mini"])
        p.remove_arm("gpt-4o-mini")
        assert "gpt-4o-mini" not in p._arms

    def test_remove_nonexistent_is_noop(self):
        p = _make_linucb()
        p.remove_arm("nonexistent")  # should not raise

    def test_serialize_deserialize_roundtrip(self):
        p = _make_linucb()
        msgs = [{"role": "user", "content": "test"}]
        p.update(msgs, "gpt-4o", 0.85)
        blob = p.serialize_state()
        assert isinstance(blob, bytes)
        state = json.loads(blob.decode())
        assert "arms" in state

        p2 = _make_linucb()
        p2.load_state(blob)
        assert p2._total_updates == p._total_updates
