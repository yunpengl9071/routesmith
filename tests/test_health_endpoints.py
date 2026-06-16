from routesmith import RouteSmith
from routesmith.proxy.handler import RequestHandler


def make_handler():
    rs = RouteSmith()
    rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.85)
    return RequestHandler(rs)


class TestHealthEndpoint:
    async def test_health_returns_ok(self):
        handler = make_handler()
        resp = await handler.handle_health()
        assert resp["status"] == "healthy"

    async def test_health_includes_models_count(self):
        handler = make_handler()
        resp = await handler.handle_health()
        assert "registered_models" in resp

    async def test_liveness_always_alive(self):
        handler = make_handler()
        resp = await handler.handle_liveness()
        assert resp["status"] == "alive"

    async def test_readiness_fails_with_no_models(self):
        rs = RouteSmith()  # No models = not ready
        handler = RequestHandler(rs)
        resp = await handler.handle_readiness()
        assert resp["status"] != "ready"
        assert "reason" in resp

    async def test_readiness_ok_with_models(self):
        handler = make_handler()
        resp = await handler.handle_readiness()
        assert resp["status"] == "ready"
        assert resp["models"] == 1
