"""Real use case: exception hierarchy enforces consistent error handling."""
import json

from routesmith.exceptions import (
    BudgetExceededError,
    CircuitOpenError,
    NoCapableModelError,
    ProviderUnavailableError,
    RouteSmithError,
)


class TestExceptionHierarchy:
    """Verify exception classes have correct inheritance and info."""

    def test_budget_exceeded_is_routesmith_error(self):
        err = BudgetExceededError("Budget exceeded", current_spend=5.0, limit=10.0, reset_seconds=3600)
        assert isinstance(err, RouteSmithError)

    def test_budget_exceeded_stores_context(self):
        err = BudgetExceededError("message", current_spend=5.0, limit=10.0, reset_seconds=3600)
        assert err.current_spend == 5.0
        assert err.limit == 10.0
        assert err.reset_seconds == 3600

    def test_budget_exceeded_str_includes_details(self):
        err = BudgetExceededError("nope", current_spend=9.50, limit=10.0, reset_seconds=1800)
        s = str(err)
        assert "9.50" in s
        assert "10.0" in s

    def test_no_capable_model_stores_requirements(self):
        err = NoCapableModelError(
            "No model supports the required capabilities",
            required_capabilities={"tool_calling", "vision"},
            available_models=["gpt-4o-mini"],
        )
        assert err.required_capabilities == {"tool_calling", "vision"}
        assert err.available_models == ["gpt-4o-mini"]

    def test_provider_unavailable_stores_original_error(self):
        original = ValueError("connection refused")
        err = ProviderUnavailableError("gpt-4o", original)
        assert err.model_id == "gpt-4o"
        assert err.original_error is original

    def test_circuit_open_stores_retry_after(self):
        err = CircuitOpenError("gpt-4o-mini", retry_after=30.0)
        assert err.model_id == "gpt-4o-mini"
        assert err.retry_after == 30.0

    def test_arbitrary_kwargs_survive(self):
        """Custom exceptions shouldn't lose kwargs passed through."""
        err = RouteSmithError("test", custom_field="x")
        assert err.custom_field == "x"


class TestRealUseCaseErrors:
    """End-to-end: exceptions carry enough info for a monitoring system."""

    def test_budget_error_serializable_for_alerts(self):
        """Alerts can parse budget context from exceptions."""
        err = BudgetExceededError("over", current_spend=99.99, limit=100.0, reset_seconds=600)
        data = {
            "type": type(err).__name__,
            "message": str(err),
            "current_spend": err.current_spend,
            "limit": err.limit,
            "reset_seconds": err.reset_seconds,
        }
        json.dumps(data)  # Should not raise

    def test_circuit_error_carries_retry_after_for_scheduler(self):
        """Scheduler can parse retry_after to re-enable model later."""
        err = CircuitOpenError("claude-3-opus", retry_after=60.0)
        assert err.retry_after > 0
        assert err.retry_after == 60.0

    def test_provider_error_chains_cause_for_debugging(self):
        """Root cause is preserved through raise...from chain."""
        try:
            try:
                raise ConnectionRefusedError("Connection refused")
            except ConnectionRefusedError as original:
                raise ProviderUnavailableError("gpt-4o", original) from original
        except ProviderUnavailableError as err:
            assert err.__cause__ is not None
            assert isinstance(err.__cause__, ConnectionRefusedError)
