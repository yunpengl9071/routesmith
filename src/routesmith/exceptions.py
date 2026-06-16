"""RouteSmith exception hierarchy for production error handling."""

from __future__ import annotations


class RouteSmithError(Exception):
    """Base exception for all RouteSmith errors."""

    def __init__(self, message: str = "", **kwargs: object) -> None:
        super().__init__(message)
        for key, value in kwargs.items():
            setattr(self, key, value)


class BudgetExceededError(RouteSmithError):
    """
    Raised when budget limit is exceeded.

    Attributes:
        current_spend: Current spend in USD.
        limit: Budget limit in USD.
        reset_seconds: Seconds until budget resets.
    """

    def __init__(
        self,
        message: str = "",
        *,
        current_spend: float = 0.0,
        limit: float = 0.0,
        reset_seconds: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.current_spend = current_spend
        self.limit = limit
        self.reset_seconds = reset_seconds

    def __str__(self) -> str:
        base = super().__str__() or "Budget exceeded"
        parts = [base]
        if self.current_spend > 0:
            parts.append(f"Spent ${self.current_spend:.2f}")
        if self.limit > 0:
            parts.append(f"Limit ${self.limit:.2f}")
        if self.reset_seconds > 0:
            minutes = int(self.reset_seconds / 60)
            parts.append(f"Resets in {minutes} minutes")
        return ", ".join(parts)


class NoCapableModelError(RouteSmithError):
    """
    Raised when no registered model satisfies required capabilities.

    Attributes:
        required_capabilities: Set of capabilities that were required.
        available_models: List of model IDs that were registered.
    """

    def __init__(
        self,
        message: str = "",
        *,
        required_capabilities: set[str] | None = None,
        available_models: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(message, **kwargs)
        self.required_capabilities = required_capabilities or set()
        self.available_models = available_models or []


class ProviderUnavailableError(RouteSmithError):
    """
    Raised when a provider/model returns a non-retryable error.

    Attributes:
        model_id: The model that failed.
        original_error: The original exception from the provider.
    """

    def __init__(
        self,
        model_id: str,
        original_error: Exception | None = None,
        **kwargs: object,
    ) -> None:
        msg = f"Provider unavailable for model '{model_id}'"
        if original_error:
            msg += f": {original_error}"
        super().__init__(msg, **kwargs)
        self.model_id = model_id
        self.original_error = original_error


class CapacityExhaustedError(RouteSmithError):
    """
    Raised when provisioned capacity is exhausted and no on-demand fallback exists.

    Attributes:
        model_id: The model whose capacity is exhausted.
    """

    def __init__(
        self,
        model_id: str,
        **kwargs: object,
    ) -> None:
        msg = f"Provisioned capacity exhausted for model '{model_id}'"
        super().__init__(msg, **kwargs)
        self.model_id = model_id


class NoCompliantModelError(RouteSmithError):
    """
    Raised when no registered model satisfies required compliance tags.

    Attributes:
        required_tags: The compliance tags that were required.
        available_tags: All compliance tags available across registered models.
    """

    def __init__(
        self,
        required_tags: set[str],
        available_tags: set[str] | None = None,
        **kwargs: object,
    ) -> None:
        msg = f"No models with required compliance tags: {required_tags}"
        if available_tags:
            msg += f". Available tags: {available_tags}"
        super().__init__(msg, **kwargs)
        self.required_tags = required_tags
        self.available_tags = available_tags or set()


class CircuitOpenError(RouteSmithError):
    """
    Raised when the circuit breaker is open for a model.

    Attributes:
        model_id: The model whose circuit is open.
        retry_after: Seconds until the circuit can be tested again.
    """

    def __init__(
        self,
        model_id: str,
        retry_after: float = 30.0,
        **kwargs: object,
    ) -> None:
        msg = f"Circuit breaker open for model '{model_id}'"
        super().__init__(msg, **kwargs)
        self.model_id = model_id
        self.retry_after = retry_after
