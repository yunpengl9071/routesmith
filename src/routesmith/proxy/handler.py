"""Request parsing and routing for proxy server."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from routesmith import RouteSmith
from routesmith.config import RouteContext, RoutingStrategy
from routesmith.proxy.responses import (
    format_models_list,
    format_stream_chunk,
    format_stream_done,
)

logger = logging.getLogger(__name__)

_ROUTESMITH_HEADERS = {
    "x-routesmith-agent-id": "agent_id",
    "x-routesmith-agent-role": "agent_role",
    "x-routesmith-conversation-id": "conversation_id",
}


def extract_route_context_from_headers(
    headers: dict[str, str],
) -> RouteContext | None:
    """Build RouteContext from X-RouteSmith-* HTTP headers.

    Returns None if no RouteSmith headers are present.
    Header matching is case-insensitive.
    """
    normalized = {k.lower(): v for k, v in headers.items()}
    kwargs: dict[str, str] = {}
    for header, attr in _ROUTESMITH_HEADERS.items():
        if header in normalized:
            kwargs[attr] = normalized[header]
    return RouteContext(**kwargs) if kwargs else None


@dataclass
class ChatCompletionRequest:
    """Parsed OpenAI chat completion request."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = False
    stop: list[str] | str | None = None
    user: str | None = None
    # RouteSmith-specific extensions (passed in request body)
    routesmith_strategy: str | None = None
    routesmith_min_quality: float | None = None
    routesmith_max_cost: float | None = None
    # Extra kwargs to pass through
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatCompletionRequest:
        """
        Parse request from JSON dict.

        Args:
            data: Raw request body as dict.

        Returns:
            Parsed ChatCompletionRequest.

        Raises:
            ValueError: If required fields are missing.
        """
        if "messages" not in data:
            raise ValueError("'messages' is required")

        model = data.get("model", "auto")
        messages = data["messages"]

        # Validate messages format
        if not isinstance(messages, list):
            raise ValueError("'messages' must be a list")
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg:
                raise ValueError("Each message must have a 'role' field")

        # Known fields
        known_fields = {
            "model", "messages", "temperature", "max_tokens", "top_p",
            "frequency_penalty", "presence_penalty", "stream", "stop", "user",
            "routesmith_strategy", "routesmith_min_quality", "routesmith_max_cost",
        }

        # Extract extra kwargs (anything not in known fields)
        extra_kwargs = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            model=model,
            messages=messages,
            temperature=data.get("temperature", 1.0),
            max_tokens=data.get("max_tokens"),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            user=data.get("user"),
            routesmith_strategy=data.get("routesmith_strategy"),
            routesmith_min_quality=data.get("routesmith_min_quality"),
            routesmith_max_cost=data.get("routesmith_max_cost"),
            extra_kwargs=extra_kwargs,
        )

    def to_litellm_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for litellm completion."""
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.user is not None:
            kwargs["user"] = self.user
        # Add any extra kwargs
        kwargs.update(self.extra_kwargs)
        return kwargs


@dataclass
class RoutingDecision:
    route: bool
    passthrough_reason: str | None = None


class RequestHandler:
    """
    Handles incoming HTTP requests and routes through RouteSmith.

    Supports:
    - /v1/chat/completions (POST) - Main completion endpoint
    - /v1/models (GET) - List registered models
    - /v1/stats (GET) - RouteSmith statistics
    - /health (GET) - Health check
    """

    # Model names that trigger intelligent routing
    AUTO_MODELS = {"auto", "routesmith", "routesmith/auto"}

    def __init__(self, routesmith: RouteSmith) -> None:
        """
        Initialize request handler.

        Args:
            routesmith: RouteSmith instance to use for routing and completion.
        """
        self.routesmith = routesmith
        self.routed_requests: int = 0
        self.passthrough_requests: int = 0
        self.passthrough_by_reason: dict[str, int] = {}
        self._passthrough_consecutive: int = 0

    def resolve_routing(
        self,
        request: ChatCompletionRequest,
        headers: dict[str, str],
    ) -> RoutingDecision:
        config = self.routesmith.config
        model_lower = request.model.lower()

        # Auto models always route
        if model_lower in self.AUTO_MODELS:
            self.routed_requests += 1
            self._passthrough_consecutive = 0
            return RoutingDecision(route=True)

        normalized_headers = {k.lower(): v for k, v in headers.items()}

        # Explicit passthrough header
        if normalized_headers.get("x-routesmith-passthrough", "").lower() == "true":
            self.passthrough_requests += 1
            self.passthrough_by_reason["explicit_header"] = self.passthrough_by_reason.get("explicit_header", 0) + 1
            self._passthrough_consecutive += 1
            self._check_passthrough_warning()
            return RoutingDecision(route=False, passthrough_reason="explicit_header")

        # Passthrough models list
        if request.model in config.passthrough_models:
            self.passthrough_requests += 1
            self.passthrough_by_reason["passthrough_list"] = self.passthrough_by_reason.get("passthrough_list", 0) + 1
            self._passthrough_consecutive += 1
            self._check_passthrough_warning()
            return RoutingDecision(route=False, passthrough_reason="passthrough_list")

        # Unregistered model
        registered = {m.model_id for m in self.routesmith.registry.list_models()}
        if request.model not in registered:
            self.passthrough_requests += 1
            self.passthrough_by_reason["unregistered_model"] = self.passthrough_by_reason.get("unregistered_model", 0) + 1
            self._passthrough_consecutive += 1
            self._check_passthrough_warning()
            return RoutingDecision(route=False, passthrough_reason="unregistered_model")

        # Intercept all
        if config.intercept == "all":
            self.routed_requests += 1
            self._passthrough_consecutive = 0
            return RoutingDecision(route=True)

        # Auto intercept, non-auto model -> passthrough
        self.passthrough_requests += 1
        self.passthrough_by_reason["intercept_auto"] = self.passthrough_by_reason.get("intercept_auto", 0) + 1
        self._passthrough_consecutive += 1
        self._check_passthrough_warning()
        return RoutingDecision(route=False, passthrough_reason="intercept_auto")

    def _check_passthrough_warning(self) -> None:
        if self._passthrough_consecutive == 5 and self.routesmith.config.intercept == "auto":
            logger.warning(
                "5 requests passed through unrouted "
                "— set routing.intercept: all to route them"
            )

    async def handle_completion(
        self,
        request: ChatCompletionRequest,
        headers: dict[str, str] | None = None,
        decision: RoutingDecision | None = None,
    ) -> dict[str, Any]:
        """
        Handle non-streaming chat completion request.

        Routes through RouteSmith and returns OpenAI-compatible response.

        Args:
            request: Parsed completion request.
            headers: Optional HTTP request headers; X-RouteSmith-* values are
                     extracted and forwarded as a RouteContext.
            decision: Pre-computed routing decision (avoids double resolution).

        Returns:
            OpenAI-compatible response dict.
        """
        if decision is None:
            decision = self.resolve_routing(request, headers or {})

        model = None if decision.route else request.model

        strategy = None
        if request.routesmith_strategy:
            try:
                strategy = RoutingStrategy(request.routesmith_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown strategy: {request.routesmith_strategy}")

        ctx = extract_route_context_from_headers(headers or {})

        response = await self.routesmith.acompletion(
            messages=request.messages,
            model=model,
            strategy=strategy,
            max_cost=request.routesmith_max_cost,
            min_quality=request.routesmith_min_quality,
            include_metadata=True,
            context=ctx,
            **request.to_litellm_kwargs(),
        )

        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            response_dict = dict(response)

        selected_model = response_dict.get("model", request.model)
        routesmith_metadata = dict(response.routesmith_metadata) if hasattr(response, "routesmith_metadata") and response.routesmith_metadata else {}
        routesmith_metadata.update({
            "routed": decision.route,
            "selected_model": selected_model,
            "requested_model": request.model,
            "passthrough_reason": decision.passthrough_reason,
        })
        response_dict["routesmith_metadata"] = routesmith_metadata

        return response_dict

    async def handle_completion_stream(
        self,
        request: ChatCompletionRequest,
        headers: dict[str, str] | None = None,
        decision: RoutingDecision | None = None,
    ) -> AsyncIterator[str]:
        """
        Handle streaming chat completion.

        Yields Server-Sent Events (SSE) formatted chunks.

        Args:
            request: Parsed completion request.
            headers: Optional HTTP request headers for routing decisions.
            decision: Pre-computed routing decision (avoids double resolution).

        Yields:
            SSE-formatted strings for each chunk.
        """
        if decision is None:
            decision = self.resolve_routing(request, headers or {})

        model = None if decision.route else request.model

        strategy = None
        if request.routesmith_strategy:
            try:
                strategy = RoutingStrategy(request.routesmith_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown strategy: {request.routesmith_strategy}")

        stream = self.routesmith.acompletion_stream(
            messages=request.messages,
            model=model,
            strategy=strategy,
            **request.to_litellm_kwargs(),
        )

        selected_model = request.model
        async for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    content = choice.delta.content or ""
                    finish_reason = getattr(choice, "finish_reason", None)
                    model_name = getattr(chunk, "model", request.model)
                    selected_model = model_name
                    yield format_stream_chunk(content, model_name, finish_reason=finish_reason)

        routesmith_metadata = {
            "routed": decision.route,
            "selected_model": selected_model,
            "requested_model": request.model,
            "passthrough_reason": decision.passthrough_reason,
        }
        yield format_stream_chunk("", request.model, finish_reason=None, routesmith_metadata=routesmith_metadata)
        yield format_stream_done()

    async def handle_models(self) -> dict[str, Any]:
        """
        Return list of registered models.

        Returns:
            OpenAI-compatible /v1/models response.
        """
        models = self.routesmith.registry.list_models()
        model_list = [{"id": m.model_id} for m in models]
        # Add the auto model
        model_list.append({"id": "routesmith/auto", "owned_by": "routesmith"})
        return format_models_list(model_list)

    async def handle_stats(self) -> dict[str, Any]:
        """
        Return RouteSmith statistics including proxy-level counters.

        Returns:
            Stats dict from RouteSmith merged with proxy counters.
        """
        stats = dict(self.routesmith.stats)
        stats["routed_requests"] = self.routed_requests
        stats["passthrough_requests"] = self.passthrough_requests
        stats["passthrough_by_reason"] = dict(self.passthrough_by_reason)
        return stats

    async def handle_health(self) -> dict[str, Any]:
        """
        Return health check response.

        Returns:
            Health status dict.
        """
        return {
            "status": "healthy",
            "registered_models": len(self.routesmith.registry),
        }

    async def handle_feedback(self, body: bytes) -> tuple[dict, int]:
        """Process an outcome report. Returns (response_dict, http_status).

        Request JSON:
          request_id  str, required — from routesmith_metadata.request_id
          score       float in [0,1], optional
          success     bool, optional (exactly one of score/success required)
        """
        import json
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, 400

        request_id = data.get("request_id")
        score = data.get("score")
        success = data.get("success")
        if not isinstance(request_id, str) or not request_id:
            return {"error": {"message": "'request_id' (string) is required",
                              "type": "invalid_request_error"}}, 400
        if (score is None) == (success is None):
            return {"error": {"message": "Provide exactly one of 'score' or 'success'",
                              "type": "invalid_request_error"}}, 400
        if score is not None and not (isinstance(score, (int, float)) and 0.0 <= score <= 1.0):
            return {"error": {"message": "'score' must be a number in [0, 1]",
                              "type": "invalid_request_error"}}, 400

        found = self.routesmith.record_outcome(request_id=request_id, score=score, success=success)
        if not found:
            return {"error": {"message": f"Unknown request_id: {request_id}",
                              "type": "not_found"}}, 404
        return {"status": "ok", "request_id": request_id}, 200

    async def handle_liveness(self) -> dict[str, Any]:
        """
        Return liveness probe response.
        Returns status 'alive' if the process is running.
        """
        return {"status": "alive"}

    async def handle_readiness(self) -> dict[str, Any]:
        """
        Return readiness probe response.
        Returns status 'ready' only when models are registered.
        """
        models = self.routesmith.registry.list_models()
        if not models:
            return {"status": "not_ready", "reason": "no models registered"}
        return {"status": "ready", "models": len(models)}
