"""Request parsing and routing for proxy server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from routesmith import RouteSmith
from routesmith.config import RoutingStrategy
from routesmith.proxy.responses import (
    format_models_list,
    format_stream_chunk,
    format_stream_done,
)

logger = logging.getLogger(__name__)


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
    def from_dict(cls, data: dict[str, Any]) -> "ChatCompletionRequest":
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

    async def handle_completion(
        self,
        request: ChatCompletionRequest,
    ) -> dict[str, Any]:
        """
        Handle non-streaming chat completion request.

        Routes through RouteSmith and returns OpenAI-compatible response.

        Args:
            request: Parsed completion request.

        Returns:
            OpenAI-compatible response dict.
        """
        # Determine if we should use routing or explicit model
        model = None
        if request.model.lower() not in self.AUTO_MODELS:
            model = request.model

        # Parse RouteSmith-specific options
        strategy = None
        if request.routesmith_strategy:
            try:
                strategy = RoutingStrategy(request.routesmith_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown strategy: {request.routesmith_strategy}")

        # Execute through RouteSmith
        response = await self.routesmith.acompletion(
            messages=request.messages,
            model=model,
            strategy=strategy,
            max_cost=request.routesmith_max_cost,
            min_quality=request.routesmith_min_quality,
            include_metadata=True,
            **request.to_litellm_kwargs(),
        )

        # Convert to dict - litellm responses have model_dump()
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        else:
            # Fallback for older litellm versions
            response_dict = dict(response)

        # Include RouteSmith metadata if present
        if hasattr(response, "routesmith_metadata"):
            response_dict["routesmith_metadata"] = response.routesmith_metadata

        return response_dict

    async def handle_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """
        Handle streaming chat completion.

        Yields Server-Sent Events (SSE) formatted chunks.

        Args:
            request: Parsed completion request.

        Yields:
            SSE-formatted strings for each chunk.
        """
        # Determine if we should use routing or explicit model
        model = None
        if request.model.lower() not in self.AUTO_MODELS:
            model = request.model

        # Parse RouteSmith-specific options
        strategy = None
        if request.routesmith_strategy:
            try:
                strategy = RoutingStrategy(request.routesmith_strategy.lower())
            except ValueError:
                logger.warning(f"Unknown strategy: {request.routesmith_strategy}")

        # Stream through RouteSmith
        stream = self.routesmith.acompletion_stream(
            messages=request.messages,
            model=model,
            strategy=strategy,
            **request.to_litellm_kwargs(),
        )

        async for chunk in stream:
            # Extract content from chunk
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    content = choice.delta.content or ""
                    finish_reason = getattr(choice, "finish_reason", None)
                    model_name = getattr(chunk, "model", request.model)
                    yield format_stream_chunk(content, model_name, finish_reason=finish_reason)

        # Send done marker
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
        Return RouteSmith statistics.

        Returns:
            Stats dict from RouteSmith.
        """
        return self.routesmith.stats

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
