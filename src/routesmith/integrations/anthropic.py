"""Anthropic SDK integration for RouteSmith.

Provides RouteSmithAnthropic, a drop-in replacement for anthropic.Anthropic
that routes messages through RouteSmith's adaptive routing engine.

Example:
    >>> from routesmith.integrations.anthropic import RouteSmithAnthropic
    >>> client = RouteSmithAnthropic.with_openrouter_models()
    >>> msg = client.messages.create(
    ...     model="auto",
    ...     max_tokens=1024,
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ... )
    >>> print(msg.content[0].text)
"""

from __future__ import annotations

from typing import Any

try:
    import anthropic
    from anthropic.types import Message, TextBlock, Usage
except ImportError as e:
    raise ImportError(
        "Anthropic integration requires the anthropic package. "
        "Install it with: pip install routesmith[anthropic]"
    ) from e

from routesmith.client import RouteSmith
from routesmith.config import RouteSmithConfig, PredictorConfig


def _anthropic_to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic message format to OpenAI format."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Anthropic content blocks → flatten to text for routing
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
            result.append({"role": role, "content": " ".join(text_parts)})
        else:
            result.append({"role": role, "content": str(content)})
    return result


def _litellm_to_anthropic_message(response: Any, model: str) -> Message:
    """Convert a LiteLLM ModelResponse to an anthropic.types.Message."""
    choice = response.choices[0]
    text = choice.message.content or ""
    usage = getattr(response, "usage", None)

    # Map OpenAI finish_reason → Anthropic stop_reason literal
    _STOP_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "refusal",
    }
    stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, "end_turn")

    return Message(
        id=getattr(response, "id", "msg_routesmith"),
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text=text)],
        model=model,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        ),
    )


class _MessagesResource:
    """Mimics anthropic.resources.Messages."""

    def __init__(self, rs: RouteSmith) -> None:
        self._rs = rs

    def create(
        self,
        *,
        model: str = "auto",
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Message:
        """Create a message, routing through RouteSmith."""
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(_anthropic_to_openai_messages(messages))

        extra: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            extra["temperature"] = temperature

        response = self._rs.completion(messages=openai_messages, **extra)
        selected = getattr(self._rs._last_routing_metadata, "model_selected", model)
        return _litellm_to_anthropic_message(response, selected)


class RouteSmithAnthropic:
    """Drop-in replacement for anthropic.Anthropic that routes via RouteSmith.

    Args:
        routesmith: Pre-configured RouteSmith instance. If None, creates one.
        config: RouteSmithConfig to use when creating a new instance.

    Example:
        client = RouteSmithAnthropic()
        client.register_model("anthropic/claude-3-haiku", ...)
        client.register_model("anthropic/claude-3-5-sonnet", ...)
        msg = client.messages.create(model="auto", max_tokens=512,
                                     messages=[{"role": "user", "content": "Hi"}])
    """

    def __init__(
        self,
        routesmith: RouteSmith | None = None,
        config: RouteSmithConfig | None = None,
    ) -> None:
        self._rs = routesmith or RouteSmith(config=config)
        self.messages = _MessagesResource(self._rs)

    # Delegate registry helpers so callers can do client.register_model(...)
    def register_model(self, model_id: str, **kwargs: Any) -> None:
        self._rs.register_model(model_id, **kwargs)

    @property
    def stats(self) -> dict[str, Any]:
        return self._rs.stats

    @classmethod
    def with_openrouter_models(
        cls,
        model_ids: list[str] | None = None,
        predictor_type: str = "lints",
    ) -> "RouteSmithAnthropic":
        """Create a client pre-loaded with Claude models from OpenRouter.

        Args:
            model_ids: Specific model IDs to use. Defaults to a curated
                Claude tier set (haiku → sonnet → opus).
            predictor_type: Routing algorithm — 'lints' or 'linucb'.
        """
        from routesmith.registry.openrouter import fetch_pricing

        default_models = [
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-5-sonnet",
            "anthropic/claude-3-opus",
        ]
        ids = model_ids or default_models
        pricing = fetch_pricing(ids)

        config = RouteSmithConfig(
            predictor_type=predictor_type,
            predictor=PredictorConfig(),
        )
        instance = cls(config=config)
        for mid in ids:
            if mid in pricing:
                m = pricing[mid]
                instance.register_model(
                    mid,
                    cost_per_1k_input=m.cost_per_1k_input,
                    cost_per_1k_output=m.cost_per_1k_output,
                    quality_score=m.quality_score,
                    context_window=m.context_window,
                )
        return instance
