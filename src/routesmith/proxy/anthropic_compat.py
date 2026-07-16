"""Anthropic Messages API compatibility layer for the RouteSmith proxy.

Translates between Anthropic's /v1/messages format and OpenAI-style
internal format so any Anthropic SDK client can route through RouteSmith
by setting ANTHROPIC_BASE_URL.
"""

from __future__ import annotations

import json
from typing import Any

# Reuse the stop-reason map from the Anthropic SDK integration
_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}


def anthropic_to_internal(data: dict[str, Any]) -> tuple[list[dict], dict]:
    """Convert an Anthropic /v1/messages request to OpenAI-format messages + kwargs.

    Args:
        data: Raw Anthropic Messages API request body.

    Returns:
        (openai_style_messages, kwargs_dict)

    Raises:
        ValueError: On malformed input.
    """
    if "messages" not in data:
        raise ValueError("'messages' is required")
    if "max_tokens" not in data:
        raise ValueError("'max_tokens' is required (Anthropic API)")

    max_tokens = data["max_tokens"]
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("'max_tokens' must be a positive integer")

    messages: list[dict] = []

    # System prompt: convert to a system-role message
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            texts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                else:
                    raise ValueError(f"Unsupported system block type: {block.get('type', 'unknown')}")
            messages.append({"role": "system", "content": "\n".join(texts)})

    # Convert messages
    for msg in data["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                else:
                    raise ValueError(
                        f"Unsupported content block type '{btype}' in /v1/messages "
                        f"(text-only supported in v1)"
                    )
            messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            raise ValueError(f"Unexpected content type: {type(content).__name__}")

    # Build kwargs
    kwargs: dict[str, Any] = {"max_tokens": max_tokens}

    model = data.get("model", "auto")
    kwargs["model"] = model

    if "temperature" in data:
        kwargs["temperature"] = data["temperature"]
    if "top_p" in data:
        kwargs["top_p"] = data["top_p"]
    stop_sequences = data.get("stop_sequences")
    if stop_sequences:
        kwargs["stop"] = stop_sequences if len(stop_sequences) > 1 else stop_sequences[0]
    if data.get("stream"):
        kwargs["stream"] = True

    return messages, kwargs


def internal_to_anthropic(
    response: Any,
    request_model: str,
    request_id: str = "",
) -> dict:
    """Convert a LiteLLM/OpenAI completion response to Anthropic format.

    Args:
        response: Litellm ModelResponse (or similar) with choices/usage.
        request_model: Original model from the Anthropic request.
        request_id: Request ID for the response id field.

    Returns:
        Anthropic Messages API response dict.
    """
    choice = response.choices[0]
    content_text = choice.message.content or ""
    finish = choice.finish_reason or "stop"

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    return {
        "id": f"msg_{request_id}" if request_id else "msg_unknown",
        "type": "message",
        "role": "assistant",
        "model": getattr(response, "model", request_model),
        "content": [{"type": "text", "text": content_text}],
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "routesmith_metadata": getattr(response, "routesmith_metadata", None),
    }


class AnthropicSSEStream:
    """Build Anthropic-format SSE event strings from streamed chunks.

    Emits: message_start → content_block_start → content_block_delta+ →
           content_block_stop → message_delta → [routesmith_metadata] → message_stop
    """

    def __init__(self, request_id: str, request_model: str, routesmith_metadata: dict | None = None) -> None:
        self._request_id = request_id
        self._request_model = request_model
        self._routesmith_metadata = routesmith_metadata
        self._started = False

    def iter_chunks(self, chunks: list[dict]) -> list[str]:
        """Convert a collected list of OpenAI-format stream chunks to SSE events."""
        events: list[str] = []
        full_text = ""

        for chunk_data in chunks:
            choice = chunk_data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason")
            content = delta.get("content", "")

            if not self._started:
                events.extend(self._build_start(content))
                self._started = True
            elif content:
                events.extend(self._build_delta(content))
            full_text += content

            if finish:
                events.extend(self._build_stop(finish, full_text))

        if not self._started:
            events.extend(self._build_start(""))
            events.extend(self._build_stop("stop", ""))

        return events

    def _build_start(self, initial_text: str) -> list[str]:
        content = {"type": "text", "text": initial_text}
        return [
            self._event("message_start", {
                "type": "message",
                "id": f"msg_{self._request_id}",
                "role": "assistant",
                "model": self._request_model,
                "content": [content],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }),
            self._event("content_block_start", {
                "index": 0,
                "content_block": content,
            }),
        ]

    def _build_delta(self, text: str) -> list[str]:
        return [self._event("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        })]

    def _build_stop(self, finish: str, full_text: str) -> list[str]:
        stop_reason = _STOP_REASON_MAP.get(finish, "end_turn")
        output_tokens = max(1, len(full_text) // 4)
        events = [
            self._event("content_block_stop", {"index": 0}),
            self._event("message_delta", {
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }),
        ]
        if self._routesmith_metadata:
            events.append(self._event("routesmith_metadata", self._routesmith_metadata))
        events.append(self._event("message_stop", {}))
        return events

    @staticmethod
    def _event(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"
