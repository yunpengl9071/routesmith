"""OpenAI-compatible response formatting."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


def format_chat_completion(
    content: str,
    model: str,
    usage: dict[str, int],
    routesmith_metadata: dict[str, Any] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """
    Format a chat completion response in OpenAI format.

    Args:
        content: The assistant's response content.
        model: Model identifier used.
        usage: Token usage dict with prompt_tokens, completion_tokens, total_tokens.
        routesmith_metadata: Optional routing metadata to include.
        finish_reason: Reason for completion (default: "stop").

    Returns:
        OpenAI-compatible chat completion response dict.
    """
    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }

    if routesmith_metadata:
        response["routesmith_metadata"] = routesmith_metadata

    return response


def format_stream_chunk(
    content: str,
    model: str,
    chunk_id: str | None = None,
    finish_reason: str | None = None,
) -> str:
    """
    Format a streaming chunk as Server-Sent Event (SSE).

    Args:
        content: The content delta for this chunk.
        model: Model identifier used.
        chunk_id: Optional chunk ID (generated if not provided).
        finish_reason: Set to "stop" for final chunk, None otherwise.

    Returns:
        SSE-formatted string with "data: " prefix.
    """
    chunk: dict[str, Any] = {
        "id": chunk_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def format_stream_done() -> str:
    """Format the final SSE done message."""
    return "data: [DONE]\n\n"


def format_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    status_code: int = 400,
) -> tuple[dict[str, Any], int]:
    """
    Format an error response in OpenAI format.

    Args:
        message: Human-readable error message.
        error_type: Error type (e.g., "invalid_request_error").
        code: Optional error code.
        status_code: HTTP status code.

    Returns:
        Tuple of (error response dict, HTTP status code).
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }, status_code


def format_models_list(models: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Format a models list response in OpenAI format.

    Args:
        models: List of model info dicts with "id" and optional metadata.

    Returns:
        OpenAI-compatible models list response.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.get("owned_by", "routesmith"),
                "permission": [],
                "root": m["id"],
                "parent": None,
            }
            for m in models
        ],
    }
