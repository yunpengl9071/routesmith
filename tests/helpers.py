"""Shared test helpers. No routesmith imports at module level."""
from __future__ import annotations

from types import SimpleNamespace


def fake_response(
    content: str = "ok",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    tool_calls: list | None = None,
):
    """Build an object that quacks like a litellm ModelResponse.

    Args:
        tool_calls: Optional list of SimpleNamespace with id/function.name/function.arguments.
                    Each item should be: SimpleNamespace(id="...", function=SimpleNamespace(name="...", arguments="{}"))
    """
    message = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    resp = SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model,
        id="chatcmpl-test",
    )
    resp.model_dump = lambda: {
        "id": resp.id,
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return resp


def make_rs(**config_kwargs):
    """RouteSmith with two standard test models registered.

    gpt-4o:      expensive, quality 0.95
    gpt-4o-mini: cheap,     quality 0.85
    """
    from routesmith import RouteSmith
    from routesmith.config import RouteSmithConfig

    rs = RouteSmith(config=RouteSmithConfig(**config_kwargs))
    rs.register_model(
        "gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        quality_score=0.95,
    )
    rs.register_model(
        "gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        quality_score=0.85,
    )
    return rs
