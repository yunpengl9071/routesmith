"""LangChain integration for RouteSmith.

Provides ChatRouteSmith, a BaseChatModel subclass that routes queries
through RouteSmith's adaptive routing engine. Works as a drop-in
replacement anywhere LangChain expects a ChatModel.

Example:
    >>> from routesmith.integrations.langchain import ChatRouteSmith
    >>> llm = ChatRouteSmith.with_openai_models()
    >>> llm.invoke("What is 2+2?")
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Iterator, AsyncIterator

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models import BaseChatModel, LanguageModelInput
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.messages.tool import ToolCall, ToolCallChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with: pip install routesmith[langchain]"
    ) from e

from pydantic import ConfigDict

from routesmith.client import RouteSmith


def _langchain_messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain message objects to OpenAI-format dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
        else:
            # Fallback for unknown message types
            result.append({"role": msg.type, "content": msg.content})
    return result


def _litellm_response_to_ai_message(
    response: Any, routing_meta: dict[str, Any] | None = None
) -> AIMessage:
    """Convert a litellm ModelResponse to a LangChain AIMessage."""
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""

    # Parse tool calls if present
    tool_calls: list[ToolCall] = []
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(
                ToolCall(name=tc.function.name, args=args or {}, id=tc.id)
            )

    # Build response metadata
    response_metadata: dict[str, Any] = {}
    if hasattr(response, "model") and response.model:
        response_metadata["model_name"] = response.model
    if hasattr(choice, "finish_reason") and choice.finish_reason:
        response_metadata["finish_reason"] = choice.finish_reason
    if hasattr(response, "usage") and response.usage:
        response_metadata["token_usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    if routing_meta:
        response_metadata["routesmith"] = routing_meta
        if "request_id" in routing_meta:
            response_metadata["routesmith_request_id"] = routing_meta["request_id"]

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        response_metadata=response_metadata,
    )


def _litellm_chunk_to_generation_chunk(chunk: Any) -> ChatGenerationChunk | None:
    """Convert a litellm streaming chunk to a LangChain ChatGenerationChunk."""
    if not chunk.choices:
        return None

    delta = chunk.choices[0].delta
    content = delta.content or ""

    # Parse streaming tool call chunks
    tool_call_chunks: list[ToolCallChunk] = []
    if hasattr(delta, "tool_calls") and delta.tool_calls:
        for tc in delta.tool_calls:
            tool_call_chunks.append(
                ToolCallChunk(
                    name=tc.function.name if tc.function and tc.function.name else None,
                    args=tc.function.arguments if tc.function and tc.function.arguments else None,
                    id=tc.id,
                    index=tc.index,
                )
            )

    msg = AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks)
    return ChatGenerationChunk(message=msg)


class ChatRouteSmith(BaseChatModel):
    """LangChain ChatModel backed by RouteSmith's adaptive routing.

    Routes queries to optimal models based on cost, quality, and latency
    constraints. Works anywhere a LangChain ChatModel is accepted:
    create_react_agent, supervisor, swarm, or raw invoke().

    Example:
        >>> from routesmith.integrations.langchain import ChatRouteSmith
        >>> llm = ChatRouteSmith()
        >>> llm.routesmith.register_model(
        ...     "gpt-4o-mini",
        ...     cost_per_1k_input=0.00015,
        ...     cost_per_1k_output=0.0006,
        ... )
        >>> result = llm.invoke("Hello!")
    """

    routesmith: Any = None
    strategy: str | None = None
    min_quality: float | None = None
    max_cost: float | None = None
    include_routing_metadata: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.routesmith is None:
            self.routesmith = RouteSmith()

    @property
    def _llm_type(self) -> str:
        return "routesmith"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools to the model.

        Converts tools to OpenAI format and passes them as kwargs
        through to litellm.completion(), which handles them natively.
        """
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        bind_kwargs: dict[str, Any] = {"tools": formatted_tools, **kwargs}
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        return self.bind(**bind_kwargs)

    def _build_completion_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Build kwargs for routesmith.completion() from instance + per-call overrides."""
        completion_kwargs: dict[str, Any] = {}
        if self.strategy is not None:
            completion_kwargs["strategy"] = self.strategy
        if self.min_quality is not None:
            completion_kwargs["min_quality"] = self.min_quality
        if self.max_cost is not None:
            completion_kwargs["max_cost"] = self.max_cost
        completion_kwargs["include_metadata"] = self.include_routing_metadata
        # Per-call kwargs override instance defaults
        completion_kwargs.update(kwargs)
        return completion_kwargs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg_dicts = _langchain_messages_to_dicts(messages)
        completion_kwargs = self._build_completion_kwargs(**kwargs)
        if stop:
            completion_kwargs["stop"] = stop

        response = self.routesmith.completion(messages=msg_dicts, **completion_kwargs)

        routing_meta = getattr(response, "routesmith_metadata", None)
        ai_message = _litellm_response_to_ai_message(response, routing_meta)

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg_dicts = _langchain_messages_to_dicts(messages)
        completion_kwargs = self._build_completion_kwargs(**kwargs)
        if stop:
            completion_kwargs["stop"] = stop

        response = await self.routesmith.acompletion(messages=msg_dicts, **completion_kwargs)

        routing_meta = getattr(response, "routesmith_metadata", None)
        ai_message = _litellm_response_to_ai_message(response, routing_meta)

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg_dicts = _langchain_messages_to_dicts(messages)
        stream_kwargs: dict[str, Any] = {}
        if self.strategy is not None:
            stream_kwargs["strategy"] = self.strategy
        if stop:
            stream_kwargs["stop"] = stop
        stream_kwargs.update(kwargs)

        for chunk in self.routesmith.completion_stream(messages=msg_dicts, **stream_kwargs):
            gen_chunk = _litellm_chunk_to_generation_chunk(chunk)
            if gen_chunk is not None:
                if run_manager and gen_chunk.text:
                    run_manager.on_llm_new_token(gen_chunk.text)
                yield gen_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        msg_dicts = _langchain_messages_to_dicts(messages)
        stream_kwargs: dict[str, Any] = {}
        if self.strategy is not None:
            stream_kwargs["strategy"] = self.strategy
        if stop:
            stream_kwargs["stop"] = stop
        stream_kwargs.update(kwargs)

        async for chunk in self.routesmith.acompletion_stream(
            messages=msg_dicts, **stream_kwargs
        ):
            gen_chunk = _litellm_chunk_to_generation_chunk(chunk)
            if gen_chunk is not None:
                if run_manager and gen_chunk.text:
                    await run_manager.on_llm_new_token(gen_chunk.text)
                yield gen_chunk

    def record_outcome(
        self,
        request_id: str,
        success: bool | None = None,
        score: float | None = None,
        feedback: str | None = None,
    ) -> bool:
        """Record feedback for a previous request to improve future routing.

        Args:
            request_id: From response_metadata["routesmith_request_id"].
            success: Whether the response was successful.
            score: Quality score (0-1).
            feedback: Free-text feedback.

        Returns:
            True if the request was found.
        """
        return self.routesmith.record_outcome(
            request_id=request_id,
            success=success,
            score=score,
            feedback=feedback,
        )

    @classmethod
    def with_openai_models(cls, **kwargs: Any) -> ChatRouteSmith:
        """Create a ChatRouteSmith pre-configured with common OpenAI models."""
        instance = cls(**kwargs)
        _all_caps = {"tool_calling", "vision", "json_mode", "streaming"}
        models: list[tuple[str, float, float, float, set[str]]] = [
            ("gpt-4o", 0.0025, 0.01, 0.95, _all_caps),
            ("gpt-4o-mini", 0.00015, 0.0006, 0.85, {"tool_calling", "json_mode", "streaming"}),
            ("gpt-4.1", 0.002, 0.008, 0.96, _all_caps),
            ("gpt-4.1-mini", 0.0004, 0.0016, 0.87, {"tool_calling", "json_mode", "streaming"}),
            ("gpt-4.1-nano", 0.0001, 0.0004, 0.75, {"json_mode", "streaming"}),
        ]
        for model_id, inp, out, quality, caps in models:
            instance.routesmith.register_model(
                model_id, cost_per_1k_input=inp, cost_per_1k_output=out,
                quality_score=quality,
                supports_function_calling="tool_calling" in caps,
                supports_vision="vision" in caps,
                supports_json_mode="json_mode" in caps,
            )
        return instance

    @classmethod
    def with_groq_models(cls, **kwargs: Any) -> ChatRouteSmith:
        """Create a ChatRouteSmith pre-configured with common Groq models."""
        instance = cls(**kwargs)
        models: list[tuple[str, float, float, float, set[str]]] = [
            ("groq/llama-3.3-70b-versatile", 0.00059, 0.00079, 0.90, {"tool_calling", "json_mode", "streaming"}),
            ("groq/llama-3.1-8b-instant", 0.00005, 0.00008, 0.75, {"streaming"}),
        ]
        for model_id, inp, out, quality, caps in models:
            instance.routesmith.register_model(
                model_id, cost_per_1k_input=inp, cost_per_1k_output=out,
                quality_score=quality,
                supports_function_calling="tool_calling" in caps,
                supports_vision="vision" in caps,
                supports_json_mode="json_mode" in caps,
            )
        return instance

    @classmethod
    def with_anthropic_models(cls, **kwargs: Any) -> ChatRouteSmith:
        """Create a ChatRouteSmith pre-configured with common Anthropic models."""
        instance = cls(**kwargs)
        _all_caps = {"tool_calling", "vision", "json_mode", "streaming"}
        models: list[tuple[str, float, float, float, set[str]]] = [
            ("anthropic/claude-opus-4-6", 0.015, 0.075, 0.97, _all_caps),
            ("anthropic/claude-sonnet-4-5-20250929", 0.003, 0.015, 0.93, _all_caps),
            ("anthropic/claude-haiku-4-5-20251001", 0.0008, 0.004, 0.85, _all_caps),
        ]
        for model_id, inp, out, quality, caps in models:
            instance.routesmith.register_model(
                model_id, cost_per_1k_input=inp, cost_per_1k_output=out,
                quality_score=quality,
                supports_function_calling="tool_calling" in caps,
                supports_vision="vision" in caps,
                supports_json_mode="json_mode" in caps,
            )
        return instance
