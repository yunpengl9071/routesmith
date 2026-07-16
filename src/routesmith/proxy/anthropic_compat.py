"""Anthropic Messages API compatibility layer for the RouteSmith proxy.

Translates between Anthropic's /v1/messages format and OpenAI-style
internal format so any Anthropic SDK client can route through RouteSmith
by setting ANTHROPIC_BASE_URL.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "refusal",
}

_TOOL_CHOICE_MAP: dict[str, str] = {
    "auto": "auto",
    "any": "required",
}


def _convert_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def _convert_tool_choice(tool_choice: dict) -> str | dict:
    tc_type = tool_choice.get("type", "auto")
    mapped = _TOOL_CHOICE_MAP.get(tc_type)
    if mapped is not None:
        return mapped
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    raise ValueError(f"Unsupported tool_choice type: {tc_type}")


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "".join(texts)
    return str(content)


def _convert_assistant_blocks(blocks: list[dict]) -> list[dict]:
    text_parts: list[str] = []
    tool_uses: list[dict] = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_uses.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        elif btype in ("thinking", "redacted_thinking"):
            continue
        else:
            raise ValueError(f"Unsupported content block type '{btype}'")
    result: list[dict] = []
    if text_parts:
        result.append({"role": "assistant", "content": "\n".join(text_parts)})
    if tool_uses:
        result.append({"role": "assistant", "content": None, "tool_calls": tool_uses})
    return result


def _convert_user_blocks(blocks: list[dict]) -> list[dict]:
    tool_msgs: list[dict] = []
    text_parts: list[str] = []
    image_parts: list[dict] = []
    for block in blocks:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_result":
            flat = _flatten_content(block.get("content", ""))
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": flat,
            })
        elif btype == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                image_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source['media_type']};base64,{source['data']}",
                    },
                })
        else:
            raise ValueError(f"Unsupported content block type '{btype}'")
    result: list[dict] = []
    result.extend(tool_msgs)
    if text_parts and not image_parts:
        result.append({"role": "user", "content": "\n".join(text_parts)})
    elif text_parts or image_parts:
        parts: list[dict] = []
        for t in text_parts:
            parts.append({"type": "text", "text": t})
        parts.extend(image_parts)
        result.append({"role": "user", "content": parts})
    return result


def anthropic_to_internal(data: dict[str, Any]) -> tuple[list[dict], dict]:
    if "messages" not in data:
        raise ValueError("'messages' is required")
    if "max_tokens" not in data:
        raise ValueError("'max_tokens' is required (Anthropic API)")

    max_tokens = data["max_tokens"]
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("'max_tokens' must be a positive integer")

    messages: list[dict] = []

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

    for msg in data["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            if role == "assistant":
                messages.extend(_convert_assistant_blocks(content))
            elif role == "user":
                messages.extend(_convert_user_blocks(content))
            else:
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype in ("thinking", "redacted_thinking"):
                        continue
                    else:
                        raise ValueError(f"Unsupported content block type '{btype}' in {role} message")
                messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            raise ValueError(f"Unexpected content type: {type(content).__name__}")

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

    tools = data.get("tools")
    if tools:
        kwargs["tools"] = _convert_tools(tools)

    tool_choice = data.get("tool_choice")
    if tool_choice:
        kwargs["tool_choice"] = _convert_tool_choice(tool_choice)

    return messages, kwargs


def internal_to_anthropic(
    response: Any,
    request_model: str,
    request_id: str = "",
) -> dict:
    choice = response.choices[0]
    content_text = choice.message.content or ""
    finish = choice.finish_reason or "stop"
    tool_calls = getattr(choice.message, "tool_calls", None)

    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    content_blocks: list[dict] = []
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    if tool_calls:
        for tc in tool_calls:
            args_raw = getattr(tc.function, "arguments", "{}")
            try:
                parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                logger.warning("Malformed JSON arguments in tool_call %s: %s", tc.id, args_raw[:200])
                parsed_args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": parsed_args,
            })

    return {
        "id": f"msg_{request_id}" if request_id else "msg_unknown",
        "type": "message",
        "role": "assistant",
        "model": getattr(response, "model", request_model),
        "content": content_blocks,
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "routesmith_metadata": getattr(response, "routesmith_metadata", None),
    }


class AnthropicSSEStream:
    def __init__(self, request_id: str, request_model: str, routesmith_metadata: dict | None = None) -> None:
        self._request_id = request_id
        self._request_model = request_model
        self._routesmith_metadata = routesmith_metadata
        self._started = False

    def iter_chunks(self, chunks: list[dict]) -> list[str]:
        events: list[str] = []
        full_text = ""
        tool_states: dict[int, dict] = {}
        next_tool_block = 1
        has_text = False

        for chunk_data in chunks:
            choice = chunk_data.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            finish = choice.get("finish_reason")
            content = delta.get("content", "")
            tool_calls_delta = delta.get("tool_calls")

            if not self._started:
                content_blocks: list[dict] = []

                if tool_calls_delta:
                    for tc in tool_calls_delta:
                        tc_idx = tc.get("index", 0)
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("function", {}).get("name", "")
                        tc_args = tc.get("function", {}).get("arguments", "")
                        tool_states[tc_idx] = {
                            "block_index": next_tool_block,
                            "id": tc_id,
                            "name": tc_name,
                            "args_buffer": tc_args,
                        }
                        next_tool_block += 1
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input": {},
                        })

                has_text = bool(content) or not tool_calls_delta
                if has_text:
                    content_blocks.insert(0, {"type": "text", "text": ""})

                events.append(self._event("message_start", {
                    "type": "message",
                    "id": f"msg_{self._request_id}",
                    "role": "assistant",
                    "model": self._request_model,
                    "content": content_blocks,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }))

                if has_text:
                    events.append(self._event("content_block_start", {
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }))

                for tc_idx, state in sorted(tool_states.items(), key=lambda x: x[1]["block_index"]):
                    events.append(self._event("content_block_start", {
                        "index": state["block_index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": state["id"],
                            "name": state["name"],
                            "input": {},
                        },
                    }))

                self._started = True

                if content:
                    events.append(self._event("content_block_delta", {
                        "index": 0,
                        "delta": {"type": "text_delta", "text": content},
                    }))
                    full_text += content

            elif content:
                events.append(self._event("content_block_delta", {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                }))
                full_text += content

            elif tool_calls_delta:
                for tc in tool_calls_delta:
                    tc_idx = tc.get("index", 0)
                    args_fragment = tc.get("function", {}).get("arguments", "")

                    if tc_idx not in tool_states:
                        new_block = next_tool_block
                        next_tool_block += 1
                        tc_id = tc.get("id", "")
                        tc_name = tc.get("function", {}).get("name", "")
                        tool_states[tc_idx] = {
                            "block_index": new_block,
                            "id": tc_id,
                            "name": tc_name,
                            "args_buffer": args_fragment,
                        }
                        events.append(self._event("content_block_start", {
                            "index": new_block,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": {},
                            },
                        }))
                    else:
                        state = tool_states[tc_idx]
                        state["args_buffer"] += args_fragment
                        events.append(self._event("content_block_delta", {
                            "index": state["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_fragment,
                            },
                        }))

            if finish:
                stop_reason = _STOP_REASON_MAP.get(finish, "end_turn")
                output_tokens = max(1, len(full_text) // 4)

                if has_text:
                    events.append(self._event("content_block_stop", {"index": 0}))

                for tc_idx, state in sorted(tool_states.items(), key=lambda x: x[1]["block_index"]):
                    events.append(self._event("content_block_stop", {"index": state["block_index"]}))

                if chunk_data.get("usage"):
                    usage_data = chunk_data["usage"]
                    output_tokens = usage_data.get("completion_tokens", output_tokens)

                events.append(self._event("message_delta", {
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                }))

                if self._routesmith_metadata:
                    events.append(self._event("routesmith_metadata", self._routesmith_metadata))
                events.append(self._event("message_stop", {}))

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
