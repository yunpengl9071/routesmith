"""OpenAI-compatible HTTP server for RouteSmith."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx

from routesmith import RouteSmith
from routesmith.budget import BudgetExceededError
from routesmith.exceptions import NoCapableModelError
from routesmith.proxy.anthropic_compat import (
    _STOP_REASON_MAP,
    AnthropicSSEStream,
    anthropic_to_internal,
)
from routesmith.proxy.handler import ChatCompletionRequest, RequestHandler
from routesmith.proxy.responses import format_error

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_HEADERS_TO_STRIP = {"host", "content-length", "transfer-encoding", "connection"}


@dataclass
class ServerConfig:
    """Configuration for the proxy server."""

    host: str = "127.0.0.1"
    port: int = 9119
    read_timeout: float = 30.0
    max_request_size: int = 10 * 1024 * 1024
    api_key: str | None = None


class RouteSmithProxyServer:
    def __init__(
        self,
        routesmith: RouteSmith,
        config: ServerConfig | None = None,
    ) -> None:
        self.routesmith = routesmith
        self.config = config or ServerConfig()
        self.handler = RequestHandler(routesmith)
        self._server: asyncio.Server | None = None
        self._running = False

    @property
    def host(self) -> str:
        return self.config.host

    @property
    def port(self) -> int:
        if self._server and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self.config.port

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.config.host,
            self.config.port,
        )
        self._running = True
        logger.info(f"Server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Server stopped")

    async def serve_forever(self) -> None:
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        try:
            while self._running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request_line = await asyncio.wait_for(
                reader.readline(),
                timeout=self.config.read_timeout,
            )
            if not request_line:
                return

            request_line = request_line.decode("utf-8").strip()
            parts = request_line.split(" ")
            if len(parts) < 2:
                await self._send_error(writer, "Bad request", 400)
                return

            method = parts[0].upper()
            path = parts[1]

            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                line = line.decode("utf-8").strip()
                if not line:
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            body = b""
            content_length = int(headers.get("content-length", 0))
            if content_length > 0:
                if content_length > self.config.max_request_size:
                    await self._send_error(writer, "Request too large", 413)
                    return
                body = await reader.read(content_length)

            await self._route_request(writer, method, path, headers, body)

        except asyncio.TimeoutError:
            await self._send_error(writer, "Request timeout", 408)
        except Exception as e:
            logger.exception(f"Error handling request: {e}")
            await self._send_error(writer, f"Internal server error: {e}", 500)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _check_auth(self, headers: dict[str, str], path: str) -> bool:
        api_key = self.config.api_key
        if api_key is None:
            return True
        if path in ("/health", "/live", "/ready"):
            return True
        auth = headers.get("authorization", "")
        if auth == f"Bearer {api_key}":
            return True
        x_api_key = headers.get("x-api-key", "")
        return x_api_key == api_key

    async def _route_request(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        if not self._check_auth(headers, path):
            error_body, _ = format_error("Unauthorized", status_code=401)
            await self._send_json(writer, error_body, 401)
            return

        if path == "/health" and method == "GET":
            result = await self.handler.handle_health()
            await self._send_json(writer, result, 200)
            return

        if path == "/live" and method == "GET":
            result = await self.handler.handle_liveness()
            await self._send_json(writer, result, 200)
            return

        if path == "/ready" and method == "GET":
            result = await self.handler.handle_readiness()
            await self._send_json(writer, result, 200)
            return

        if path == "/v1/models" and method == "GET":
            result = await self.handler.handle_models()
            await self._send_json(writer, result, 200)
            return

        if path == "/v1/stats" and method == "GET":
            result = await self.handler.handle_stats()
            await self._send_json(writer, result, 200)
            return

        if path == "/v1/chat/completions" and method == "POST":
            await self._handle_completion(writer, body, headers=headers)
            return

        if path == "/v1/messages" and method == "POST":
            await self._handle_anthropic_messages(writer, body, headers=headers, path=path)
            return

        if path == "/v1/messages/count_tokens" and method == "POST":
            await self._handle_count_tokens(writer, body, headers=headers)
            return

        if path == "/v1/feedback" and method == "POST":
            result, status = await self.handler.handle_feedback(body)
            await self._send_json(writer, result, status)
            return

        await self._send_error(writer, f"Not found: {path}", 404)

    def _anthropic_error(self, message: str) -> dict:
        return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}

    async def _handle_completion(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            data = json.loads(body.decode("utf-8"))
            request = ChatCompletionRequest.from_dict(data)
        except json.JSONDecodeError as e:
            await self._send_error(writer, f"Invalid JSON: {e}", 400)
            return
        except ValueError as e:
            await self._send_error(writer, str(e), 400)
            return

        if request.stream:
            await self._send_stream(writer, request, headers=headers)
        else:
            try:
                result = await self.handler.handle_completion(request, headers=headers)
                await self._send_json(writer, result, 200)
            except BudgetExceededError as e:
                await self._send_json(writer, {"error": {"message": str(e), "type": "budget_exceeded", "code": 429}}, 429)
            except Exception as e:
                logger.exception(f"Completion error: {e}")
                await self._send_error(writer, str(e), 500)

    async def _handle_count_tokens(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                request_data = json.loads(body.decode("utf-8"))
                forward_body = json.dumps(request_data).encode("utf-8")
            except json.JSONDecodeError as e:
                await self._send_json(writer, self._anthropic_error(f"Invalid JSON: {e}"), 400)
                return

            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages/count_tokens",
                        headers={
                            "x-api-key": anthropic_key,
                            "anthropic-version": _ANTHROPIC_VERSION,
                            "content-type": "application/json",
                        },
                        content=forward_body,
                    )
                    await self._send_json(writer, resp.json(), resp.status_code)
                    return
                except Exception as e:
                    logger.warning("count_tokens upstream failed, falling back: %s", e)

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            await self._send_json(writer, self._anthropic_error(f"Invalid JSON: {e}"), 400)
            return

        total_chars = 0
        messages = data.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total_chars += len(block.get("text", ""))
        system = data.get("system", "")
        if isinstance(system, str):
            total_chars += len(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))

        import math
        estimated = math.ceil(total_chars / 4)
        await self._send_json(writer, {"input_tokens": estimated}, 200)

    async def _handle_anthropic_messages(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
        headers: dict[str, str] | None = None,
        path: str = "/v1/messages",
    ) -> None:
        headers = headers or {}

        try:
            data = json.loads(body.decode("utf-8"))
            messages, kwargs = anthropic_to_internal(data)
        except json.JSONDecodeError as e:
            await self._send_json(writer, self._anthropic_error(f"Invalid JSON: {e}"), 400)
            return
        except ValueError as e:
            await self._send_json(writer, self._anthropic_error(str(e)), 400)
            return

        stream = data.get("stream", False)
        request_model = kwargs.pop("model", "auto")

        openai_request = ChatCompletionRequest(
            model=request_model,
            messages=messages,
            stream=stream,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            stop=kwargs.get("stop"),
        )

        decision = self.handler.resolve_routing(openai_request, headers)

        # R4.5: Protocol-native fast path
        is_claude_request = data.get("model", "").startswith("claude-")
        is_fast_path = (
            decision.route
            and is_claude_request
            and path == "/v1/messages"
        )

        if is_fast_path:
            await self._handle_anthropic_fast_path(
                writer, body, headers, kwargs, decision, stream
            )
            return

        # Translation path (crossing or not fast-path-eligible)
        if kwargs.get("tools") is not None:
            openai_request.extra_kwargs["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            openai_request.extra_kwargs["tool_choice"] = kwargs["tool_choice"]

        if stream:
            routesmith_metadata: dict[str, Any] = {
                "routed": decision.route,
                "selected_model": kwargs.get("model", request_model),
                "requested_model": request_model,
                "passthrough_reason": decision.passthrough_reason,
            }

            anthropic_chunks: list[dict] = []
            try:
                async for chunk in self.handler.handle_completion_stream(
                    openai_request, headers=headers, decision=decision
                ):
                    raw = chunk.removeprefix("data: ").strip()
                    if raw == "[DONE]":
                        continue
                    try:
                        data_chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if data_chunk.get("choices"):
                        anthropic_chunks.append(data_chunk)
                        if data_chunk.get("model"):
                            routesmith_metadata["selected_model"] = data_chunk["model"]
            except NoCapableModelError as e:
                await self._send_json(writer, self._anthropic_error(str(e)), 400)
                return
            except Exception as e:
                logger.exception(f"Anthropic stream error: {e}")

            request_id = (
                self.routesmith._last_routing_metadata.request_id
                if self.routesmith._last_routing_metadata
                else "unknown"
            )

            sse = AnthropicSSEStream(
                request_id=request_id,
                request_model=kwargs.get("model", request_model),
                routesmith_metadata=routesmith_metadata,
            )
            events = sse.iter_chunks(anthropic_chunks)
            response_body = "".join(events).encode("utf-8")
            await writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Connection: keep-alive\r\n"
                b"Access-Control-Allow-Origin: *\r\n"
                b"\r\n" + response_body
            )
            await writer.drain()
        else:
            try:
                result = await self.handler.handle_completion(
                    openai_request, headers=headers, decision=decision
                )
                choice = result.get("choices", [{}])[0]
                msg = choice.get("message", {})
                finish = choice.get("finish_reason", "stop")
                usage = result.get("usage", {})
                actual_model = result.get("model", kwargs.get("model", request_model))

                content_blocks: list[dict] = []
                content_text = msg.get("content", "") or ""
                if content_text:
                    content_blocks.append({"type": "text", "text": content_text})

                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        args_raw = tc["function"]["arguments"]
                        try:
                            parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except json.JSONDecodeError:
                            logger.warning("Malformed JSON arguments in response tool_call: %s", args_raw[:200])
                            parsed_args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": parsed_args,
                        })

                anthropic_resp = {
                    "id": f"msg_{self.routesmith._last_routing_metadata.request_id}"
                    if self.routesmith._last_routing_metadata
                    else "msg_unknown",
                    "type": "message",
                    "role": "assistant",
                    "model": actual_model,
                    "content": content_blocks,
                    "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                }
                if result.get("routesmith_metadata"):
                    anthropic_resp["routesmith_metadata"] = result["routesmith_metadata"]

                audit_extra: dict[str, Any] = {}
                if not kwargs.get("model", "").startswith("claude-"):
                    original_body = json.loads(body.decode("utf-8"))
                    cache_blocks = _has_cache_control(original_body)
                    if cache_blocks:
                        audit_extra["cache_control_stripped"] = True
                if audit_extra and anthropic_resp.get("routesmith_metadata"):
                    anthropic_resp["routesmith_metadata"].update(audit_extra)

                await self._send_json(writer, anthropic_resp, 200)
            except NoCapableModelError as e:
                await self._send_json(writer, self._anthropic_error(str(e)), 400)
            except BudgetExceededError as e:
                await self._send_json(writer, self._anthropic_error(str(e)), 429)
            except Exception as e:
                logger.exception(f"Anthropic completion error: {e}")
                await self._send_json(writer, self._anthropic_error(str(e)), 500)

    async def _handle_anthropic_fast_path(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
        headers: dict[str, str],
        kwargs: dict[str, Any],
        decision: Any,
        stream: bool,
    ) -> None:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            logger.warning("Fast path requested but ANTHROPIC_API_KEY not set; falling through")
            await self._send_json(
                writer,
                self._anthropic_error("ANTHROPIC_API_KEY not configured for native forwarding"),
                400,
            )
            return

        try:
            body_decoded = body.decode("utf-8")
            parsed = json.JSONDecoder(object_pairs_hook=OrderedDict).decode(body_decoded)
            selected = kwargs.get("model", parsed.get("model", ""))
            parsed["model"] = selected
            new_body = json.dumps(
                parsed, ensure_ascii=True, sort_keys=False, indent=None, separators=(",", ":")
            ).encode("utf-8")
        except Exception as e:
            await self._send_json(writer, self._anthropic_error(f"Body processing error: {e}"), 400)
            return

        norm_hdrs = {k.lower(): v for k, v in headers.items()}
        forward_hdrs: dict[str, str] = {
            "x-api-key": anthropic_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        anthropic_beta = norm_hdrs.get("anthropic-beta")
        if anthropic_beta:
            forward_hdrs["anthropic-beta"] = anthropic_beta

        if stream:
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    async with client.stream(
                        "POST",
                        _ANTHROPIC_API_URL,
                        headers=forward_hdrs,
                        content=new_body,
                    ) as upstream_resp:
                        status_code = upstream_resp.status_code
                        if status_code != 200:
                            error_body = await upstream_resp.aread()
                            await self._send_raw(writer, status_code, error_body, content_type="application/json")
                            return

                        await writer.write(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: text/event-stream\r\n"
                            b"Cache-Control: no-cache\r\n"
                            b"Connection: keep-alive\r\n"
                            b"Access-Control-Allow-Origin: *\r\n"
                            b"\r\n"
                        )
                        await writer.drain()

                        async for line in upstream_resp.aiter_lines():
                            if line:
                                sse_line = line + "\n"
                                writer.write(sse_line.encode("utf-8"))
                                await writer.drain()
                except Exception as e:
                    logger.exception(f"Fast path stream error: {e}")
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                try:
                    upstream_resp = await client.post(
                        _ANTHROPIC_API_URL,
                        headers=forward_hdrs,
                        content=new_body,
                    )
                    await self._send_raw(
                        writer,
                        upstream_resp.status_code,
                        upstream_resp.content,
                        content_type=upstream_resp.headers.get("content-type", "application/json"),
                    )
                except Exception as e:
                    logger.exception(f"Fast path error: {e}")
                    await self._send_json(writer, self._anthropic_error(str(e)), 502)

    async def _send_raw(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        content: bytes,
        content_type: str = "application/json",
    ) -> None:
        status_text = {
            200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
            429: "Too Many Requests", 500: "Internal Server Error", 502: "Bad Gateway",
        }.get(status, "OK")
        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(content)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
        ).encode() + content
        writer.write(response)
        await writer.drain()

    async def _send_stream(
        self,
        writer: asyncio.StreamWriter,
        request: ChatCompletionRequest,
        headers: dict[str, str] | None = None,
    ) -> None:
        response_headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(response_headers.encode("utf-8"))
        await writer.drain()

        try:
            async for chunk in self.handler.handle_completion_stream(request, headers=headers):
                writer.write(chunk.encode("utf-8"))
                await writer.drain()
        except Exception as e:
            logger.exception(f"Stream error: {e}")

    async def _send_json(
        self,
        writer: asyncio.StreamWriter,
        data: dict[str, Any],
        status: int,
    ) -> None:
        body = json.dumps(data).encode("utf-8")
        status_text = {
            200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
            429: "Too Many Requests", 500: "Internal Server Error",
        }.get(status, "OK")

        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
        ).encode() + body

        writer.write(response)
        await writer.drain()

    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        message: str,
        status: int,
    ) -> None:
        error_body, _ = format_error(message, status_code=status)
        await self._send_json(writer, error_body, status)


def _has_cache_control(data: dict) -> bool:
    messages = data.get("messages", [])
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    system = data.get("system", "")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                return True
    return False
