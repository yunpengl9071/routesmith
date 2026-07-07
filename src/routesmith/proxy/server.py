"""OpenAI-compatible HTTP server for RouteSmith."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from typing import Any

from routesmith import RouteSmith
from routesmith.budget import BudgetExceededError
from routesmith.proxy.anthropic_compat import (
    _STOP_REASON_MAP,
    AnthropicSSEStream,
    anthropic_to_internal,
)
from routesmith.proxy.handler import ChatCompletionRequest, RequestHandler
from routesmith.proxy.responses import format_error

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for the proxy server."""

    host: str = "127.0.0.1"
    port: int = 9119
    read_timeout: float = 30.0
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    api_key: str | None = None  # If set, require Bearer auth on all routes except /health


class RouteSmithProxyServer:
    """
    OpenAI-compatible proxy server wrapping RouteSmith.

    Accepts requests at /v1/chat/completions and routes them
    through RouteSmith's intelligent routing layer.

    Example:
        >>> from routesmith import RouteSmith
        >>> from routesmith.proxy import RouteSmithProxyServer, ServerConfig
        >>>
        >>> rs = RouteSmith()
        >>> rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015, ...)
        >>>
        >>> server = RouteSmithProxyServer(rs, ServerConfig(port=9119))
        >>> asyncio.run(server.serve_forever())
    """

    def __init__(
        self,
        routesmith: RouteSmith,
        config: ServerConfig | None = None,
    ) -> None:
        """
        Initialize the proxy server.

        Args:
            routesmith: RouteSmith instance for routing and completion.
            config: Server configuration (host, port, etc.).
        """
        self.routesmith = routesmith
        self.config = config or ServerConfig()
        self.handler = RequestHandler(routesmith)
        self._server: asyncio.Server | None = None
        self._running = False

    @property
    def host(self) -> str:
        """Get server host."""
        return self.config.host

    @property
    def port(self) -> int:
        """Get server port."""
        if self._server and self._server.sockets:
            # Get actual port if using port 0 (random)
            return self._server.sockets[0].getsockname()[1]
        return self.config.port

    async def start(self) -> None:
        """Start the server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.config.host,
            self.config.port,
        )
        self._running = True
        logger.info(f"Server started on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Gracefully stop the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Server stopped")

    async def serve_forever(self) -> None:
        """Run the server until interrupted."""
        await self.start()

        # Set up signal handlers for graceful shutdown
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
        """Handle a single HTTP connection."""
        try:
            # Read request line and headers
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

            # Read headers
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                line = line.decode("utf-8").strip()
                if not line:
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()

            # Read body if present
            body = b""
            content_length = int(headers.get("content-length", 0))
            if content_length > 0:
                if content_length > self.config.max_request_size:
                    await self._send_error(writer, "Request too large", 413)
                    return
                body = await reader.read(content_length)

            # Route request
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
        """Check if request is authorized.

        Returns True if authorized, False if not.
        Accepts both Authorization: Bearer <key> and x-api-key: <key>
        (Anthropic convention). Health/liveness/readiness endpoints are
        always allowed.
        """
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
        """Route request to appropriate handler."""
        # Auth check
        if not self._check_auth(headers, path):
            error_body, _ = format_error("Unauthorized", status_code=401)
            await self._send_json(writer, error_body, 401)
            return

        # Health check
        if path == "/health" and method == "GET":
            result = await self.handler.handle_health()
            await self._send_json(writer, result, 200)
            return

        # Liveness probe (Kubernetes)
        if path == "/live" and method == "GET":
            result = await self.handler.handle_liveness()
            await self._send_json(writer, result, 200)
            return

        # Readiness probe (Kubernetes)
        if path == "/ready" and method == "GET":
            result = await self.handler.handle_readiness()
            await self._send_json(writer, result, 200)
            return

        # Models list
        if path == "/v1/models" and method == "GET":
            result = await self.handler.handle_models()
            await self._send_json(writer, result, 200)
            return

        # Stats
        if path == "/v1/stats" and method == "GET":
            result = await self.handler.handle_stats()
            await self._send_json(writer, result, 200)
            return

        # Chat completions (OpenAI format)
        if path == "/v1/chat/completions" and method == "POST":
            await self._handle_completion(writer, body)
            return

        # Anthropic Messages API endpoint
        if path == "/v1/messages" and method == "POST":
            await self._handle_anthropic_messages(writer, body)
            return

        # Feedback endpoint
        if path == "/v1/feedback" and method == "POST":
            result, status = await self.handler.handle_feedback(body)
            await self._send_json(writer, result, status)
            return

        # 404 for unknown paths
        await self._send_error(writer, f"Not found: {path}", 404)

    async def _handle_completion(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
    ) -> None:
        """Handle OpenAI-format chat completion request."""
        try:
            data = json.loads(body.decode("utf-8"))
            request = ChatCompletionRequest.from_dict(data)
        except json.JSONDecodeError as e:
            await self._send_error(writer, f"Invalid JSON: {e}", 400)
            return
        except ValueError as e:
            await self._send_error(writer, str(e), 400)
            return

        # Streaming response
        if request.stream:
            await self._send_stream(writer, request)
        else:
            # Non-streaming response
            try:
                result = await self.handler.handle_completion(request)
                await self._send_json(writer, result, 200)
            except BudgetExceededError as e:
                await self._send_json(writer, {"error": {"message": str(e), "type": "budget_exceeded", "code": 429}}, 429)
            except Exception as e:
                logger.exception(f"Completion error: {e}")
                await self._send_error(writer, str(e), 500)

    async def _handle_anthropic_messages(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
    ) -> None:
        """Handle Anthropic /v1/messages request."""
        try:
            data = json.loads(body.decode("utf-8"))
            messages, kwargs = anthropic_to_internal(data)
        except json.JSONDecodeError as e:
            await self._send_error(writer, f"Invalid JSON: {e}", 400)
            return
        except ValueError as e:
            await self._send_error(writer, str(e), 400)
            return

        stream = data.get("stream", False)
        request_model = kwargs.pop("model", "auto")

        # Extract RouteSmith headers from HTTP request context
        # (headers are passed through via the calling code)

        # Build an OpenAI-format ChatCompletionRequest

        openai_request = ChatCompletionRequest(
            model="auto",
            messages=messages,
            stream=stream,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            stop=kwargs.get("stop"),
        )

        if stream:
            anthropic_chunks: list[dict] = []
            try:
                async for chunk in self.handler.handle_completion_stream(openai_request):
                    data_chunk = json.loads(chunk.removeprefix("data: ").strip())
                    if data_chunk.get("choices"):
                        anthropic_chunks.append(data_chunk)
            except Exception as e:
                logger.exception(f"Anthropic stream error: {e}")

            sse = AnthropicSSEStream(
                request_id=self.routesmith._last_routing_metadata.request_id if self.routesmith._last_routing_metadata else "unknown",
                request_model=request_model,
            )
            events = sse.iter_chunks(anthropic_chunks)
            response_body = "".join(events).encode("utf-8")
            status_text = "OK"
            status = 200
            await writer.write(
                f"HTTP/1.1 {status} {status_text}\r\n"
                f"Content-Type: text/event-stream\r\n"
                f"Cache-Control: no-cache\r\n"
                f"Connection: keep-alive\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"\r\n".encode() + response_body
            )
            await writer.drain()
        else:
            try:
                result = await self.handler.handle_completion(openai_request)
                choice = result.get("choices", [{}])[0]
                msg = choice.get("message", {})
                content_text = msg.get("content", "")
                finish = choice.get("finish_reason", "stop")
                usage = result.get("usage", {})
                actual_model = result.get("model", request_model)
                anthropic_resp = {
                    "id": f"msg_{self.routesmith._last_routing_metadata.request_id}" if self.routesmith._last_routing_metadata else "msg_unknown",
                    "type": "message",
                    "role": "assistant",
                    "model": actual_model,
                    "content": [{"type": "text", "text": content_text}],
                    "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                }
                if result.get("routesmith_metadata"):
                    anthropic_resp["routesmith_metadata"] = result["routesmith_metadata"]
                await self._send_json(writer, anthropic_resp, 200)
            except BudgetExceededError as e:
                await self._send_json(writer, {"error": {"message": str(e), "type": "budget_exceeded", "code": 429}}, 429)
            except Exception as e:
                logger.exception(f"Anthropic completion error: {e}")
                await self._send_error(writer, str(e), 500)

    async def _send_stream(
        self,
        writer: asyncio.StreamWriter,
        request: ChatCompletionRequest,
    ) -> None:
        """Send streaming response."""
        # Send headers for SSE
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
            async for chunk in self.handler.handle_completion_stream(request):
                writer.write(chunk.encode("utf-8"))
                await writer.drain()
        except Exception as e:
            logger.exception(f"Stream error: {e}")
            # Can't send error mid-stream, just log it

    async def _send_json(
        self,
        writer: asyncio.StreamWriter,
        data: dict[str, Any],
        status: int,
    ) -> None:
        """Send JSON response."""
        body = json.dumps(data).encode("utf-8")
        status_text = {200: "OK", 400: "Bad Request", 404: "Not Found", 429: "Too Many Requests", 500: "Internal Server Error"}.get(status, "OK")

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
        """Send error response."""
        error_body, _ = format_error(message, status_code=status)
        await self._send_json(writer, error_body, status)
