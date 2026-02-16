"""OpenAI-compatible HTTP server for RouteSmith."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from typing import Any

from routesmith import RouteSmith
from routesmith.proxy.handler import RequestHandler, ChatCompletionRequest
from routesmith.proxy.responses import format_error

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for the proxy server."""

    host: str = "127.0.0.1"
    port: int = 9119
    read_timeout: float = 30.0
    max_request_size: int = 10 * 1024 * 1024  # 10MB


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

    async def _route_request(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Route request to appropriate handler."""
        # Health check
        if path == "/health" and method == "GET":
            result = await self.handler.handle_health()
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

        # Chat completions
        if path == "/v1/chat/completions" and method == "POST":
            await self._handle_completion(writer, body)
            return

        # 404 for unknown paths
        await self._send_error(writer, f"Not found: {path}", 404)

    async def _handle_completion(
        self,
        writer: asyncio.StreamWriter,
        body: bytes,
    ) -> None:
        """Handle chat completion request."""
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
            except Exception as e:
                logger.exception(f"Completion error: {e}")
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
        status_text = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}.get(status, "OK")

        response = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: *\r\n"
            f"\r\n"
        ).encode("utf-8") + body

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
