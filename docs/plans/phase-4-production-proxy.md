# Phase 4 — Production-grade proxy

The proxy is the drop-in product surface, and today it is single-tenant-localhost-only:
no auth, wildcard CORS, a hand-rolled HTTP parser, no metrics, and multi-replica deployments
would fragment bandit learning. This phase makes "run it for your team" defensible.

**Exit criteria:** G10 (`tests/test_proxy_auth.py`, `tests/test_proxy_metrics.py`) green.

Task order: P4.1 → P4.2 → P4.3 → P4.4. (P4.1/P4.2 independent; P4.4 last.)

---

## Task P4.1 — Bearer-token auth + CORS tightening  (size: S)

**Files:** `src/routesmith/proxy/server.py`, `src/routesmith/cli/serve.py`,
`tests/test_proxy_auth.py` (new)

**Spec:**
1. `ServerConfig` gains `api_keys: list[str] = field(default_factory=list)` and
   `cors_origin: str = "*"`. In `cli/serve.py`, populate `api_keys` from env
   `ROUTESMITH_API_KEYS` (comma-separated, stripped, empties dropped) and/or YAML
   `server.api_keys`; `cors_origin` from `server.cors_origin`.
2. In `_route_request`, BEFORE any endpoint dispatch except `/health`:

   ```python
   if self.config.api_keys and path != "/health":
       auth = headers.get("authorization", "")
       token = auth[7:] if auth.lower().startswith("bearer ") else ""
       if token not in self.config.api_keys:
           await self._send_json(writer, {"error": {
               "message": "Invalid or missing API key",
               "type": "authentication_error", "code": 401}}, 401)
           return
   ```
   Add `401: "Unauthorized"` and `403: "Forbidden"` to the `_send_json` status map.
3. Replace both hardcoded `Access-Control-Allow-Origin: *` occurrences (in `_send_json` and
   `_send_stream`) with `self.config.cors_origin`.
4. Semantics: empty `api_keys` list ⇒ auth disabled (backwards compatible, dev-friendly);
   log a one-line warning at startup when the server binds to a non-loopback host with auth
   disabled.

**Tests** (`tests/test_proxy_auth.py`, drive `_route_request` with a fake writer per the
existing `tests/test_proxy.py` pattern):
- `test_no_keys_configured_allows_all`.
- `test_valid_bearer_accepted` / `test_missing_header_401` / `test_wrong_key_401` /
  `test_malformed_header_401` (`"Basic abc"`).
- `test_health_open_without_auth`.
- `test_cors_origin_configurable` — response headers contain the configured origin, not `*`.

**Acceptance:** G10 auth half green; README "Securing the proxy" section (keys, env var,
reverse-proxy TLS note).

---

## Task P4.2 — Prometheus `/metrics` endpoint  (size: M)

No external dependency — emit the text exposition format by hand.

**Files:** `src/routesmith/proxy/metrics.py` (new), `src/routesmith/proxy/server.py`,
`src/routesmith/proxy/handler.py`, `tests/test_proxy_metrics.py` (new)

**Spec — `metrics.py`:**

```python
"""Minimal Prometheus text-format metrics. Thread-safe, no deps."""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._latency_sum = 0.0
        self._latency_count = 0

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += value

    def observe_latency(self, seconds: float) -> None:
        with self._lock:
            self._latency_sum += seconds
            self._latency_count += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), val in sorted(self._counters.items()):
                label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{label_str}}} {val}" if label_str else f"{name} {val}")
            lines.append(f"routesmith_request_latency_seconds_sum {self._latency_sum}")
            lines.append(f"routesmith_request_latency_seconds_count {self._latency_count}")
        return "\n".join(lines) + "\n"
```

**Metric names (fixed contract):**
`routesmith_requests_total{model,outcome}` (outcome: `ok|error|budget_rejected|cache_hit`),
`routesmith_cost_usd_total{model}`, `routesmith_feedback_total{source}`
(source: `explicit|implicit|judge`), `routesmith_routing_decisions_total{reason}`.

**Wiring:** one `Metrics` instance on the server, passed into `RequestHandler`; increment at
the natural points in `handle_completion` / `handle_feedback` / the budget-429 path /
cache hits (read `cache_hits` delta from `rs.stats` or increment where the client exposes a
hook — simplest: increment in handler based on `routesmith_metadata.routing_reason` and
response fields). `GET /metrics` in `_route_request` returns `render()` with
`Content-Type: text/plain; version=0.0.4` — add a `_send_text` helper mirroring `_send_json`.
`/metrics` IS auth-protected when keys are configured.

**Tests** (`tests/test_proxy_metrics.py`):
- `test_counters_increment_and_render` — 2 completions + 1 feedback → rendered text contains
  `routesmith_requests_total{model="...",outcome="ok"} 2` and
  `routesmith_feedback_total{source="explicit"} 1`.
- `test_render_escaping_and_format` — no trailing spaces; ends with newline; latency sum/count present.
- `test_metrics_endpoint_served` / `test_metrics_requires_auth_when_enabled`.

**Acceptance:** G10 metrics half green; example Grafana-friendly scrape config in README.

---

## Task P4.3 — Pluggable state backend (multi-replica learning)  (size: L)

Today each proxy replica learns a private, divergent posterior (state persists to local
SQLite only). Define a backend seam and add Redis so N replicas share one brain.

**Files:** `src/routesmith/state.py` (new), `src/routesmith/feedback/storage.py`,
`src/routesmith/client.py`, `pyproject.toml` (extra `redis = ["redis>=5.0"]`),
`tests/test_state_backend.py` (new)

**Spec — protocol (`state.py`):**

```python
"""Pluggable predictor-state and spend-counter backends."""
from __future__ import annotations

from typing import Protocol


class StateBackend(Protocol):
    def save_predictor_state(self, blob: bytes, version: int) -> None: ...
    def load_predictor_state(self) -> tuple[bytes, int] | None: ...
    def incr_spend(self, day_bucket: str, amount: float) -> float: ...
    def get_spend(self, day_bucket: str) -> float: ...
```

1. `SQLiteStateBackend` — thin adapter over the EXISTING
   `FeedbackStorage.save_predictor_state/load_predictor_state` (grep those names in
   `storage.py`) plus a new `spend(day_bucket TEXT PRIMARY KEY, amount REAL)` table with
   `INSERT ... ON CONFLICT ... DO UPDATE SET amount = amount + ?` for `incr_spend`.
   `version` = monotonically increasing int stored beside the blob (add a column or a
   suffixed key, matching however the blob is stored today).
2. `RedisStateBackend` — keys `routesmith:state` (blob), `routesmith:state:version` (INCR),
   `routesmith:spend:<day_bucket>` (INCRBYFLOAT, `EXPIRE` 172800 s). Constructor takes a URL
   (`redis.Redis.from_url`); import redis lazily; raise a clear error naming
   `pip install routesmith[redis]` when missing.
3. Client wiring:
   - `RouteSmithConfig` gains `state_backend_url: str | None = None`
     (`None` → SQLite/current behavior; `redis://...` → Redis).
   - Where the client currently saves state every 50 updates
     (`self.feedback._storage.save_predictor_state(` in `client.py`), route through the
     backend and bump version.
   - **Cross-replica refresh:** track `self._state_version`; on each `completion()` where
     `time.time() - self._last_state_check > 60`, compare backend version; if newer, reload
     predictor state (`predictor.load_state(blob)`). Eventual consistency (≤60 s skew) is the
     documented model — do NOT attempt per-update synchronization.
   - Day-window budget (`max_cost_per_day`) reads `get_spend(today_utc)` from the backend
     when one is configured, so replicas share the cap; minute/hour windows remain local
     (documented limitation). Integration seam: `BudgetTracker.__init__` (from P0.4) gains an
     optional `backend: StateBackend | None = None`; when set, `check()` consults
     `backend.get_spend(day_bucket)` for the day window and `record()` also calls
     `backend.incr_spend(...)`. Local deque behavior is unchanged otherwise.

**Tests** (`tests/test_state_backend.py`):
- `test_sqlite_backend_roundtrip` — save blob v1, load → same blob+version;
  `incr_spend` twice → `get_spend` sums.
- `test_redis_backend_roundtrip` — against `fakeredis` if importable else
  `pytest.importorskip("fakeredis")`. Add `fakeredis` to the dev extra.
- `test_client_reloads_newer_state` — two RouteSmith instances sharing one SQLite backend;
  instance A records 5 outcomes (forcing a save — temporarily set the save interval low via
  monkeypatching the constant); force B's check timer past 60 s → B's predictor `_t`/update
  count reflects A's training.
- `test_day_budget_shared_via_backend` — A spends to the cap; B's `check()` raises.

**Acceptance:** tests pass; `docs/deploy.md` (new) documents the 2-replica docker-compose
topology (2× `routesmith serve` + redis) and the 60 s consistency window.

---

## Task P4.4 — Replace the hand-rolled HTTP server with Starlette/Uvicorn  (size: L)

The raw-socket parser (manual request-line/header parsing in `server.py
_handle_connection`) closes every connection (no keep-alive), skips chunked encoding, and is
an avoidable security surface.

**Files:** `src/routesmith/proxy/app.py` (new), `src/routesmith/proxy/server.py` (slims to a
runner), `src/routesmith/cli/serve.py`, `pyproject.toml` (proxy extra gains
`starlette>=0.37`, `uvicorn>=0.29`), all proxy tests

**Spec:**
1. `app.py`: `build_app(routesmith, server_config) -> Starlette` with routes:
   `POST /v1/chat/completions` (JSON + `StreamingResponse` for SSE when `stream:true`),
   `POST /v1/feedback`, `GET /v1/models`, `GET /v1/stats`, `GET /metrics`, `GET /health`.
   Reuse `RequestHandler` VERBATIM — the handler layer is transport-agnostic already; only
   the socket layer is replaced. Auth (P4.1) becomes a pure-ASGI middleware function;
   CORS via `starlette.middleware.cors.CORSMiddleware` with `allow_origins=[cors_origin]`.
2. `server.py`: `RouteSmithProxyServer.start/stop/serve_forever` now wrap
   `uvicorn.Server(uvicorn.Config(app, host, port, log_level="info"))`. Keep the class name
   and constructor signature — `cli/serve.py` and user code must not change.
3. Error mapping: `BudgetExceededError` → 429; JSON parse errors → 400; unknown path → 404 —
   identical bodies to today (reuse `format_error`).
4. Port the existing proxy tests to `starlette.testclient.TestClient` (this REPLACES the
   fake-writer pattern; update `tests/test_proxy*.py` accordingly — this is the one
   sanctioned test rewrite, keep assertions identical).
5. Delete the raw parser only after all proxy tests pass on the ASGI app.

**Tests:** all existing + new proxy suites green under TestClient; add
`test_streaming_chunks_are_sse` (chunks prefixed `data: `, terminated by `data: [DONE]`) and
`test_keep_alive_two_requests_one_client` (TestClient session issues 2 requests).

**Acceptance:** proxy behavior byte-compatible for JSON endpoints (existing tests unchanged
except transport shim); README updated (`pip install routesmith[proxy]` now pulls uvicorn).
