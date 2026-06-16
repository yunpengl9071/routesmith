# Section B: Production Hardening — Implementation Plan

> **REQUIRED SUB-SKILL:** Use subagent-driven-development to execute.
> **Branch:** feature/v0.2.0-prod-hardening (from dev)
> **CRITICAL:** Real use case tests — metrics must export actual data, Docker must serve real traffic

**Goal:** Add Prometheus metrics, Docker image, CI/CD pipeline, and health endpoints.

**Architecture:** Metrics endpoint on proxy server (`/metrics`), health/liveness/readiness on `/health`, `/ready`. Multi-stage Dockerfile, GitHub Actions for CI and nightly tests.

**Tech Stack:** Python 3.13, prometheus-client, Docker, GitHub Actions

---

## Task B1: Prometheus Metrics Endpoint

**Files:**
- Create: `src/routesmith/proxy/metrics.py`
- Modify: `src/routesmith/proxy/handler.py` (add /metrics route)
- Test: `tests/test_metrics_endpoint.py`
- Deps: `prometheus-client` (add to pyproject.toml proxy extras)

### Step 1: Add prometheus-client dependency

In `pyproject.toml`, add to `[project.optional-dependencies]` under `proxy`:
```toml
proxy = [
    "pyyaml>=6.0.0",
    "httpx>=0.24.0",
    "questionary>=2.0.0",
    "certifi>=2024.0.0",
    "prometheus-client>=0.16.0",
]
```

### Step 2: Write failing tests

```python
# tests/test_metrics_endpoint.py
"""Real use case: Prometheus metrics expose all routing data."""
import pytest
from unittest.mock import patch, MagicMock
from prometheus_client import REGISTRY, CollectorRegistry
from routesmith.proxy.metrics import (
    init_metrics, ROUTING_REQUESTS, ROUTING_LATENCY, COST_USD,
    CACHE_HITS, ACTIVE_CIRCUITS,
)


class TestMetricsRegistration:
    """Metrics are registered with prometheus-client."""

    def test_metrics_are_registered(self):
        # Re-init with fresh registry
        registry = CollectorRegistry()
        init_metrics(registry)
        names = {m.name for m in registry.collect()}
        assert "routesmith_requests_total" in names
        assert "routesmith_routing_latency_seconds" in names
        assert "routesmith_cost_usd_total" in names
        assert "routesmith_cache_hits_total" in names
        assert "routesmith_active_circuits" in names

    def test_requests_counter_has_labels(self):
        registry = CollectorRegistry()
        init_metrics(registry)
        ROUTING_REQUESTS.labels(model="gpt-4o", strategy="direct", project="default").inc()
        # Should not raise

    def test_latency_histogram_has_buckets(self):
        registry = CollectorRegistry()
        init_metrics(registry)
        ROUTING_LATENCY.labels(strategy="cascade").observe(0.003)
        ROUTING_LATENCY.labels(strategy="cascade").observe(0.008)
        # Verify buckets are set
        for metric in registry.collect():
            if metric.name == "routesmith_routing_latency_seconds":
                assert len(metric.samples) > 0


class TestMetricsEndpointIntegration:
    """Real use case: /metrics returns Prometheus text format."""

    def test_metrics_endpoint_returns_200(self):
        from routesmith.proxy.handler import RequestHandler
        from routesmith import RouteSmith
        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                          cost_per_1k_output=0.0006, quality_score=0.85)
        handler = RequestHandler(rs)
        response = handler.handle_metrics()
        assert response  # Should return bytes or string with metrics


class TestMetricsRealData:
    """Real use case: metrics reflect actual routing decisions."""

    @patch("litellm.completion")
    def test_request_count_increments(self, mock_litellm):
        """After routing a request, request counter increments."""
        registry = CollectorRegistry()
        init_metrics(registry)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_litellm.return_value = mock_response

        from routesmith import RouteSmith
        rs = RouteSmith()
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                          cost_per_1k_output=0.0006, quality_score=0.85)
        # TODO: wire metrics into client
        rs.completion(messages=[{"role": "user", "content": "test"}])

        # Verify counter went up
        for metric in registry.collect():
            if metric.name == "routesmith_requests_total":
                total = sum(s.value for s in metric.samples if s.name.endswith("_total"))
                assert total >= 1
```

Run: `.venv/bin/pytest tests/test_metrics_endpoint.py -v` → FAIL (test uses integration not wired yet)

### Step 3: Write implementation

```python
# src/routesmith/proxy/metrics.py
"""Prometheus metrics for RouteSmith."""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

ROUTING_REQUESTS: Counter
ROUTING_LATENCY: Histogram
COST_USD: Counter
CACHE_HITS: Counter
ACTIVE_CIRCUITS: Gauge


def init_metrics(registry: CollectorRegistry = REGISTRY) -> None:
    global ROUTING_REQUESTS, ROUTING_LATENCY, COST_USD, CACHE_HITS, ACTIVE_CIRCUITS
    ROUTING_REQUESTS = Counter(
        "routesmith_requests_total",
        "Total requests routed",
        ["model", "strategy", "project"],
        registry=registry,
    )
    ROUTING_LATENCY = Histogram(
        "routesmith_routing_latency_seconds",
        "Routing decision latency",
        ["strategy"],
        buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0],
        registry=registry,
    )
    COST_USD = Counter(
        "routesmith_cost_usd_total",
        "Cumulative cost in USD",
        ["model", "project"],
        registry=registry,
    )
    CACHE_HITS = Counter(
        "routesmith_cache_hits_total",
        "Cache hits",
        ["type"],
        registry=registry,
    )
    ACTIVE_CIRCUITS = Gauge(
        "routesmith_active_circuits",
        "Number of open circuit breakers",
        registry=registry,
    )


# Auto-init on import
init_metrics()
```

**Add `/metrics` handler in handler.py:**
```python
def handle_metrics(self) -> bytes:
    from prometheus_client import generate_latest
    return generate_latest()
```

### Step 4: Commit

```bash
git add src/routesmith/proxy/metrics.py tests/test_metrics_endpoint.py src/routesmith/proxy/handler.py pyproject.toml
git commit -m "feat(metrics): add Prometheus metrics endpoint with counter, histogram, gauge"
```

---

## Task B2: Health/Liveness/Readiness Endpoints

**Files:**
- Modify: `src/routesmith/proxy/handler.py`
- Test: `tests/test_health_endpoints.py`

### Step 1: Write failing tests

```python
# tests/test_health_endpoints.py
import json
from unittest.mock import patch, MagicMock
from routesmith.proxy.handler import RequestHandler
from routesmith import RouteSmith


def make_handler():
    rs = RouteSmith()
    rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                      cost_per_1k_output=0.0006, quality_score=0.85)
    return RequestHandler(rs)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        handler = make_handler()
        resp = handler.handle_health()
        data = json.loads(resp)
        assert data["status"] == "ok"

    def test_health_includes_version(self):
        handler = make_handler()
        resp = handler.handle_health()
        data = json.loads(resp)
        assert "version" in data

    def test_liveness_always_200(self):
        """Liveness: always 200 if process is alive."""
        handler = make_handler()
        resp = handler.handle_liveness()
        data = json.loads(resp)
        assert data["status"] == "alive"

    def test_readiness_checks_registry(self):
        """Readiness: fails if no models registered."""
        from routesmith import RouteSmith
        rs = RouteSmith()  # No models registered
        handler = RequestHandler(rs)
        resp = handler.handle_readiness()
        data = json.loads(resp)
        assert data["status"] != "ready"
        assert "registry" in str(data)

    def test_readiness_ok_with_models(self):
        handler = make_handler()
        resp = handler.handle_readiness()
        data = json.loads(resp)
        assert data["status"] == "ready"
```

### Step 2: Write implementation

```python
# In handler.py, add:
def handle_health(self) -> str:
    import routesmith
    return json.dumps({
        "status": "ok",
        "version": getattr(routesmith, "__version__", "0.2.0"),
    })

def handle_liveness(self) -> str:
    return json.dumps({"status": "alive"})

def handle_readiness(self) -> str:
    models = self.rs.registry.list_models()
    if not models:
        return json.dumps({"status": "not_ready", "registry": "no models"})
    return json.dumps({"status": "ready", "models": len(models)})
```

### Step 3: Commit

```bash
git add tests/test_health_endpoints.py src/routesmith/proxy/handler.py
git commit -m "feat(health): add /health, /live, /ready endpoints"
```

---

## Task B3: Multi-Stage Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

### Step 1: Create Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.13-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
RUN uv sync --frozen --no-dev --extra proxy 2>/dev/null || \
    uv pip install --system -e ".[proxy]"

FROM python:3.13-slim
WORKDIR /app
RUN groupadd -r routesmith && useradd -r -g routesmith routesmith
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ src/
RUN chown -R routesmith:routesmith /app
USER routesmith
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "routesmith.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create .dockerignore

```
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
benchmark/results/
paper/
.git/
.worktrees/
worktrees/
.env
*.pyc
__pycache__/
docs/
tests/
```

### Step 3: Build and verify

```bash
docker build -t routesmith:v0.2.0-dev .
docker run --rm -p 8000:8000 routesmith:v0.2.0-dev &
sleep 3
curl http://localhost:8000/health
kill %1
```

### Step 4: Commit

```bash
git add Dockerfile .dockerignore
git commit -m "feat(docker): add multi-stage Dockerfile with healthcheck and non-root user"
```

---

## Task B4: docker-compose for Local Dev

**Files:**
- Create: `docker-compose.yml`
- Create: `prometheus/prometheus.yml`
- Create: `grafana/dashboards/routesmith.json`

### Step 1: Create docker-compose.yml

```yaml
version: "3.8"
services:
  routesmith:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ROUTESMITH_LOG_LEVEL=INFO
      - ROUTESMITH_LOG_FORMAT=json
    volumes:
      - ./routesmith.yaml:/app/routesmith.yaml:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s

  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=routesmith
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  prometheus_data:
  grafana_data:
```

### Step 2: Create prometheus.yml

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: "routesmith"
    static_configs:
      - targets: ["routesmith:8000"]
```

### Step 3: Commit

```bash
git add docker-compose.yml prometheus/
git commit -m "feat(docker): add docker-compose with Prometheus and Grafana"
```

---

## Task B5: GitHub Actions CI/CD Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/nightly.yml`

### Step 1: Create ci.yml

```yaml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra dev --extra anthropic --extra langchain
      - run: uv run ruff check src/ tests/
      - run: uv run mypy src/ --strict

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra dev --extra anthropic --extra langchain
      - run: uv run pytest tests/ -x -q --tb=short

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run pip-audit

  docker-build:
    needs: [lint, test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/dev' || startsWith(github.ref, 'refs/heads/release/')
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: routesmith:ci
```

### Step 2: Create nightly.yml

```yaml
name: Nightly Live Tests
on:
  schedule:
    - cron: "0 6 * * *"  # 6 AM UTC daily
  workflow_dispatch:
jobs:
  live-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen --extra dev --extra anthropic --extra langchain --extra langchain-agents
      - env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: uv run python tests/manual/test_real_api.py
```

### Step 3: Commit

```bash
git add .github/workflows/ci.yml .github/workflows/nightly.yml
git commit -m "feat(ci): add GitHub Actions CI pipeline and nightly live tests"
```

---

## UAT Validation (Real Use Case)

```bash
# 1. Build and run docker
docker compose up -d

# 2. Test health
curl http://localhost:8000/health | jq

# 3. Test metrics
curl http://localhost:8000/metrics | grep routesmith

# 4. Test models list
curl http://localhost:8000/models | jq

# 5. Test actual completion
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}' | jq

# 6. Stop
docker compose down
```
