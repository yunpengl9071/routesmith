FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv sync --no-dev --extra proxy

FROM python:3.13-slim
WORKDIR /app
RUN groupadd -r routesmith && useradd -r -g routesmith routesmith
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY src/ src/
COPY pyproject.toml ./
RUN chown -R routesmith:routesmith /app
USER routesmith
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=2 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["routesmith", "serve", "--host", "0.0.0.0", "--port", "8000"]