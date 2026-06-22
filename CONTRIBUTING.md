# Contributing to RouteSmith

Thank you for your interest in contributing! RouteSmith is an open-source adaptive LLM execution engine that intelligently routes queries to optimal models.

## Quick Start

```bash
git clone https://github.com/yunpengl9071/routesmith.git
cd routesmith
uv sync --extra dev --extra anthropic --extra langchain --extra cache
uv run pytest tests/ -q
uv run ruff check src/ tests/
```

## Development Workflow

1. **Check `state.md`** for current development status and open items
2. **Branch from `dev`**: `git checkout dev && git pull origin dev && git checkout -b feature/your-feature`
3. **Write tests first** (TDD preferred — see test files in `tests/` for patterns)
4. **Implement** your feature
5. **Run full test suite**: `uv run pytest tests/ -q`
6. **Lint**: `uv run ruff check src/ tests/`
7. **Type check**: `uv run mypy src/`
8. **Create PR** against `dev`

## CI/CD Pipeline

Every feature follows this path:
```
feature/<name> → CI gate (ruff, mypy, pytest, pip-audit, bandit) → dev → UAT branch → smoke test → tag
```

CI must pass before merge. Never skip the UAT smoke test stage.

## Code Style

- **Python 3.10+** with type annotations
- **Ruff** for linting (`line-length = 100`)
- **Mypy** for type checking (enforced in CI)
- **Docstrings** for public APIs (Google style)
- Follow existing patterns in the module you're modifying

## Project Structure

```
src/routesmith/
├── __init__.py           # Main exports
├── client.py             # RouteSmith client (drop-in replacement)
├── config.py             # Configuration dataclasses
├── registry/             # Model registry & capability mapping
├── predictor/            # Quality prediction (embedding, RF, bandits)
├── strategy/             # Routing strategies (direct, cascade, parallel)
├── cache/                # Semantic caching layer
├── feedback/             # Feedback collection & adaptation
├── integrations/         # Framework adapters (LangChain, DSPy, etc.)
├── proxy/                # OpenAI-compatible HTTP proxy server
├── cli/                  # CLI commands (serve, stats, init, dashboard)
└── utils/                # Shared utilities (logging, retry)
```

## Testing

| Test Type | Command | API Keys Needed |
|-----------|---------|-----------------|
| Unit tests | `uv run pytest tests/ -m "not requires_api"` | No |
| Full suite | `uv run pytest tests/ -q` | No |
| Live smoke | `uv run pytest tests/manual/test_real_api.py` | Yes (Groq recommended) |

## Adding a New Predictor

1. Create `src/routesmith/predictor/your_predictor.py`
2. Extend `BasePredictor` with `predict()` and `update()` methods
3. Add your predictor to `src/routesmith/predictor/__init__.py`
4. Add config fields to `PredictorConfig` in `config.py`
5. Add a `_create_predictor` branch in `strategy/router.py`
6. Add tests in `tests/`
7. Update `CHANGELOG.md`

See `src/routesmith/predictor/linucb.py` for a complete example.

## Questions?

Open an issue or check `state.md` and `design.md` for architectural context.