# RouteSmith v0.2.0 → v0.3.0 — Resumption Checkpoint

**Date:** 2026-06-15
**v0.2.0:** Released (PRs #23-27 merged, tag v0.2.0 pushed)
**Next:** v0.3.0 Enterprise Features

## Quick Resume

```bash
cd /Users/yliulupo/Apps/routesmith/.worktrees/v0.2.0-stable
git checkout uat/v0.2.0  # or: dev
.venv/bin/pytest tests/ -q  # 595 passed, 16 skipped
```

## v0.2.0 Status (COMPLETE)

## Test Verification

```bash
# Unit tests (no API keys needed)
.venv/bin/pytest tests/ -q
# Expected: 584 passed, 14 skipped

# Live tests (needs Groq key in env)
export GROQ_API_KEY=<key>
.venv/bin/python tests/manual/test_real_api.py
.venv/bin/python tests/manual/test_langchain_live.py

# Multi-model eval (needs Groq key)
.venv/bin/python scripts/run_multi_model_eval.py

# Build docs
mkdocs build --strict
```

## v0.3.0 — Next Phase: Enterprise Features (Planned)

Phase 3 from design.md — not yet started:

- CostModel enum (ON_DEMAND, PROVISIONED, SELF_HOSTED)
- Capacity/utilization tracking for provisioned throughput
- PROVISIONED_FIRST routing strategy
- Overflow routing (provisioned → on-demand)
- Tag-based model filtering for compliance (HIPAA, SOC2, region)
- Per-project cost allocation / multi-tenant support
- Budget enforcement behaviors (FAIL/FALLBACK/QUEUE/THROTTLE)

Plan docs to reference:
- docs/plans/2026-06-15-v0.2.0-stable-release.md
- design.md (Persona 7: Priya - Enterprise Architect)

## Plan Documents

- Master: docs/plans/2026-06-15-v0.2.0-stable-release.md
- Section A: docs/plans/2026-06-15-section-a-core-stability.md
- Section B: docs/plans/2026-06-15-section-b-production-hardening.md
- Section C: docs/plans/2026-06-15-section-c-live-testing.md
- Section D: docs/plans/2026-06-15-section-d-documentation.md

## Key Files Changed (10 new modules, 40+ new tests)

- src/routesmith/exceptions.py
- src/routesmith/utils/logging.py
- src/routesmith/utils/retry.py
- src/routesmith/strategy/circuit_breaker.py
- src/routesmith/proxy/metrics.py
- src/routesmith/client.py (modified - circuit breaker + retry + logging wired)
- src/routesmith/proxy/handler.py (modified - health endpoints)
- Dockerfile
- docker-compose.yml
- .github/workflows/ci.yml
- .github/workflows/nightly.yml
- tests/perf/test_routing_latency.py
- tests/perf/test_memory.py
- scripts/run_multi_model_eval.py
- mkdocs.yml
- docs/ (20+ documentation files)
