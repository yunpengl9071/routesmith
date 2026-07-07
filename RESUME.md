# Resume

## Current Pipeline Step
**Step 6/6** — v0.7.0 tagged on `dev`

## Status
- All 4 phases complete — nothing pending
- v0.7.0 tagged: https://github.com/yunpengl9071/routesmith/tree/v0.7.0
- PR #43 merged: https://github.com/yunpengl9071/routesmith/pull/43
- CI gate: ruff ✓, mypy ✓, pytest 844/23 ✓

## What's Done
- Phase 1: Feature normalization, record-all, proxy feedback, implicit signals, LLM judge, convergence
- Phase 2: Routing quality benchmarking, experiment runner
- Phase 3: Parallel/speculative strategies, LinTS predictor, multi-tenant cache, evaluate CLI, per-project stats, audit log, per-role policies, v0.7.0 tagged
- Phase 4: LangChain, Anthropic, DSPy, CrewAI, AutoGen, OpenClaw integrations + A/B test framework

## What's Next (if continuing)
- Combined UAT with real API keys (needs GROQ_API_KEY or similar)
- PyPI publish: `python -m build && twine check dist/* && twine upload dist/*`
- arXiv paper publication
- New feature requests / bug reports

## Commands to Resume
```bash
git checkout dev && git pull origin dev
# Check status:
git log --oneline -5
# Un-skip UAT with API key:
# export GROQ_API_KEY=...
# uv run pytest tests/manual/test_real_api.py -v
```

## Key Files Changed (this session)
- `src/routesmith/cli/audit.py` — **NEW** audit CLI
- `src/routesmith/cli/roles.py` — **NEW** per-role policy CLI
- `src/routesmith/cli/stats.py` — **MODIFIED** `--project` filter, per-project breakdown
- `src/routesmith/feedback/audit.py` — **NEW** AuditStorage class
- `src/routesmith/feedback/storage.py` — **MODIFIED** get_project_stats(), get_known_projects()
- `src/routesmith/client.py` — **MODIFIED** _record_audit() wired into all 6 routing points
- `src/routesmith/cli/main.py` — **MODIFIED** audit + roles subparsers, --project flag
