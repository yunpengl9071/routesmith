# Resume

## Current Pipeline Step
**Step 1/6** — Phase 5, P5.1: examples/ directory + CI smoke harness

## Status
- Phase 5 feature branch: `feature/phase-5-integrations`
- PR #44 merged (docs + QUEUE + proxy auth)
- CI gate: ruff ✓, mypy ✓, pytest 844/23 ✓

## What's Done (All 4 phases + v0.7.1 enhancements)
- Phase 1: Feature normalization, record-all, proxy feedback, implicit signals, LLM judge, convergence
- Phase 2: Routing quality benchmarking, experiment runner
- Phase 3: Parallel/speculative strategies, LinTS predictor, multi-tenant cache, evaluate CLI, per-project stats, audit log, per-role policies
- Phase 4: LangChain, Anthropic, DSPy, CrewAI, AutoGen, OpenClaw integrations + A/B test framework
- v0.7.1: docs/cli.md, QUEUE budget behavior, proxy --api-key auth

## What's Next (if continuing)
- Combined UAT with real API keys (needs GROQ_API_KEY or similar)
- PyPI publish: `python -m build && twine check dist/* && twine upload dist/*`
- arXiv paper publication
- Benchmark experiments (LinTS-27d, 5-arm multi-model, ablations)
- New feature requests / bug reports

## Commands to Resume
```bash
git checkout dev && git pull origin dev
# Check PR #44 status:
gh pr view 44
# Un-skip UAT with API key:
# export GROQ_API_KEY=...
# uv run pytest tests/manual/test_real_api.py -v
```

## Key Files Changed (v0.7.1 session)
- `docs/cli.md` — **NEW** full CLI reference
- `README.md` — **MODIFIED** updated version, added audit/roles examples
- `src/routesmith/budget.py` — **MODIFIED** wait_until_available(), await_until_available()
- `src/routesmith/client.py` — **MODIFIED** QUEUE mode uses blocking wait instead of error
- `src/routesmith/proxy/server.py` — **MODIFIED** api_key config, _check_auth()
- `src/routesmith/cli/serve.py` — **MODIFIED** --api-key flag
- `src/routesmith/cli/main.py` — **MODIFIED** --api-key on serve parser
- `src/routesmith/utils/logging.py` — **MODIFIED** waited_seconds in _EXTRA_KEYS
