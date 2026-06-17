# RESUME.md — RouteSmith Session Checkpoint

**Updated**: 2026-06-17 06:39 UTC  
**Branch**: `dev` (up to date with origin)  
**Tag**: `v0.3.0`

## CI/CD Pipeline: COMPLETE ✅

```
feature/* → CI gate ✅ → dev ✅ → UAT ✅ → smoke test ✅ → tag v0.3.0 ✅
                                              ALL COMPLETE
```

## v0.3.0 Summary

| Artifact | Status |
|----------|--------|
| PR #28 v0.3.0 Enterprise Features | ✅ Merged to dev |
| PR #29 OpenClaw smoke test | ✅ Merged to dev |
| UAT branch `uat/v0.3.0` | ✅ Created & smoke tested |
| test_real_api.py | 4/4 passed |
| test_langchain_live.py | 10/10 passed |
| test_openclaw_live.py | 11/11 passed |
| UAT fix (model_quality_priors → _ema_priors) | ✅ Applied & merged |
| Merged uat/v0.3.0 → dev | ✅ |
| Tagged v0.3.0 | ✅ |
| Full unit test suite | 639 passed, 2 skipped |

## Exact commands to resume

```bash
cd /Users/yliulupo/Apps/routesmith
git checkout dev
git pull origin dev
.venv/bin/pytest tests/ --ignore=tests/manual -q
# Expected: 639 passed, 2 skipped
```

## Remaining for v0.3.1+

### Documentation
- v0.3.0 guides already on dev: cost-models.md, compliance.md, multi-project.md, budget-enforcement.md
- New nav items added to mkdocs.yml

### LangChain live tests
- Groq tool_call flakiness — there's already a `_retry_on_groq_tool_error()` helper
- Needs manual run with both Groq and OpenAI keys to identify which tests fail

### Housekeeping
- The `site/` directory (mkdocs built output) is in the repo — may want to add to `.gitignore`

## PRs merged

| PR | Branch | Description |
|----|--------|-------------|
| #28 | feature/v0.3.0-enterprise | v0.3.0 Enterprise Features |
| #29 | feature/openclaw-smoke-test | OpenClaw smoke test |

## API keys

Located in `.env` (gitignored):
- `GROQ_API_KEY` — present, free tier
- `OPENAI_API_KEY` — present
- `ANTHROPIC_API_KEY` — not set
