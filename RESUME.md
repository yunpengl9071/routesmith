# RESUME.md — v0.5.0 ML Predictor Merge

## Current CI/CD Step: 1/7 — Create PR

Branch: `feature/v0.5.0-ml-paper-merge`
Worktree: `.worktrees/v0.5.0-ml-merge`

## What Was Done (Path A)

Merged ML advances from `feature/contextual-bandit-routing` research branch INTO `dev`,
preserving ALL v0.4.x enterprise features and integrations.

### New Predictors
- `src/routesmith/predictor/neural_ucb.py` — NeuralUCB (shallow NN + UCB)
- `src/routesmith/predictor/reinforce.py` — Policy-gradient REINFORCE
- `src/routesmith/predictor/warmstart_linucb.py` — WarmStartLinUCB

### Simplified Core
- Feature vector: 35→27 dim (removed 8 context features per paper results)
- LinUCBPredictor: removed `context`, `reward_override`, `add_arm`, `remove_arm`, `serialize_state`, `load_state`
- Router: added `neural_ucb`, `reinforce`, `warmstart_linucb` predictor types

### Paper + Benchmarks
- `paper/main.tex`, `paper/references.bib`, `paper/sections/`, `paper/figures/`
- `benchmark/run_linucb_27d.py`, `benchmark/run_linucb_fast.py`

### Preserved
- All v0.4.x features: polls, explanations, verification, dashboard, conversation stickiness
- All enterprise: compliance routing, PROVISIONED_FIRST, business rules, budget enforcement
- All integrations: LangChain, Anthropic, DSPy, CrewAI, AutoGen
- All existing `context` plumbing (accepted but no longer contributes to predictor features)

## Exact Commands to Resume

```bash
cd /Users/yliulupo/Apps/routesmith/.worktrees/v0.5.0-ml-merge

# Verify clean
uv run ruff check src/ tests/
uv run pytest tests/ -q

# Create PR
gh pr create --base dev --head feature/v0.5.0-ml-paper-merge \
  --title "feat: v0.5.0 - ML predictor merge (NeuralUCB, REINFORCE, WarmStart LinUCB)" \
  --body "Merges ML advances from paper branch while keeping all v0.4.x features.
  
  New: NeuralUCB, REINFORCE, WarmStartLinUCB predictors
  Simplified: 27-dim features, cleaner LinUCB API
  Preserved: All enterprise features, integrations, v0.4.x UX
  
  721 tests pass, ruff clean."
```

## Test Count
- 721 passed, 28 skipped
- ruff: All checks passed

## Key Files Changed
| File | Change |
|------|--------|
| `src/routesmith/predictor/neural_ucb.py` | NEW |
| `src/routesmith/predictor/reinforce.py` | NEW |
| `src/routesmith/predictor/warmstart_linucb.py` | NEW |
| `src/routesmith/predictor/features.py` | 35→27 dim, removed context features |
| `src/routesmith/predictor/linucb.py` | Simplified API |
| `src/routesmith/predictor/__init__.py` | New exports |
| `src/routesmith/strategy/router.py` | New predictor types |
| `src/routesmith/config.py` | New PredictorConfig fields |
| `src/routesmith/client.py` | Removed reward_override from update() |
| `tests/test_features.py` | Updated for 27-dim |
| `tests/test_linucb.py` | Updated for simplified API |
| `tests/test_reward.py` | Updated for simplified API |
| `benchmark/run_linucb_27d.py` | NEW |
| `benchmark/run_linucb_fast.py` | NEW |
| `paper/` | Updated from paper branch |

## Next in Pipeline
1. CI gate passes on PR
2. Merge to dev
3. UAT branch + smoke tests
4. Tag v0.5.0
5. Publish PyPI + arXiv