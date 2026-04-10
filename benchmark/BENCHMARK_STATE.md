# RouteSmith Benchmark State

Last updated: 2026-04-08

## Overview

Running ICML-style benchmark on branch `routesmith-paper` to validate LinTS-27d
and LinUCB-27d routing algorithms against static baselines and RouteLLM-SW.

## Model Pair

- **Strong**: `openai/gpt-4o` ($2.50/$10.00 per 1M tokens)
- **Weak**: `openai/gpt-4o-mini` ($0.15/$0.60 per 1M tokens)
- All calls via **OpenRouter** (`https://openrouter.ai/api/v1`)

## Datasets

| Dataset | N | Cache file |
|---------|---|------------|
| MMLU (5 categories × 120) | 600 | `results/mmlu_600_queries.json` |
| GSM8K | 300 | `results/gsm8k_300_queries.json` |
| MBPP | 100 | `results/mbpp_100_queries.json` (unused in Exp 1) |

## Seeds

`SEEDS = [42, 43, 44, 45, 46]` (5 seeds for all bandit methods)

All seeds encoded in result filenames for reproducibility.

## Experiment 1: Binary Routing (MMLU + GSM8K)

### Strategies

| Strategy | Files pattern | Status |
|----------|--------------|--------|
| Static-Strong | `static_strong_{dataset}_results.json` | ✅ Complete |
| Static-Weak | `static_weak_{dataset}_results.json` | ✅ Complete |
| Random | `random_router_seed42_{dataset}_results.json` | ✅ Complete |
| RouteLLM-SW (t=0.30) | `routellm_sw_t0.30_{dataset}_results.json` | ✅ Complete |
| RouteLLM-SW (t=0.50) | `routellm_sw_t0.50_{dataset}_results.json` | ✅ Complete |
| RouteLLM-SW (t=0.70) | `routellm_sw_t0.70_{dataset}_results.json` | ✅ Complete |
| TS-Cat (5 seeds) | `ts_cat_seed{42-46}_{dataset}_results.json` | ✅ Complete |
| LinUCB-27d (5 seeds) | `linucb_27d_alpha1.5_seed{42-46}_{dataset}_results.json` | 🔄 In progress (MMLU done, GSM8K running) |
| LinTS-27d (5 seeds) | `lints_27d_vsq1.0_seed{42-46}_{dataset}_results.json` | ⏳ Pending |

### Results (completed strategies)

| Strategy | MMLU Acc | MMLU PGR | GSM8K Acc | GSM8K PGR | Cost ratio |
|----------|----------|----------|-----------|-----------|------------|
| Static-Strong | 77.7% | 1.00 | 97.3% | 1.00 | 1.0× |
| Static-Weak | 73.2% | 0.00 | 91.7% | 0.00 | 0.06× |
| Random | 75.8% | 0.59 | 94.0% | 0.41 | ~0.5× |
| RouteLLM-SW (t=0.30) | 72.7% | -0.11 | 92.3% | 0.11 | ~0.06× |
| RouteLLM-SW (t=0.50) | 72.7% | -0.11 | 92.0% | 0.05 | ~0.06× |
| RouteLLM-SW (t=0.70) | 72.2% | -0.23 | 90.3% | -0.24 | ~0.06× |

**Key finding**: RouteLLM-SW routes 0% queries to strong model on both datasets.
SW win-rate scores cluster at 0.218–0.233 (below all thresholds) because MMLU/GSM8K
queries are out-of-distribution for Chatbot Arena embeddings. APGR ≈ 0, making it
equivalent to always-weak at best. This is a transferability failure, not a bug.

## Experiment 2: 5-Arm Multi-Model Routing

⏳ **Pending** — starts after Exp 1 completes.

Arms: GPT-4o, Claude-Sonnet-4-5, Qwen-Plus, MiniMax-M1, GLM-4-Plus  
Seeds: [42, 43, 44] (3 seeds for cost control)  
Result files: `exp2_lints_5arm_seed{42-44}_results.json`

## Ablation Experiments

⏳ **Pending** — starts after Exp 2.

Defined in `benchmark/experiments/ablations.py`.

## How to Resume / Rerun

```bash
# Resume all remaining experiments (safe to rerun anytime)
bash scripts/run_experiments.sh all

# Run a specific phase
bash scripts/run_experiments.sh exp1_static
bash scripts/run_experiments.sh exp1_ts_cat
bash scripts/run_experiments.sh exp1_linucb
bash scripts/run_experiments.sh exp1_lints
bash scripts/run_experiments.sh exp2
bash scripts/run_experiments.sh ablations

# Check status
bash scripts/run_experiments.sh status

# Recovery: each strategy saves after every query
# Kill at any time and rerun — resumes from last saved query
```

## Per-Query Data Saved (for analysis)

All bandit strategies save per query:
- `query_id`, `dataset`, `category`, `strategy`
- `arm_chosen`, `routing_decision` (strong/weak)
- `correct`, `weak_correct` (enables PGR and counterfactual analysis)
- `cost_usd`, `weak_cost_usd`, `strong_cost_usd`
- `prompt_tokens`, `completion_tokens`

Strategy-specific:
- **LinTS**: `router_state` (full mu + Sigma posterior per arm) — enables posterior evolution plots
- **LinUCB**: `ucb_scores` — enables confidence bound evolution plots
- **TS-Cat**: `prior_state` (alpha/beta per category) — enables per-category learning curves

Learning trajectories can be reconstructed from array order (index = timestep t).

## Paper TODOs

- [ ] Wait for all experiments to complete
- [ ] Run `make figures` to regenerate all 9 figures with real data
- [ ] Fill real numbers into `paper/sections/` (Table 1, Table 2, key metrics)
- [ ] Compile PDF: `bash paper/build.sh`
- [ ] Discuss model pair for Exp 2 / real-world scenario (GPT-4.1/mini vs Gemini 2.5 Pro/Flash etc.)

## Backed-up Results

`benchmark/results_gpt4o/` — identical copy of all completed results (same model pair).
Safe to delete once paper is done.
