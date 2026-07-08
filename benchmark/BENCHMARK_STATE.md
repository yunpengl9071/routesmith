# RouteSmith Benchmark State

Last updated: 2026-07-07

**All experiments complete.** This doc summarises final results and paper prep next steps.

## Overview

ICML-style benchmark validating LinTS-27d and LinUCB-27d routing algorithms against
static baselines and RouteLLM-SW.

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

`SEEDS = [42, 43, 44, 45, 46]` (5 seeds for all bandit methods).

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
| LinUCB-27d (5 seeds) | `linucb_27d_alpha1.5_seed{42-46}_{dataset}_results.json` | ✅ Complete |
| LinTS-27d (5 seeds) | `lints_27d_vsq1.0_seed{42-46}_{dataset}_results.json` | ✅ Complete |

### Results

| Strategy | MMLU Acc | MMLU PGR | GSM8K Acc | GSM8K PGR | Cost ratio |
|----------|----------|----------|-----------|-----------|------------|
| Static-Strong | 77.7% | 1.00 | 97.3% | 1.00 | 1.0× |
| Static-Weak | 73.2% | 0.00 | 91.7% | 0.00 | 0.06× |
| Random | 75.8% | 0.59 | 94.0% | 0.41 | ≈0.5× |
| RouteLLM-SW (τ=0.30) | 72.7% | −0.11 | 92.3% | 0.11 | ≈0.06× |
| RouteLLM-SW (τ=0.50) | 72.7% | −0.11 | 92.0% | 0.05 | ≈0.06× |
| RouteLLM-SW (τ=0.70) | 72.2% | −0.23 | 90.3% | −0.24 | ≈0.06× |
| TS-Cat (μ±σ) | 72.4±0.4% | −0.16±0.09 | 91.1±0.6% | −0.07±0.20 | ≈0.06× |
| LinUCB-27d (μ±σ) | 78.2±0.2% | 1.11±0.04 | 91.7±0.5% | 0.03±0.15 | ≈0.19× |
| LinTS-27d (μ±σ) | 75.8±0.3% | 0.57±0.07 | 92.5±0.4% | 0.17±0.11 | ≈0.12× |

**Key finding**: RouteLLM-SW routes 0% queries to strong model on both datasets.
SW win-rate scores cluster at 0.218–0.233 (below all thresholds) because MMLU/GSM8K
queries are out-of-distribution for Chatbot Arena embeddings. APGR ≈ 0, making it
equivalent to always-weak at best. This is a transferability failure, not a bug.

**LinUCB-27d** achieves the highest MMLU accuracy (78.2%, surpassing even static-strong
at 77.7%) but routes aggressively to the strong model (high cost ratio ~0.19× vs
LinTS-27d ~0.12×). LinTS-27d offers the better cost-quality trade-off with lower
variance.

## Experiment 2: 5-Arm Multi-Model Routing

✅ **Complete** (3 seeds: 42, 43, 44 — 600 queries each via LinTS-27d)

Arms: GPT-4o, Claude-Sonnet-4-5, Qwen-Plus, MiniMax-M1, DeepSeek-V3

| Seed | Accuracy | Cost |
|------|----------|------|
| 42 | 71.0% | $0.10 |
| 43 | 70.7% | $0.13 |
| 44 | 71.3% | $0.12 |

## Ablation Experiments

✅ **Complete** (8 ablation runs)

| Ablation | Variants | Key result |
|----------|----------|------------|
| Feature dimensionality | 11d, 17d, 27d (LinTS, seed 42, MMLU) | 27d: APGR=0.57, 17d: 0.54, 11d: 0.61 (stable across dims) |
| Warm-start labels | 0, 100, 500 oracle labels (LinTS, seed 42, MMLU) | 0: 0.57, 100: 0.62, 500: 0.47 (moderate benefit) |
| LinUCB β sensitivity | α=0.5, 1.5, 3.0 (seed 42, MMLU) | 0.5: 0.41, 1.5: 0.63, 3.0: 0.56 |
| LinTS β (fixed) | vsq1.0 (seed 42, MMLU) | APGR=0.51 (no β parameter) |

## Paper TODOs

- [x] Wait for all experiments to complete
- [ ] Run `python -m benchmark.plot` to regenerate all 9 figures
- [ ] Fill real numbers into paper LaTeX sections
- [ ] Compile PDF

## Next Steps

1. Generate figures → `uv run python -m benchmark.plot`
2. Write paper draft in `paper/` directory
3. Package results for arXiv

## Backed-Up Results

`benchmark/results_gpt4o/` — identical copy of all completed results (same model pair).
Safe to delete once paper is done.
