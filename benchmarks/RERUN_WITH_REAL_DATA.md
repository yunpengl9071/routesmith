# Rerunning the Benchmark with Real RouteLLM Data

This guide explains how to reproduce the CB-RouteSmith benchmark results using
the actual RouteLLM pre-computed evaluation data (GPT-4-Turbo vs Mixtral-8x7B
correctness labels on MMLU and GSM8K).

## Why real data matters

The synthetic data generator in `benchmarks/synthetic_data.py` is calibrated to
match published accuracy statistics (MMLU: 80.6% GPT-4, 68.1% Mixtral), but it
cannot replicate the exact per-question difficulty structure present in the real
MMLU dataset. Real data will:

- Produce the authoritative APGR values comparable to RouteLLM's paper figures
- Validate that the 27-dim feature extractor captures difficulty signals from
  real academic questions (not just synthetic ones)
- Confirm the APGR 0.583 claim for LinUCB warm=500 or revise it accurately

## Step 1: Clone the RouteLLM repository

```bash
git clone https://github.com/lm-sys/RouteLLM.git /path/to/routellm
cd /path/to/routellm
pip install -e ".[eval]"
```

## Step 2: Download the pre-computed eval responses

RouteLLM provides pre-computed model responses (GPT-4-Turbo and Mixtral
correctness on MMLU/GSM8K) to avoid re-running expensive inference:

```bash
cd /path/to/routellm
python -m routellm.evals.download_data
```

This creates:
- `routellm/evals/mmlu/responses/*.csv`   (57 subject CSV files, ~14K questions)
- `routellm/evals/gsm8k/gsm8k_responses.csv`   (1,319 questions)

Each CSV has columns:
```
prompt, gpt-4-1106-preview, mistralai/Mixtral-8x7B-Instruct-v0.1
```
where the model columns are `True`/`False` strings indicating correctness.

## Step 3: Run the benchmark

```bash
# From routesmith repo root:
ROUTELLM_DATA_DIR=/path/to/routellm python benchmarks/run_benchmark.py \
    --output results/benchmark_real_data.json
```

For a faster run (3 seeds instead of 10):
```bash
ROUTELLM_DATA_DIR=/path/to/routellm python benchmarks/run_benchmark.py \
    --fast --output results/benchmark_real_data.json
```

## Step 4: Compare with published numbers

The benchmark will print and save:

| Method | Expected APGR (real data) | Paper claim |
|--------|--------------------------|-------------|
| CB-RS-LinUCB (warm=0) | ~0.50 | 0.499 |
| CB-RS-LinUCB (warm=500) | ~0.55–0.60 | 0.583 |
| SW-Features (warm=500) | TBD | N/A |
| RouteLLM-MF (published) | 0.57 | 0.57 |
| RouteLLM-SW (published) | 0.80 | 0.80 |

## Step 5: Updating the paper

After running with real data, update `paper/main.tex`:

1. Replace synthetic-data results with real-data results in:
   - Table 1 (cold-start learning speed)
   - Table 2 (cumulative regret)
   - Table 3 (APGR warm-start ablation)

2. Update the abstract and conclusion with verified numbers.

3. If APGR values change, rerun `paper/` to regenerate the PDF:
   ```bash
   cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
   ```

## Alternative: Use OpenRouter API for fresh evaluation

If you have an OpenRouter API key, you can collect fresh correctness labels
on a smaller MMLU subset (e.g., 500 questions) using Claude 3 Haiku (cheap)
vs Claude 3 Opus (strong) instead of the original Mixtral/GPT-4 pair.

See `benchmarks/collect_labels_openrouter.py` for a script that does this.
This lets you evaluate on current models without needing the RouteLLM data files.

## What the synthetic benchmark validates

Even without real data, the synthetic benchmark confirms:

1. **Context-free bandits converge to always-strong** (verified: >99% strong routing)
2. **TS/UCB reduce regret vs random** (verified: ~50% reduction with binary accuracy regret)
3. **8× convergence gap between UCB c=0.5 and c=2.0** (verified in learning curves)
4. **LinUCB with 0 warm labels is equivalent to random** (APGR ≈ 0.50)
5. **LinUCB improves with warm-start labels** (direction correct; magnitude needs real data)

Claims that require real data for authoritative validation:
- Exact APGR figures (0.583 on MMLU, 0.610 on GSM8K)
- Whether 500 binary labels match RouteLLM-MF (0.57 APGR)
