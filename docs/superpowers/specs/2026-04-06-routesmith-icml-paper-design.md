# RouteSmith ICML Paper — Design Spec

**Date:** 2026-04-06
**Branch:** `routesmith-paper` (new work here), merging into `dev` via PRs
**Status:** Approved by user, proceeding to implementation plan

---

## 1. Research Objective

Benchmark RouteSmith's contextual bandit routing against RouteLLM's static SW/MF methods and find where **online RL routing has a genuine structural moat**. Produce an ICML-quality paper (or equivalent venue) that can withstand peer review.

### Honesty Constraints (non-negotiable)
- Every number in the paper must trace back to a real API call recorded in a JSON result file under `benchmark/results/`. No simulated data, no illustrative plots.
- Figures are generated programmatically from those JSON files. ASCII art and hand-crafted numbers are forbidden.
- When RouteLLM-SW/MF numbers are cited from their paper (Ong et al., ICLR 2025), they must be labeled as "reported by Ong et al." with the exact experimental conditions noted. Our re-run numbers (if SW works) are labeled separately.
- The paper must explicitly state when a result is on a subset of MMLU/GSM8K (e.g., "300 of 14K MMLU questions") — never imply the full dataset was used if it wasn't.
- LinTS is presented as a new algorithm contribution. Its regret bound cites Abeille & Lazaric (2017); we do not claim to prove a new bound.
- The existing draft on `origin/feature/contextual-bandit-routing` had unverified/simulated numbers. The new paper **replaces** all those results with real API results. This is called out in a Reproducibility section.

---

## 2. Methods

### 2.1 Binary Routing (Experiment 1)

| Method | Category | Data required | Source |
|--------|----------|--------------|--------|
| Static-Strong (GPT-4o always) | Baseline | 0 | New run |
| Static-Weak (GPT-4o-mini always) | Baseline | 0 | New run |
| Random Router | Baseline | 0 | New run |
| RouteLLM-SW (τ=0.3, 0.5, 0.7) | Static pretrained | 55K Arena labels | New run (or fallback to reported) |
| TS-Cat | Context-free RL | 0 | New run |
| LinUCB-27d | Contextual RL | 0 (+opt warm-start) | Existing impl, new real-data run |
| **LinTS-27d** | Contextual RL | 0 (+opt warm-start) | **New algorithm, new run** |
| Oracle | Upper bound | Full ground truth | Computed from results |

### 2.2 Multi-Model Quality Routing (Experiment 2)

Five arms via OpenRouter. RouteLLM is excluded (structurally binary — this is one moat point).

| Arm | Model ID (OpenRouter) | Role |
|-----|-----------------------|------|
| 1 | `openai/gpt-4o` | Frontier reasoning |
| 2 | `anthropic/claude-sonnet-4-5` | Strong reasoning + coding |
| 3 | `qwen/qwen-plus` | Math + multilingual |
| 4 | `minimax/minimax-m1` | General + cost-efficient |
| 5 | `zhipuai/glm-4-plus` | Knowledge + Chinese |

Reward function (quality-dominant):
$$R_t = 0.85 \cdot \text{acc}(a_t, x_t) - 0.15 \cdot \frac{c_{a_t}}{c_{\max}}$$

Methods: TS-Cat (5-arm), LinUCB-27d (5-arm), LinTS-27d (5-arm).

### 2.3 LinTS Algorithm (New Contribution)

Linear Thompson Sampling maintains a Gaussian posterior per arm $a$:
$$p(\theta_a \mid \mathcal{D}_a) = \mathcal{N}(\mu_a, v^2 \Sigma_a)$$

where $\Sigma_a = (A_a)^{-1}$, $\mu_a = \Sigma_a b_a$, and $v^2$ is the noise variance (set to 1.0 by default — no tuning needed, unlike LinUCB's $\beta$).

At each step: sample $\tilde{\theta}_a \sim \mathcal{N}(\mu_a, v^2 \Sigma_a)$ for each arm, select $a^* = \arg\max_a \phi(x_t)^\top \tilde{\theta}_a$.

Regret bound: $O(d\sqrt{T \log T})$ with high probability (Abeille & Lazaric, 2017), same order as LinUCB but with no $\beta$ hyperparameter.

**Structural advantage over LinUCB:** LinTS explores proportionally to posterior variance, which decays naturally as data accumulates. LinUCB's UCB bonus requires manual $\beta$ tuning; wrong $\beta$ causes either over-exploration (slow convergence) or under-exploration (suboptimal routing). The paper shows an ablation confirming this: LinUCB at $\beta \in \{0.5, 1.5, 3.0\}$ vs LinTS (no $\beta$).

### 2.4 Feature Space

27-dimensional features from existing `src/routesmith/predictor/features.py` (contextual-bandit-routing branch). Dimensions:
- [0–10]: 11 original message features (lengths, counts, question marks, etc.)
- [11–16]: 6 extended features (math/reasoning/code/creative keyword scores, difficulty estimate, vocabulary richness)
- [17–24]: 8 model features (cost rates, quality prior, latency, context window, capability flags)
- [25–26]: 2 interaction features (estimated response tokens, difficulty × quality prior)

All L2-normalized before input to bandit predictors. Feature ablation (27d vs 17d vs 11d) is an ablation experiment.

---

## 3. Benchmarks & Metrics

### 3.1 Datasets

| Dataset | Task | Metric | Size used | Full dataset size | Ground truth |
|---------|------|--------|-----------|------------------|-------------|
| MMLU (`cais/mmlu`) | Knowledge MCQ | Accuracy | 300 questions, 5 categories × 60 | 14,042 test | Label in dataset |
| GSM8K (`openai/gsm8k`, `main`) | Math word problems | Exact numeric match (after `####`) | 150 questions | 1,319 test | Label in dataset |
| MBPP (`google-research-datasets/mbpp`, `sanitized`) | Python coding | Pass@1 (local execution) | 50 problems (random seed=42 sample) | 257 test | Unit tests in `test_list` |

MBPP pass@1: generate code, execute it locally against `test_list` assertions with `subprocess` in a 10s timeout sandbox. No LLM judge needed — this is a fully objective coding metric. We explicitly report that 50 of 257 sanitized test problems are used.

### 3.2 Primary Metrics

- **Accuracy**: fraction of questions answered correctly (or tests passing for MBPP)
- **Cost (USD)**: actual tokens × OpenRouter pricing, recorded from API response
- **APGR** (Area under Performance-Gap Recovery): RouteLLM's metric, computed for Experiment 1 to enable direct comparison with their published numbers. Definition: let $q_s$ = strong-model accuracy, $q_w$ = weak-model accuracy, $q_r$ = router accuracy. $\text{PGR} = (q_r - q_w) / (q_s - q_w)$. APGR = area under the PGR-vs-cost-reduction curve.
- **Cumulative regret**: $\sum_{t=1}^T (r_t^* - r_t)$ where $r_t^*$ is the oracle reward at step $t$
- **Convergence query**: first query $t$ where rolling-100-window accuracy stays within 2pp of oracle

### 3.3 Statistical Rigor

- All bandit methods: 5 independent seeds → mean ± 95% CI (bootstrap, 1000 samples)
- RouteLLM-SW/MF: deterministic for fixed threshold → point estimate with experimental conditions noted
- Primary comparisons: paired t-test (RouteSmith LinTS vs RouteLLM-SW on same 300 queries)
- Effect sizes: Cohen's d reported alongside p-values
- Bonferroni correction for multiple comparisons across 3 datasets

---

## 4. The RL Moat — Paper Narrative

Four distinct structural advantages of online RL routing, each backed by an experiment or ablation:

**M1 — No pretraining data.** RouteLLM-SW/MF require 55K+ human preference labels before any query can be routed. LinTS requires 0 labels, improves from query 1. *Evidence: cold-start learning curves in Experiment 1.*

**M2 — N-arm natural extension.** RouteLLM is architecturally binary (strong/weak). Extending to K models requires K(K-1)/2 separate routers or re-training. LinUCB/LinTS scale to K arms with O(Kd²) storage, no re-training. *Evidence: Experiment 2 (5 arms).*

**M3 — Automatic exploration.** Static routers (RouteLLM, FrugalGPT) never test under-explored model-query pairs. LinTS's posterior sampling automatically identifies high-uncertainty routes. *Evidence: regret curves and exploration analysis.*

**M4 — Interpretability.** LinUCB's learned $\hat{\theta}_a$ weights are directly readable as feature importances. We show which features drive routing to each model (e.g., "high math score → Qwen-Plus"). RouteLLM is a black-box classifier. *Evidence: feature importance figure.*

---

## 5. Codebase Architecture

### 5.1 What is Reused (from existing branches)

From `feature/contextual-bandit-routing`:
- `src/routesmith/predictor/features.py` — 27-dim FeatureExtractor (**reuse as-is**)
- `src/routesmith/predictor/linucb.py` — LinUCB predictor (**reuse, adapt for benchmark harness**)
- `paper/main.tex` — ICML LaTeX skeleton (**reuse template, replace all content**)
- `paper/references.bib` — citation library (**reuse, add LinTS citation**)

From `feature/adaptive-routing-model` (on dev):
- `src/routesmith/predictor/learner.py` — TS-Cat (**reuse, adapt for benchmark**)

### 5.2 New Files

```
benchmark/
├── config.py                  # API key, model IDs, pricing, paths
├── dataset.py                 # MMLU + GSM8K + MBPP loaders (cached JSON)
├── harness.py                 # Orchestrator: runs any strategy on any dataset
├── metrics.py                 # Accuracy, cost, APGR, regret computation
├── strategies/
│   ├── static.py              # Static-Strong, Static-Weak, Random
│   ├── routellm_sw.py         # RouteLLM-SW wrapper (real run + fallback to reported)
│   ├── ts_cat.py              # Context-free TS (2-arm and K-arm)
│   ├── linucb.py              # LinUCB-27d wrapper (thin harness adapter)
│   └── lints.py               # LinTS-27d (NEW — Gaussian posterior TS)
├── experiments/
│   ├── exp1_binary.py         # Experiment 1: binary routing on MMLU+GSM8K
│   ├── exp2_multimodel.py     # Experiment 2: 5-arm quality routing
│   └── ablations.py           # Feature dim, warm-start, beta, N-arm scaling
├── plot.py                    # All figures (programmatic, from results JSON)
├── resume.py                  # Show current status of all experiments
└── results/                   # All JSON outputs (gitignored except summary)
    ├── .gitkeep
    └── [strategy]_[dataset]_seed[N].json

paper/                         # Built section-by-section then compiled
├── sections/
│   ├── 00_abstract.md
│   ├── 01_introduction.md
│   ├── 02_related_work.md
│   ├── 03_method.md           # Formulation + features + LinUCB + LinTS
│   ├── 04_exp1_binary.md      # Binary comparison results
│   ├── 04_exp2_multimodel.md  # Multi-model quality routing results
│   ├── 05_analysis.md         # Ablations + feature importance + convergence
│   └── 06_conclusion.md
├── main.tex                   # Master LaTeX (inputs each section .tex)
├── sections/                  # Auto-converted from MD sections by build.sh
├── figures/                   # All PNGs from benchmark/plot.py
├── references.bib
└── build.sh                   # MD → LaTeX → PDF compilation pipeline
```

### 5.3 CI/CD

`.github/workflows/` (or equivalent local pre-commit + Makefile):

```
make lint          # ruff check benchmark/ src/
make type-check    # mypy benchmark/ src/routesmith/predictor/
make test-unit     # pytest tests/ (mocked, no API calls)
make verify-figs   # assert all figures exist and are >10KB (non-empty)
make paper-build   # pandoc MD→LaTeX + pdflatex compilation
```

Pre-commit hooks:
- `ruff` linting on all Python files
- `mypy` on `benchmark/` and modified predictor files
- `black` formatting

Each experiment phase committed separately:
- Phase 1 commit: dataset download + static baselines
- Phase 2 commit: RouteLLM-SW results
- Phase 3 commit: TS-Cat + LinUCB results
- Phase 4 commit: LinTS results
- Phase 5 commit: figures + paper sections (one commit per section)
- Phase 6 commit: full LaTeX compilation + PDF

---

## 6. Paper Structure (section-by-section)

Each section built as a standalone `.md` file first, then converted to LaTeX. Cross-section LaTeX dependencies (shared commands, `\label`/`\ref`) are handled in `main.tex` after all sections are complete.

### Section 00: Abstract (written last)
- 200 words max
- States all headline results with exact numbers
- Written only after all experiments complete

### Section 01: Introduction
- LLM routing problem + cost crisis
- RouteLLM's approach and its limitations (data requirements, binary-only, static)
- Our approach: online contextual bandits, no pretraining, N-arm
- Contributions list (4 items: LinTS algorithm, binary comparison, multi-model extension, feature importance analysis)

### Section 02: Related Work
- Supervised LLM routing (RouteLLM, FrugalGPT, AutoMix)
- Online learning / bandits (LinUCB, LinTS, Thompson Sampling)
- Multi-model LLM serving (FrugalGPT cascades, vLLM serving)
- What distinguishes our work (table: method vs. {online, N-arm, no labels, interpretable})

### Section 03: Method
- Problem formulation: contextual bandit (X, A, r)
- 27-dim feature space (table of all dimensions)
- Algorithm 1: LinUCB-27d (existing, restated for clarity)
- Algorithm 2: LinTS-27d (new)
- Reward function (Experiment 1: cost-quality balanced; Experiment 2: quality-dominant)
- TS-Cat as context-free baseline (non-contextual, per-category Beta priors)

### Section 04a: Experiment 1 — Binary Routing
- Setup (models, datasets, metrics, seeds)
- Table 1: Accuracy + cost + APGR (all methods, MMLU + GSM8K)
- Figure 1: Cost-quality Pareto frontier
- Figure 2: Cold-start learning curves (accuracy vs queries, 5 seeds ±1σ)
- Figure 3: Cumulative regret comparison
- Discussion: when LinTS beats LinUCB (no β tuning), when RouteLLM-SW wins (more data)

### Section 04b: Experiment 2 — Multi-Model Quality Routing
- Setup (5 models, 3 benchmarks, quality-dominant reward)
- Table 2: Per-model accuracy by benchmark (shows specialization)
- Table 3: Routing accuracy + cost comparison (TS-Cat vs LinUCB vs LinTS)
- Figure 4: Routing heatmap (which model selected per query category)
- Figure 5: LinUCB feature importance per model arm
- Discussion: N-arm moat (RouteLLM structurally excluded), quality-dominant routing behavior

### Section 05: Analysis & Ablations
- Figure 6: Feature dimensionality ablation (11d vs 17d vs 27d APGR)
- Figure 7: Warm-start label count vs APGR (0, 100, 500 labels)
- Figure 8: LinUCB β sensitivity (0.5, 1.5, 3.0 vs LinTS no-β)
- Figure 9: N-arm scaling (K=2,3,5 convergence speed)
- Distribution shift robustness: MMLU-trained router evaluated on GSM8K (cross-dataset transfer)

### Section 06: Conclusion
- Summary of headline results
- RL moat: 4 structural advantages restated
- Limitations (honest): cold-start gap vs pretrained RouteLLM, feature quality ceiling, MBPP subset size
- Future work: NeuralUCB (already implemented, needs benchmarking), production deployment, real-time distribution shift detection

---

## 7. Transparency & Limitations (to be stated explicitly in paper)

The paper must honestly state:
1. MMLU/GSM8K subset sizes used (not the full dataset — state exact N)
2. MBPP evaluation uses 50 problems (not the full 374 — state this)
3. RouteLLM-SW comparison: if SW router fails to load due to embedding API issues, state this and use their published APGR numbers from Ong et al. (2025) instead
4. The existing `origin/feature/contextual-bandit-routing` draft contained unverified/simulated results. All results in this paper are from real API calls. (State in reproducibility appendix.)
5. LinTS convergence guarantee (Abeille & Lazaric 2017) applies to stochastic linear bandits; routing is approximately stochastic but may have non-stationary reward when new models or query distributions arrive — state this caveat
6. Multi-model experiment uses only OpenRouter-available models; results may differ on other providers or direct API access

---

## 8. CI/CD Practices

### Testing Strategy
- **Unit tests** (no API calls): `tests/` — mock all LLM calls, test feature extraction, metric computation, LinTS posterior update math
- **Integration tests** (1 real API call each): `tests/integration/` — run in CI with real key, skip if key not set
- **Smoke tests**: `benchmark/resume.py` — shows which experiments are complete

### Commit Discipline
- One logical change per commit (e.g., "feat(benchmark): add LinTS strategy" not "add all strategies")
- No large JSON result files in git (gitignored); only summary JSONs committed
- All PRs target `dev`, not `main`
- Paper section files committed one at a time

### Makefile Targets
```makefile
lint:           ruff check benchmark/ src/routesmith/predictor/
type-check:     mypy benchmark/ src/routesmith/predictor/linucb.py src/routesmith/predictor/lints.py
test:           pytest tests/ -m "not api" -v
test-api:       pytest tests/integration/ -v
verify-results: python3 benchmark/resume.py
figures:        python3 benchmark/plot.py
paper-sections: bash paper/build_sections.sh
paper-pdf:      bash paper/build.sh
```

---

## 9. API Budget Estimate

| Experiment | Queries | API calls | Estimated cost |
|------------|---------|-----------|----------------|
| Exp 1: Static baselines (2 models × 450 queries) | 450 | 900 | ~$0.50 |
| Exp 1: RouteLLM-SW (1 threshold × 450) | 450 | 450+embed | ~$0.30 |
| Exp 1: TS-Cat + LinUCB + LinTS (5 seeds × 450) | 2250 | 3375 (weak always called) | ~$2.00 |
| Exp 2: 5 models × 300 queries × 3 strategies × 3 seeds | 4500 | 9000 | ~$4.50 |
| Ablations (feature dims, warm-start, β) | ~1000 | ~1500 | ~$1.00 |
| **Total** | | | **~$8-12** |

Rate limiting: 1.5s between calls. Batch by dataset to avoid context window overflow.

---

## 10. Out of Scope

- NeuralUCB: implemented in existing branch, **not benchmarked in this paper** (insufficient queries for stable neural training on 300-query budget; stated as future work)
- Production deployment / real user traffic
- Fine-tuning any of the 5 frontier models
- Multi-turn conversation routing (single-turn only)
- Latency optimization (routing overhead measured but not primary metric)
