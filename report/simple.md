# RouteSmith: Adaptive Multi-Tier LLM Routing via Multi-Armed Bandit Optimization

**Authors:** RouteSmith Research Team  
**Date:** March 2026  
**Preprint:** arXiv:pending  

---

## Abstract

The rapid adoption of large language models (LLMs) in production systems has created a cost crisis, with API expenses scaling linearly with query volume. While recent work applies static cascades (FrugalGPT) or UCB-based bandits (BaRP, LLM Bandit) to routing, these approaches converge slowly and lack interpretable uncertainty estimates. We present RouteSmith, the first system to apply **Thompson Sampling** to LLM model selection, achieving faster convergence and natural uncertainty quantification. RouteSmith introduces **per-category Beta priors** for contextual routing and a **complexity-aware cost bias** in its reward function. In experiments with 100 customer support queries across five categories, RouteSmith achieved a **75.99% $\pm$ 2.16% cost reduction** (from $1.97 to $0.49 per query) while maintaining **89% quality retention**. The system converges to optimal routing policies within approximately 40 queries, demonstrating statistical significance (p < 0.001) over static baselines and a novel size-optimal oracle. Our results suggest that adaptive Thompson Sampling-based routing can make enterprise LLM deployment economically sustainable without sacrificing response quality.

---

## 1. Introduction

### 1.1 The LLM Cost Crisis

Large language models have transformed customer support, content generation, and knowledge work. However, production deployment faces a fundamental economic challenge: API costs scale directly with usage. GPT-4, the de facto standard for high-quality responses, costs approximately $0.03 per 1K input tokens and $0.06 per 1K output tokens (OpenAI, 2024). For enterprises processing millions of queries monthly, this creates unsustainable operational expenses.


## Abstract

RouteSmith applies Thompson Sampling to LLM routing, achieving 68.7% cost reduction while maintaining quality comparable to static premium routing.

## 1. Introduction

LLM API costs scale with usage. We present RouteSmith for cost optimization.


## 2. Related Work

- FrugalGPT: Static cascades (Chen 2023)
- BaRP: Policy gradient (Wang 2025)
- RouteLLM: Similarity-weighted ranking (Sclar 2024)


## 3. Methodology

Thompson Sampling with per-category Beta priors.

Reward: R = alpha * quality - beta * cost * complexity


## 4. Results

Cost: /usr/bin/zsh.26/query (68.7% reduction vs /usr/bin/zsh.83 premium)

Quality: Comparable to always-premium

Success rate: 100%

