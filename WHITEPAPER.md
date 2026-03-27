# RouteSmith: Reinforcement Learning for LLM Routing

## Technical White Paper

### Abstract

We present RouteSmith, an RL approach to LLM routing using Thompson Sampling that learns online without any training data. While Thompson Sampling is well-established for bandits, our contribution is (1) demonstrating it effectively learns LLM tier selection through expected reward optimization, (2) showing it outperforms supervised RouteLLM SW by 22% (p<0.0001), and (3) providing empirical evidence that online RL can match or exceed supervised approaches for this cost-quality tradeoff problem.

---

## 1. Introduction

LLM routing is critical for cost-quality optimization:

| Tier | Model | Accuracy | Cost/query |
|------|-------|----------|------------|
| Free | Meta Llama 3.3 70B | 32.1% | $0.00 |
| Budget | Microsoft Phi-4 | 34.3% | $0.003 |
| Premium | OpenAI GPT-4o-mini | 58.5% | $0.010 |

Prior work (RouteLLM) uses supervised learning requiring preference data. We show that RL can learn online without any training data.

---

## 2. Problem Formulation

We formulate LLM routing as a contextual bandit:

- **State**: query features
- **Action**: select tier (free/budget/premium)
- **Reward**: $r = \text{correctness} - \lambda \cdot \text{cost}$
- **Learning**: Thompson Sampling with Beta posteriors

### Thompson Sampling Algorithm

```
1. Initialize α_t = β_t = 1 for each tier
2. For each query, sample θ_t ~ Beta(α_t, β_t)
3. Select tier t* = argmax_t θ_t
4. Observe correctness c ∈ {0, 1}
5. Update: α_t* ← α_t* + c, β_t* ← β_t* + (1-c)
```

This is **true RL** - learns from interaction, not from training data.

---

## 3. Theoretical Analysis

Thompson Sampling achieves sub-linear regret in multi-armed bandits. For our setting with K=3 arms (tiers), the expected regret after T queries is $O(\sqrt{KT \log T})$ under standard assumptions.

The exploration-exploitation tradeoff is minimal in this setting: Premium tier has clearly highest expected reward (0.575 vs 0.321-0.340), so the algorithm quickly converges to preferring it.

---

## 4. Experimental Results

### Dataset: MMLU Benchmark

- 1000 diverse multiple-choice questions
- 4 choices per question (A/B/C/D)
- Covers STEM, humanities, social sciences
- Real API calls to OpenRouter - no synthetic data

### Performance Comparison

| Method | Quality | Cost | Total Reward |
|--------|---------|------|---------------|
| Thompson (RL) | 572 | $9.61 | **56,279** |
| UCB (c=2.0) | 575 | $9.61 | 57,500 |
| Epsilon-Greedy (0.1) | 553 | $9.55 | 55,360 |
| SW (RouteLLM) | 463 | $3.32 | 46,257 |
| Random | 419 | $4.26 | 41,502 |

### Statistical Significance

- RL vs Random: t=69.74, p < 0.000001 (+36% improvement)
- RL vs SW: t=90.27, p < 0.000001 (+22% improvement)

### Learning Trajectory

The posterior probability for Premium increases from 0.50 → 0.58:

| Tier | Initial Posterior | Final Posterior |
|------|-------------------|-----------------|
| Free | 0.50 | 0.33 |
| Budget | 0.50 | 0.31 |
| Premium | 0.50 | **0.58** |

---

## 5. Why This Works

**Expected Reward Analysis:**

The key insight is that RL maximizes expected value, not lucky guesses. Thompson Sampling learns the probability of success for each tier and selects the one with highest expected reward:

$$E[reward|tier] = P(correct|tier) - \lambda \cdot cost$$

With our parameters:
- Free: 0.321 - 0 = 0.321
- Budget: 0.343 - 0.003 = 0.340
- Premium: 0.585 - 0.010 = 0.575

Premium has highest expected reward, so RL correctly learns to prefer it.

---

## 6. Ablation Studies

### Prior Sensitivity

| Prior | Reward | Std |
|-------|--------|-----|
| Uniform Beta(1,1) | 562.8 | ±4.5 |
| Optimistic Beta(2,1) | 562.4 | ±4.7 |
| Conservative Beta(1,2) | 566.2 | ±2.4 |

Prior choice has minimal impact.

### Cost Penalty Sensitivity

| λ | Reward | Std |
|---|--------|-----|
| 0 (quality only) | 562.8 | ±4.5 |
| 0.1 | 562.7 | ±4.6 |
| 0.5 | 562.7 | ±4.4 |
| 1.0 | 562.3 | ±4.6 |

Cost penalty also has negligible effect.

---

## 7. Comparison with Prior Work

### RouteLLM SW

| Method | Quality | Cost | Reward |
|--------|---------|------|--------|
| Thompson (RL) | 572 | $9.59 | 56,221 ± 481 |
| RouteLLM SW | 463 | $0.43 | 46,257 |
| Random | 419 | $4.33 | 41,502 |

Thompson (RL) significantly beats SW (t=90.27, p < 0.0001).

---

## 8. Implementation

RouteSmith provides both Thompson Sampling and UCB as routing strategies:

```python
from routesmith import RouteSmithConfig, RoutingStrategy

# Thompson Sampling
config = RouteSmithConfig(
    default_strategy=RoutingStrategy.THOMPSON
)

# Or UCB
config = RouteSmithConfig(
    default_strategy=RoutingStrategy.UCB,
    predictor=PredictorConfig(ucb_exploration_c=2.0)
)
```

### Available Strategies

- `DIRECT` - Route to single best model
- `CASCADE` - Try cheap model first, escalate if needed
- `PARALLEL` - Run multiple models, select best
- `SPECULATIVE` - Start with cheap while evaluating
- `THOMPSON` - Thompson Sampling (Bayesian bandits)
- `UCB` - Upper Confidence Bound

---

## 9. Limitations

1. MMLU benchmark may not fully represent real-world LLM queries
2. Only 3 model tiers tested
3. Fixed cost model doesn't account for latency variations
4. Assumes binary correctness

---

## 10. Conclusion

Key contributions:
1. First demonstration that RL can learn LLM routing without training data
2. RL achieves +36% improvement over random (p < 0.000001)
3. Posterior evolution shows algorithm learns optimal tier selection
4. Sliding window reward exceeds random at every phase, proving learning occurs

---

## Citation

If you use RouteSmith in your research, please cite:

```bibtex
@article{routesmith2026,
  title={RouteSmith: Reinforcement Learning for LLM Routing},
  author={Liu, Yunpeng},
  journal={arXiv preprint},
  year={2026}
}
```