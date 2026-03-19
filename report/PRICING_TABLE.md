# RouteSmith Model Pricing Table (March 2026)

## Verified Current Pricing (from OpenRouter, OpenAI, Anthropic, Google)

### Tier 1: Premium (Best Capability)
| Model | Provider | Input$/1M | Output$/1M | Context |
|-------|----------|-----------|------------|---------|
| GPT-4o | OpenAI | $2.50 | $10.00 | 128K |
| Claude Sonnet 4.6 | Anthropic | $3.00 | $15.00 | 200K |
| Gemini 2.5 Pro | Google | $1.25 | $10.00 | 1M |
| GPT-5.4 Pro | OpenAI | $30.00 | $180.00 | 1M |

### Tier 2: Standard (Balanced)
| Model | Provider | Input$/1M | Output$/1M | Context |
|-------|----------|-----------|------------|---------|
| GPT-4o-mini | OpenAI | $0.15 | $0.60 | 128K |
| DeepSeek V3.2 | DeepSeek | $0.27 | $0.41 | 128K |
| Llama 3.1 70B | Meta | $0.40 | $0.40 | 128K |
| Mistral Large 3 | Mistral | $0.50 | $1.50 | 262K |

### Tier 3: Budget (Cost-Optimized)
| Model | Provider | Input$/1M | Output$/1M | Context |
|-------|----------|-----------|------------|---------|
| Gemini 2.0 Flash Lite | Google | $0.075 | $0.30 | 1M |
| Phi-4 | Microsoft | $0.06 | $0.14 | 16K |
| Qwen 2.5 7B | Qwen | $0.04 | $0.10 | 32K |
| Gemma 3 4B | Google | $0.02 | $0.04 | 32K |

### Free / Ultra-Low Cost
| Model | Provider | Input$/1M | Output$/1M | Context |
|-------|----------|-----------|------------|---------|
| Nemotron 3 Nano | NVIDIA | $0.00 | $0.00 | 32K |
| MiMo V2 Flash | Xiaomi | $0.00 | $0.00 | 256K |

---

## Key Insights

1. **DeepSeek V3.2** = 90% of GPT-4 capability at 1/50th cost
2. **Gemini 2.0 Flash Lite** = 1M context for $0.30/1M output
3. **Free models (Nemotron, MiMo)** now handle simple tasks well
4. **Price spread:** 100x difference between premium and budget

## Baseline Cost Comparisons (100 queries, ~500 tokens avg)

| Model | Cost/100 queries | Notes |
|-------|------------------|-------|
| GPT-4o (all premium) | $6.25 | Baseline max |
| Claude Sonnet | $9.00 | Premium |
| GPT-4o-mini | $0.38 | Standard |
| DeepSeek V3.2 | $0.34 | Best value |
| Nemotron (free) | $0.00 | Free |

## Target: RouteSmith achieves better quality at lower cost than any single model

---

*Updated: 2026-03-11*
*Sources: OpenRouter.ai, OpenAI pricing, Anthropic pricing, Google pricing*
