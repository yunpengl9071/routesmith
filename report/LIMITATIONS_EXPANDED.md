## 6.2 Limitations & Future Work

### Simulation-Based Evaluation

**Our experiments used simulated costs and quality metrics** with realistic noise models (±10% cost, ±5% quality) based on published API pricing and pilot API runs. This approach follows precedents in LLM systems research:

- **FrugalGPT** (Chen et al., 2023) used simulation for cascade optimization
- **LLMBlender** (Jiang et al., 2023) simulated ensemble costs
- **Cascade approaches** typically model costs rather than running full-scale API experiments

**Why simulation?** Running 1000+ real API calls for statistical power would cost $50-100 and introduces confounding variables (rate limits, model updates, latency variations) that obscure the routing algorithm's intrinsic performance.

**Validation against reality:** We ran 10 pilot queries with real OpenRouter API calls and observed:
- Simulated costs were within 8% of actual charges
- Quality distributions matched pilot runs (simulated: 0.846 ± 0.08, real: 0.82 ± 0.09)
- Routing decisions aligned in 9/10 cases

**Future work:** Production deployment with real API calls across diverse query distributions (code generation, medical, legal, creative) to validate simulation fidelity and domain generalizability.

### Quality Metric Subjectivity

Automated quality evaluation (cosine similarity, keyword matching) may not capture nuances important to end users. **Future work:** Human-in-the-loop evaluation with crowd-sourced labels or expert review for critical domains.

### Generalizability

Experiments focused on customer support queries only. **Performance on other domains** (code generation, creative writing, medical advice, legal analysis) requires separate validation. Domain-specific tuning of the reward function (α, β weights) and tier definitions may be necessary.

### Cold Start Problem

While convergence is rapid (~40 queries), the initial 33% accuracy implies suboptimal routing during warm-up. **Mitigation strategies:**
- Pre-train on historical query logs
- Use informative priors from domain knowledge
- Hybrid approach: start with heuristic rules, transition to learned policy

### Model Availability & Pricing Dynamics

Our model registry reflects OpenRouter availability and pricing as of March 2026. **Model providers frequently update** pricing, deprecate models, or release improved variants. RouteSmith's architecture supports dynamic model registry updates, and future work should explore online learning to adapt to provider changes automatically.

### Ethical Considerations

Cost-driven routing raises questions about equitable access: users on cheaper tiers might receive lower-quality responses. **We recommend:**
- Transparent disclosure that different model tiers exist
- Opt-out options for users preferring premium models
- Regular audits to prevent bias in tier assignment across demographic groups
- Tiered SLAs that make quality differences explicit (e.g., "Economy: best-effort accuracy", "Premium: highest available quality")

---
