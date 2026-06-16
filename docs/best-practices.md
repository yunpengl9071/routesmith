# Best Practices

## Model Registration

- Use **real pricing** from provider dashboards
- Set `quality_score` from eval benchmarks, or start conservative (0.7-0.8)
- Add `capabilities` for tool_calling/vision models

## Strategy Selection

| Use Case | Strategy | Why |
|----------|----------|-----|
| General chat | `DIRECT` | Simple, fast, optimal per query |
| Cost-sensitive | `CASCADE` | Tries cheap first, escalates |
| High-stakes | `PARALLEL` | Multiple models, best result |
| Latency-critical | `SPECULATIVE` | Starts cheap, switches mid-stream |

## Budget Management

- Start with `daily_budget = monthly_budget / 30`
- Use `"fallback"` behavior for production (never fail)
- Monitor `routesmith_budget_remaining_usd` metric

## Feedback Loop

- Enable `feedback_sample_rate=1.0` during development
- Use `record_outcome()` for explicit feedback
- Monitor predictor diagnostics: `rs.router.predictor.diagnostics()`

## Production Checklist

- [ ] Structured JSON logging enabled
- [ ] Prometheus metrics exported
- [ ] Health checks configured
- [ ] Budget exceeded behavior configured
- [ ] Docker image built and scanned
- [ ] Nightly live tests passing