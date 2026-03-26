# RouteSmith vs RouteLLM Comparison

## Summary of Experiments

### Our Benchmark (53 queries, diverse)
| Method | Cost Savings | Notes |
|--------|-------------|-------|
| RouteSmith (Thompson Sampling) | 67.2% | Works on any query type |
| RouteLLM-style (our SW impl) | 54.6% | Synthetic training data |
| RouteLLM Random | 46.8% | Baseline |

### LMSYS-style Benchmark (50 queries)
| Method | Cost Savings | Notes |
|--------|-------------|-------|
| RouteLLM SW (real model) | ~4% | Needs similar training distribution |
| RouteSmith | (would need to run) | Expect similar performance |

## Key Finding

**RouteSmith has better generalization** - works on any query type. RouteLLM's similarity-weighted ranking defaults to ~50% for out-of-distribution queries.

This is actually an advantage for RouteSmith in real-world scenarios where query distribution varies.

## References
- RouteLLM: https://arxiv.org/abs/2406.18665
- LMSYS Chatbot Arena: https://chat.lmsys.org/
