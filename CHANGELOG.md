# Changelog

All notable changes to RouteSmith will be documented in this file.

## [0.5.0-beta] — 2026-06-20

### Added
- **NeuralUCB predictor**: Shallow neural network with UCB exploration bonus for nonlinear reward modeling
- **REINFORCE predictor**: Policy-gradient predictor with entropy-regularized stochastic exploration
- **WarmStart LinUCB predictor**: LinUCB initialized from benchmark quality priors
- **Paper**: ICML-style research paper with benchmark results (compiles to 10-page PDF)
- **Benchmark runner scripts**: `run_linucb_27d.py`, `run_linucb_fast.py` for reproducible experiments
- **Latency micro-benchmark**: Verified <0.5ms P99 routing overhead for 5-arm deployment

### Changed
- **Feature vector**: Simplified from 35-dim to 27-dim (removed 8 context features per paper-validated results)
- **LinUCB API**: Simplified — removed `context`, `reward_override`, `add_arm`, `remove_arm`, `serialize_state`, `load_state`
- **Router**: Added `neural_ucb`, `reinforce`, `warmstart_linucb` predictor types
- **CI**: mypy now enforced (no `|| true` escape), added bandit security scan

### Fixed
- Two conflicting paper versions consolidated into single `main.tex`
- Table symbols fixed for proper LaTeX compilation

## [0.4.4] — 2026-06-19

### Added
- **Trust-but-verify**: Shadow execution that compares routed model against alternatives
- **CLI stats**: `routesmith stats --local --watch` with live updating
- **Dashboard TUI**: `routesmith dashboard` terminal UI for real-time routing analytics

## [0.4.3] — 2026-06-18

### Added
- **Quality polls**: Adaptive sampling for collecting user feedback on model quality
- **`rs.answer_poll()`**: Explicit feedback mechanism to close the quality loop
- **Recommendations engine**: `rs.recommendations()` for proactive model selection advice

## [0.4.2] — 2026-06-17

### Added
- **Auto-registration**: `RouteSmith.with_auto()` zero-config entry point with OpenRouter model discovery
- **Tradeoff parameter**: Per-request `tradeoff` (0-10) controlling cost vs quality
- **RouteSmith explanation**: `routesmith_explanation` field on every response with routing rationale
- **Conversation stickiness**: Model persistence within a conversation scope

## [0.3.0] — 2026-06-16

### Added
- **Enterprise features**: CostModel enum (ON_DEMAND, PROVISIONED, SELF_HOSTED), capacity tracking, provisioned-first routing
- **Compliance routing**: Tag-based filtering (HIPAA, SOC2, PCI) with `required_compliance` parameter
- **Budget enforcement**: FAIL, FALLBACK, QUEUE behaviors when budget exhausted
- **Multi-project isolation**: Per-project cost allocation and stats via `project` parameter on `RouteSmith.__init__()`

## [0.2.0] — 2026-06-15

### Added
- **Production hardening**: Prometheus metrics, Docker image, health/liveness/readiness endpoints
- **Circuit breaker**: Per-model failure detection with automatic backoff
- **Structured logging**: JSON-format log output for observability
- **Comprehensive documentation**: mkdocs site with setup guides, API reference, integration docs
- **Nightly live tests**: Automated smoke tests against real APIs

## [0.1.0] — 2026-04-06

### Added
- **Core routing engine**: Model Registry, DIRECT/CASCADE/PARALLEL/SPECULATIVE strategies
- **Local proxy server**: OpenAI-compatible HTTP API via `routesmith serve`
- **CLI**: `routesmith serve`, `routesmith stats`, `routesmith init`
- **Cost tracking**: Real-time cost estimation with counterfactual comparison
- **Response metadata**: Per-response routing decisions (which model, why, what cost)
- **Quality prediction**: Embedding-based, classifier-based, random forest, and adaptive predictors
- **Semantic cache**: Embedding-based response caching for similar queries
- **Feedback collection**: Quality signal extraction, SQLite storage, predictor online learning
- **LangChain integration**: `ChatRouteSmith(BaseChatModel)` with tool calling and streaming
- **DSPy adapter**: Proxy mode and native `RouteSmithLM`
- **CrewAI adapter**: Proxy mode and native chat model
- **AutoGen integration**: `routesmith_autogen_agents()` proxy-based agent pair
- **Anthropic/OpenClaw integration**: Drop-in replacement for `anthropic.Anthropic`