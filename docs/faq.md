# FAQ

## How much can I save?

40-60% on mixed workloads. Simple queries route to cheap models, complex ones to expensive models.

## Does it work with my provider?

Yes. RouteSmith wraps LiteLLM, which supports 100+ providers (OpenAI, Anthropic, Groq, Bedrock, Azure, etc.).

## How does routing work?

RouteSmith predicts which model will produce the best quality for each query, then selects the optimal model based on your cost/quality constraints.

## What if I don't know quality scores?

Start with conservative defaults (0.7-0.8). RouteSmith learns from production feedback and improves over time.

## Can I force a specific model?

Yes: `rs.completion(model="gpt-4o", messages=[...])`

## Does it add latency?

Routing overhead is <5ms P99. LLM calls take 500-5000ms — routing is imperceptible.

## Where is data stored?

Feedback and stats are stored in a local SQLite database. No data leaves your infrastructure.

## Can I use it with LangChain?

Yes: `from routesmith.integrations.langchain import ChatRouteSmith`

## What happens when budget is exceeded?

Configurable: fail with error, fall back to cheapest model, or queue.

## Is it production ready?

v0.2.0+ is production-ready with circuit breakers, structured logging, Prometheus metrics, and Docker support.