# RouteSmith Tutorial

RouteSmith is an adaptive routing layer that sits between your code and LLM provider APIs. Instead of hardcoding which model to call, you register a pool of models and RouteSmith picks the right one for each request — routing cheaper models when they'll do, escalating to stronger models when needed. It learns from feedback and improves over time.

This tutorial walks through everything from a first request to production-grade multi-agent setups.

## Contents

1. [Installation](#1-installation)
2. [First request in 60 seconds](#2-first-request-in-60-seconds)
3. [Registering models](#3-registering-models)
4. [The completion API](#4-the-completion-api)
5. [Routing strategies](#5-routing-strategies)
6. [Budget constraints](#6-budget-constraints)
7. [Routing metadata](#7-routing-metadata)
8. [The feedback loop](#8-the-feedback-loop)
9. [Custom reward functions](#9-custom-reward-functions)
10. [Routing algorithms](#10-routing-algorithms)
11. [Semantic caching](#11-semantic-caching)
12. [Config file](#12-config-file)
13. [Framework integrations](#13-framework-integrations)
14. [Multi-agent systems](#14-multi-agent-systems)
15. [Capability-aware routing](#15-capability-aware-routing)
16. [State persistence](#16-state-persistence)
17. [CLI reference](#17-cli-reference)
18. [When RouteSmith is not the right fit](#18-when-routesmith-is-not-the-right-fit)

---

## 1. Installation

```bash
# Core Python API
pip install routesmith

# With the OpenAI-compatible proxy server
pip install "routesmith[proxy]"

# With specific framework integrations
pip install "routesmith[langchain]"
pip install "routesmith[anthropic]"
pip install "routesmith[dspy]"
pip install "routesmith[crewai]"
pip install "routesmith[autogen]"

# With semantic caching
pip install "routesmith[cache]"

# Everything
pip install "routesmith[proxy,langchain,anthropic,dspy,crewai,autogen,cache]"
```

Requires Python 3.10+.

---

## 2. First request in 60 seconds

### Option A: proxy (no code changes to existing apps)

```bash
# Generate config interactively
routesmith init

# Start the OpenAI-compatible proxy
routesmith serve
```

Point any OpenAI client at `http://localhost:9119`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9119/v1", api_key="any")
response = client.chat.completions.create(
    model="auto",  # RouteSmith picks the model
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.choices[0].message.content)
```

### Option B: Python API

```python
from routesmith import RouteSmith

rs = RouteSmith()
rs.register_model("openai/gpt-4o-mini",     cost_per_1k_input=0.15,  cost_per_1k_output=0.60,  quality_score=0.85)
rs.register_model("openai/gpt-4o",          cost_per_1k_input=5.0,   cost_per_1k_output=15.0,  quality_score=0.95)
rs.register_model("deepseek/deepseek-chat",  cost_per_1k_input=0.014, cost_per_1k_output=0.028, quality_score=0.72)

response = rs.completion(
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.choices[0].message.content)
print(rs.stats)
# {'request_count': 1, 'total_cost_usd': 0.000031, 'cost_savings_usd': 0.074, 'savings_percent': 71.2, ...}
```

---

## 3. Registering models

### Manual registration

```python
rs.register_model(
    model_id="openai/gpt-4o-mini",
    cost_per_1k_input=0.15,     # USD per 1000 input tokens
    cost_per_1k_output=0.60,    # USD per 1000 output tokens
    quality_score=0.85,         # prior quality estimate 0–1
    latency_p50_ms=800,         # optional latency hints
    latency_p99_ms=3000,
    context_window=128000,
)
```

`quality_score` is your prior belief about the model's quality. It's used for routing until enough feedback arrives to update the learned distribution. Set it conservatively — the bandit will explore and correct it.

### Factory methods

For common setups, factory methods pre-populate costs and capabilities:

```python
from routesmith.integrations.langchain import ChatRouteSmith

# Pre-loads gpt-4o, gpt-4o-mini, gpt-3.5-turbo with correct capabilities
llm = ChatRouteSmith.with_openai_models()

# Or Groq
llm = ChatRouteSmith.with_groq_models()

# Or Anthropic Claude tiers
llm = ChatRouteSmith.with_anthropic_models()
```

### Fetch pricing from OpenRouter

If you have an `OPENROUTER_API_KEY`, RouteSmith can pull current pricing directly:

```python
from routesmith.integrations.anthropic import RouteSmithAnthropic

# Fetches live pricing for haiku → sonnet → opus
client = RouteSmithAnthropic.with_openrouter_models()
```

Or via config file:

```yaml
# routesmith.yaml
openrouter_models:
  - openai/gpt-4o-mini
  - deepseek/deepseek-chat
  - anthropic/claude-3-haiku
```

### Deregistering models

```python
rs.deregister_model("openai/gpt-4o")
# Historical feedback records are preserved.
# The predictor arm is retired; the model won't receive new requests.
# Raises ValueError if it would remove the last registered model.
```

---

## 4. The completion API

### Synchronous

```python
response = rs.completion(
    messages=[{"role": "user", "content": "Explain recursion"}],
)
text = response.choices[0].message.content
```

### Async

```python
response = await rs.acompletion(
    messages=[{"role": "user", "content": "Explain recursion"}],
)
```

### Streaming

```python
for chunk in rs.completion_stream(
    messages=[{"role": "user", "content": "Write a haiku"}]
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

```python
async for chunk in rs.acompletion_stream(
    messages=[{"role": "user", "content": "Write a haiku"}]
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Bypassing routing

Pass `model=` to skip the router entirely:

```python
response = rs.completion(
    messages=[...],
    model="openai/gpt-4o",  # always uses this model
)
```

### Per-request overrides

```python
response = rs.completion(
    messages=[...],
    min_quality=0.90,     # only consider models with quality >= 0.90
    max_cost=0.01,        # only consider models costing <= $0.01/request
    strategy=RoutingStrategy.CASCADE,
)
```

---

## 5. Routing strategies

### Direct (default)

Picks the single best model for the request given current posteriors and constraints.

```python
from routesmith.config import RoutingStrategy

response = rs.completion(messages=[...], strategy=RoutingStrategy.DIRECT)
```

### Cascade

Tries the cheapest qualifying model first. If quality is too low, escalates to the next tier. Achieves high cost savings on easy queries while preserving quality on hard ones.

```python
response = rs.completion(messages=[...], strategy=RoutingStrategy.CASCADE)
```

Set the default in config:

```python
from routesmith.config import RouteSmithConfig, RoutingStrategy

config = RouteSmithConfig(default_strategy=RoutingStrategy.CASCADE)
rs = RouteSmith(config=config)
```

Or in YAML:

```yaml
routing:
  strategy: cascade
```

---

## 6. Budget constraints

### Per-request cap

```python
response = rs.completion(
    messages=[...],
    max_cost=0.05,   # will not route to any model costing > $0.05 for this request
)
```

### Global budget config

```python
from routesmith.config import RouteSmithConfig, BudgetConfig

config = RouteSmithConfig(
    budget=BudgetConfig(
        max_cost_per_request=0.10,
        max_cost_per_day=50.0,
        quality_threshold=0.75,  # minimum acceptable quality for any model
    )
)
rs = RouteSmith(config=config)
```

Or in YAML:

```yaml
budget:
  max_cost_per_request: 0.10
  max_cost_per_day: 50.0
  quality_threshold: 0.75
```

---

## 7. Routing metadata

To see what RouteSmith decided and why, enable metadata on individual calls or inspect the session stats.

### Per-request metadata

```python
response = rs.completion(messages=[...], include_metadata=True)
meta = response.routesmith_metadata
# {
#   "request_id": "a3f2b1c0",
#   "model_selected": "openai/gpt-4o-mini",
#   "routing_strategy": "direct",
#   "routing_reason": "best quality-cost tradeoff (quality=0.85, cost=$0.00038/1k)",
#   "routing_latency_ms": 0.8,
#   "estimated_cost_usd": 0.000031,
#   "counterfactual_cost_usd": 0.000742,
#   "cost_savings_usd": 0.000711,
#   "models_considered": ["openai/gpt-4o", "openai/gpt-4o-mini", "deepseek/deepseek-chat"],
# }
```

The `request_id` is what you pass to `record_outcome()` later.

### Session stats

```python
print(rs.stats)
# {
#   "request_count": 47,
#   "total_cost_usd": 0.31,
#   "estimated_without_routing": 1.73,
#   "cost_savings_usd": 1.42,
#   "savings_percent": 82.1,
#   "registered_models": 3,
#   "feedback_samples": 47,
#   "last_routing": {...},
# }
```

Via CLI:

```bash
routesmith stats
routesmith stats --json
```

### Last routing decision

```python
meta = rs.last_routing_metadata
print(meta.model_selected, meta.routing_reason, meta.cost_savings_usd)
```

---

## 8. The feedback loop

RouteSmith learns from explicit feedback. Every `record_outcome()` call updates the predictor's beliefs about which models work for which kinds of tasks.

### Recording outcomes

```python
# Binary
rs.record_outcome(request_id, success=True)
rs.record_outcome(request_id, success=False)

# Continuous quality score (0–1)
rs.record_outcome(request_id, score=0.9)
rs.record_outcome(request_id, score=0.2)  # partial success

# With free-text feedback (stored for diagnostics)
rs.record_outcome(request_id, success=False, feedback="Response was truncated mid-sentence")
```

Get the `request_id` from the response:

```python
response = rs.completion(messages=[...], include_metadata=True)
request_id = response.routesmith_metadata["request_id"]
# or
request_id = response._routesmith_request_id
```

### When to record feedback

Feedback is most useful when it reflects actual task outcome, not just response appearance. Good signal sources:

| Use case | Signal |
|---|---|
| Chatbot | Thumbs up / thumbs down, regeneration click |
| Code generation | pytest pass/fail, code runs without error |
| Data extraction | Downstream validation (schema check, type check) |
| Summarization | Reviewer accepts / rejects |
| Tool use | Tool call returns valid result |
| Multi-agent pipeline | Task completes end-to-end without escalation |

### Collecting IDs across a multi-step run

```python
request_ids = []

for step in pipeline:
    response = rs.completion(messages=step.messages, include_metadata=True)
    request_ids.append(response.routesmith_metadata["request_id"])

# After task outcome is known
quality = 1.0 if task_succeeded else 0.0
for rid in request_ids:
    rs.record_outcome(rid, score=quality)
```

---

## 9. Custom reward functions

By default RouteSmith treats the quality score you pass to `record_outcome()` as the reward. You can override this to weight cost, latency, or any combination.

### Expression syntax (YAML or string)

```yaml
# routesmith.yaml
feedback:
  reward: "quality - 0.3 * cost_normalized"
```

Available variables: `quality`, `cost_usd`, `cost_normalized` (0–1 relative to most expensive model), `latency_ms`, `tokens_in`, `tokens_out`, `model_id`. YAML expressions support `min`, `max`, `abs`, `round`. No arbitrary code execution.

### Python callable

```python
from routesmith.config import RouteSmithConfig

config = RouteSmithConfig(
    reward_fn=lambda quality, cost_normalized, latency_ms, **_:
        quality - 0.3 * cost_normalized - 0.1 * (latency_ms / 5000)
)
rs = RouteSmith(config=config)
```

The `**_` catch-all is important — new context variables may be added in future versions.

---

## 10. Routing algorithms

### LinTS (default for new installs)

Linear Thompson Sampling. Maintains a Gaussian posterior over routing quality per model, sampled at decision time. Naturally balances exploration (trying cheaper models) and exploitation (using what's known to work). No hyperparameter tuning required.

```python
from routesmith.config import RouteSmithConfig, PredictorConfig

config = RouteSmithConfig(
    predictor_type="lints",
    predictor=PredictorConfig(lints_v_sq=1.0),  # posterior variance scaling; rarely needs changing
)
```

### LinUCB

UCB-based contextual bandit. Achieves higher routing accuracy than LinTS in benchmarks (APGR 1.126 vs 0.593 on MMLU) at the cost of needing `alpha` tuning. Start with `alpha=1.5` and increase if the router is too conservative.

```python
config = RouteSmithConfig(
    predictor_type="linucb",
    predictor=PredictorConfig(
        linucb_alpha=1.5,        # exploration parameter (0.5–3.0 typical)
        linucb_cost_lambda=0.15, # cost penalty weight in reward
    ),
)
```

### Adaptive (random forest)

Falls back to `quality_score` priors during cold start, blends with a learned random forest once 100+ samples arrive, and transitions fully to the learned model after further data. Good for general-purpose use when you don't want to think about bandit hyperparameters.

```python
config = RouteSmithConfig(
    predictor_type="adaptive",
    predictor=PredictorConfig(
        min_samples_for_training=100,
        retrain_interval=50,
        blend_alpha=0.7,  # weight given to learned model vs prior during warm-up
    ),
)
```

### Choosing an algorithm

| Algorithm | Best for | Tradeoff |
|---|---|---|
| LinTS | Most use cases, no tuning needed | Slightly lower peak accuracy than LinUCB |
| LinUCB | When you have data and want max accuracy | Requires alpha tuning |
| Adaptive | When workload shifts over time | Slower to converge, requires sklearn |

### Feature vector

All algorithms use a 35-dimensional feature vector: 11 message features (length, complexity, question type, etc.) × 8 model features (cost, quality, context window, etc.) + interaction terms + 8 context features (agent role type, turn index, etc.). The feature vector describes the current task; conversation history is passed through to the model unchanged.

---

## 11. Semantic caching

Route similar past queries to cached responses instead of calling a model again. Requires `pip install "routesmith[cache]"`.

```python
from routesmith.config import RouteSmithConfig, CacheConfig

config = RouteSmithConfig(
    cache=CacheConfig(
        enabled=True,
        similarity_threshold=0.95,  # cosine similarity required for a cache hit
        ttl_seconds=3600,           # cache entries expire after 1 hour
        max_entries=10000,
        embedding_model="all-MiniLM-L6-v2",
    )
)
rs = RouteSmith(config=config)
```

Or in YAML:

```yaml
cache:
  enabled: true
  similarity_threshold: 0.95
  ttl_seconds: 3600
```

Cache lookups happen before routing — a hit skips the model call entirely.

---

## 12. Config file

Generate a config interactively:

```bash
routesmith init                      # guided setup
routesmith init --output custom.yaml # custom path
routesmith init --force              # overwrite existing
```

Full config reference:

```yaml
# routesmith.yaml

routing:
  strategy: direct          # direct | cascade | parallel | speculative
  predictor: lints          # lints | linucb | adaptive
  lints_v_sq: 1.0
  # linucb_alpha: 1.5
  # linucb_cost_lambda: 0.15

budget:
  max_cost_per_request: 0.10
  max_cost_per_day: 50.0
  quality_threshold: 0.75

cache:
  enabled: true
  similarity_threshold: 0.95
  ttl_seconds: 3600

feedback:
  storage_path: ./routesmith.db     # persist feedback + predictor state to disk
  reward: "quality - 0.3 * cost_normalized"

# Explicit model list with pricing
models:
  - id: openai/gpt-4o-mini
    cost_per_1k_input: 0.15
    cost_per_1k_output: 0.60
    quality_score: 0.85

  - id: deepseek/deepseek-chat
    cost_per_1k_input: 0.014
    cost_per_1k_output: 0.028
    quality_score: 0.72

# Or fetch pricing from OpenRouter automatically
openrouter_models:
  - openai/gpt-4o-mini
  - deepseek/deepseek-chat
  - anthropic/claude-3-haiku
```

---

## 13. Framework integrations

### Anthropic SDK

`RouteSmithAnthropic` is a drop-in replacement for `anthropic.Anthropic`. The `messages.create()` interface is identical.

```python
from routesmith.integrations.anthropic import RouteSmithAnthropic

# Manual model registration
client = RouteSmithAnthropic()
client.register_model("anthropic/claude-3-haiku",    cost_per_1k_input=0.25, cost_per_1k_output=1.25,  quality_score=0.78)
client.register_model("anthropic/claude-3-5-sonnet", cost_per_1k_input=3.0,  cost_per_1k_output=15.0,  quality_score=0.92)

msg = client.messages.create(
    model="auto",     # RouteSmith picks
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain transformers"}],
)
print(msg.content[0].text)

# Or fetch live pricing from OpenRouter
client = RouteSmithAnthropic.with_openrouter_models()
```

### LangChain

`ChatRouteSmith` implements `BaseChatModel` — it works anywhere LangChain expects a chat model.

```python
from routesmith.integrations.langchain import ChatRouteSmith

llm = ChatRouteSmith.with_openai_models()

# Invoke
response = llm.invoke("What is 2+2?")

# Streaming
for chunk in llm.stream("Write a haiku"):
    print(chunk.content, end="", flush=True)

# Async
response = await llm.ainvoke("What is 2+2?")

# With tools
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in Paris?")
```

Per-call routing constraints:

```python
llm = ChatRouteSmith(
    strategy="cascade",
    min_quality=0.85,
    max_cost=0.02,
)
```

### DSPy

```python
import dspy
from routesmith.integrations.dspy import RouteSmithLM

# Native mode — no proxy needed
lm = RouteSmithLM()
lm.register_model("openai/gpt-4o-mini",    cost_per_1k_input=0.15,  cost_per_1k_output=0.60,  quality_score=0.85)
lm.register_model("deepseek/deepseek-chat", cost_per_1k_input=0.014, cost_per_1k_output=0.028, quality_score=0.72)
dspy.configure(lm=lm)

predictor = dspy.Predict("question -> answer")
result = predictor(question="What causes rain?")
print(result.answer)

# Proxy mode (requires routesmith serve)
from routesmith.integrations.dspy import routesmith_lm
dspy.configure(lm=routesmith_lm())
```

### CrewAI

```python
from crewai import Agent, Task, Crew
from routesmith.integrations.crewai import routesmith_crewai_chat_model

llm = routesmith_crewai_chat_model()
llm.routesmith.register_model("openai/gpt-4o-mini", cost_per_1k_input=0.15, cost_per_1k_output=0.60,  quality_score=0.85)
llm.routesmith.register_model("openai/gpt-4o",      cost_per_1k_input=5.0,  cost_per_1k_output=15.0, quality_score=0.95)

agent = Agent(
    role="Analyst",
    goal="Analyze the provided data and extract key insights",
    backstory="You are a seasoned data analyst.",
    llm=llm,
)
task = Task(description="Summarize the following dataset: ...", agent=agent, expected_output="A bullet-point summary")
crew = Crew(agents=[agent], tasks=[task])
crew.kickoff()
```

Proxy mode (for multiple processes):

```python
from routesmith.integrations.crewai import routesmith_crewai_llm
# requires: routesmith serve
agent = Agent(role="Analyst", ..., llm=routesmith_crewai_llm())
```

### AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent
from routesmith.integrations.autogen import routesmith_autogen_llm_config

# requires: routesmith serve
llm_config = routesmith_autogen_llm_config()

assistant = AssistantAgent("assistant", llm_config=llm_config)
user      = UserProxyAgent("user", human_input_mode="NEVER")
user.initiate_chat(assistant, message="Write a binary search in Python.")

# One-line helper
from routesmith.integrations.autogen import routesmith_autogen_agents
assistant, user = routesmith_autogen_agents()
user.initiate_chat(assistant, message="Write a binary search in Python.")
```

---

## 14. Multi-agent systems

### Shared client, per-role routing

The core pattern: one `RouteSmith` instance, one `ChatRouteSmith` (or similar) per agent role. Each role builds its own routing history independently.

```python
from routesmith import RouteSmith
from routesmith.integrations.langchain import ChatRouteSmith
from langgraph.prebuilt import create_react_agent

rs = RouteSmith.with_openai_models()

planner_llm    = ChatRouteSmith(routesmith=rs, agent_role="planner")
researcher_llm = ChatRouteSmith(routesmith=rs, agent_role="researcher")
coder_llm      = ChatRouteSmith(routesmith=rs, agent_role="coder")
critic_llm     = ChatRouteSmith(routesmith=rs, agent_role="critic")

planner    = create_react_agent(planner_llm,    tools=[decompose_tool])
researcher = create_react_agent(researcher_llm, tools=[search_tool])
coder      = create_react_agent(coder_llm,      tools=[bash_tool])
critic     = create_react_agent(critic_llm,     tools=[])
```

After enough feedback, RouteSmith may learn that `coder` calls reliably succeed with `gpt-4o-mini` and stop routing them to `gpt-4o`. Each role builds its posteriors independently without interfering with the others.

### RouteContext (Python API)

When using `rs.completion()` directly:

```python
from routesmith.config import RouteContext

response = rs.completion(
    messages=[{"role": "user", "content": "Write a merge sort."}],
    context=RouteContext(
        agent_id="coder-1",
        agent_role="coder",
        conversation_id="task-42",
    ),
    include_metadata=True,
)
request_id = response.routesmith_metadata["request_id"]
```

### Role inference

If you pass `RouteContext()` without `agent_role`, RouteSmith infers the role from system prompt keywords in under 1ms (MD5-cached). It recognizes common roles: `planner`, `coder`, `researcher`, `critic`, `summarizer`, and others.

```python
response = rs.completion(
    messages=[
        {"role": "system", "content": "You write and debug Python code."},
        {"role": "user",   "content": "Fix the off-by-one error in this function."},
    ],
    context=RouteContext(),  # role_inferred=True will appear in metadata
)
```

Set `agent_role` explicitly when precision matters — inference is keyword-based and can misclassify novel or ambiguous system prompts.

### Conversation tracking

For multi-turn agents, `track_conversation=True` builds `RouteContext` automatically each turn, including `turn_index` and a correction count.

```python
from routesmith.integrations.langchain import ChatRouteSmith

coder_llm = ChatRouteSmith(
    agent_role="coder",
    track_conversation=True,
    conversation_id="task-42",  # optional; auto-generated if omitted
)

for user_message in conversation_turns:
    response = coder_llm.invoke(user_message)
```

The tracker is tied to the `ChatRouteSmith` instance. For parallel conversations, create one instance per conversation.

### Per-role reward functions

Different agents have different cost-quality tradeoffs:

```python
from routesmith.config import RouteSmithConfig

config = RouteSmithConfig(
    reward_fns={
        "planner":    lambda quality, cost_normalized, **_: quality - 0.1 * cost_normalized,
        "coder":      lambda quality, cost_normalized, **_: quality - 0.2 * cost_normalized,
        "researcher": lambda quality, cost_normalized, latency_ms, **_:
                          quality - 0.3 * cost_normalized - 0.1 * (latency_ms / 5000),
        "critic":     lambda quality, cost_normalized, **_: quality - 0.6 * cost_normalized,
    }
)
rs = RouteSmith(config=config)
```

Override a role's reward at runtime:

```python
rs.register_reward_fn(
    "critic",
    lambda quality, cost_normalized, **_: quality - 0.7 * cost_normalized,
)
```

Resolution order: `reward_fns[agent_role]` → global `reward_fn` / `reward_expr` → predictor default.

### Business rules (hard pre-filters)

Business rules run before the predictor. Use them for compliance, cost guardrails, or capacity management.

```python
from routesmith.config import RouteSmithConfig, RouteContext

def block_external_for_pii(models, context: RouteContext | None):
    """Only allow on-prem models when PII is present."""
    if context and context.metadata.get("contains_pii"):
        return [m for m in models if "bedrock" in m.model_id]
    return models

def cheap_models_for_low_priority(models, context: RouteContext | None):
    """Reserve expensive models for high-priority requests."""
    if context and context.metadata.get("priority") != "high":
        return [m for m in models if m.cost_per_1k_output < 1.0]
    return models

config = RouteSmithConfig(
    business_rules=[block_external_for_pii, cheap_models_for_low_priority]
)
rs = RouteSmith(config=config)

response = rs.completion(
    messages=[...],
    context=RouteContext(
        agent_role="summarizer",
        metadata={"contains_pii": True, "priority": "normal"},
    ),
)
```

Rules are applied in order. Each receives the current candidate list and the `RouteContext` and returns a filtered list. An empty result raises `NoEligibleModelsError`.

> **Design note:** Business rules run before the predictor intentionally. When a rule consistently excludes a model, the predictor's learned distribution reflects the true constrained action space rather than drifting from override contamination.

### AutoGen with per-agent identity via headers

Pass `X-RouteSmith-*` headers to get per-role routing at the proxy layer:

```python
import httpx

response = httpx.post(
    "http://localhost:9119/v1/chat/completions",
    headers={
        "X-RouteSmith-Agent-Role":      "coder",
        "X-RouteSmith-Agent-Id":        "coder-1",
        "X-RouteSmith-Conversation-Id": "session-abc",
    },
    json={"model": "auto", "messages": [...]},
)
```

Supported headers: `X-RouteSmith-Agent-Id`, `X-RouteSmith-Agent-Role`, `X-RouteSmith-Conversation-Id`.

### Inspecting what RouteSmith learned per role

```python
report = rs.recommend_model_for_agent("coder", min_samples=50)
# Returns None until 50+ quality records exist for this role.
# {
#   "recommended_model": "openai/gpt-4o",
#   "efficiency_score": 0.87,
#   "avg_quality": 0.91,
#   "avg_cost_usd": 0.0042,
#   "sample_count": 143,
#   "new_models_to_explore": ["openai/gpt-4o-mini"],
# }
```

`new_models_to_explore` lists registered models with fewer than `min_samples` records for this role — useful for deciding whether to run an explicit exploration phase.

---

## 15. Capability-aware routing

RouteSmith auto-detects required capabilities from the request and filters ineligible models before routing.

### Auto-detection

**Tool calling** — detected when `tools=` or `functions=` is in kwargs:

```python
response = rs.completion(
    messages=[...],
    tools=[{"type": "function", "function": {"name": "search", ...}}],
)
# Only models with tool_calling capability are considered
```

**Vision** — detected when messages contain `image_url` content:

```python
response = rs.completion(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ],
    }],
)
# Only models with vision capability are considered
```

### Model capabilities

Capabilities are auto-populated from boolean flags when you register a model:

```python
rs.register_model(
    "openai/gpt-4o",
    cost_per_1k_input=5.0,
    cost_per_1k_output=15.0,
    quality_score=0.95,
    supports_vision=True,
    supports_function_calling=True,
    supports_json_mode=True,
    supports_streaming=True,
)
# capabilities = {"vision", "tool_calling", "json_mode", "streaming"}
```

If a request requires a capability no registered model has, RouteSmith raises `NoEligibleModelsError` with a clear message.

---

## 16. State persistence

By default, predictor state and feedback records live in memory and are lost on restart. To persist to disk:

```python
config = RouteSmithConfig(
    feedback_storage_path="./routesmith.db",  # SQLite
)
rs = RouteSmith(config=config)
```

Or in YAML:

```yaml
feedback:
  storage_path: ./routesmith.db
```

On init, RouteSmith loads the last saved predictor weights. Weights are written every 50 updates and when you call `deregister_model()`. Cold starts after a long gap re-warm from persisted state rather than from prior-only.

---

## 17. CLI reference

```bash
# Generate config interactively
routesmith init [--output FILE] [--force]

# Start OpenAI-compatible proxy
routesmith serve [--config FILE] [--port N]

# Show session stats
routesmith stats [--server URL] [--json]

# Generate OpenClaw provider config
routesmith openclaw-config [--host URL] [-o FILE]
```

---

## 18. When RouteSmith is not the right fit

**Requires explicit feedback.** RouteSmith learns from signals you provide via `record_outcome()`. If your pipeline has no natural success signal — no tests that pass or fail, no user feedback, no downstream validation — routing will rely on the registered `quality_score` priors and won't adapt. It will still route, but it won't improve.

**Needs enough traffic to learn.** The bandit needs feedback to build meaningful posteriors. At very low volume (fewer than ~100 requests per model), routing will behave close to static scoring. LinTS's prior-based exploration will still work, but convergence will be slow.

**Role inference is keyword-based.** `AgentInferencer` classifies agent roles from system prompt keyword density. It handles common roles well but may misclassify novel or ambiguous prompts. When routing precision per role matters, set `agent_role` explicitly rather than relying on inference.

**Streaming proxy doesn't propagate context headers.** The streaming code path in the proxy (`handle_completion_stream`) doesn't read `X-RouteSmith-*` headers yet. Agent context is silently absent for streaming proxy calls. Non-streaming proxy calls are unaffected.

**Not a model quality benchmark.** RouteSmith learns relative routing performance for *your* queries and *your* feedback signal. It won't tell you which model is objectively best — it tells you which model has performed best in your specific workload given the feedback you've provided.
