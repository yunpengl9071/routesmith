# Agentic Routing Design

**Date:** 2026-04-12  
**Status:** Approved  
**Scope:** Agent-aware routing, conversation context, trajectory storage, predictor durability, per-role reward functions, `recommend_model_for_agent()`

---

## Problem

RouteSmith today routes at the per-request level with no concept of agent identity, conversation history, or multi-turn context. This limits its usefulness in two important scenarios:

1. **Multi-agent systems** (LangGraph, CrewAI, AutoGen): The unit of meaningful differentiation is the agent role — a research agent has different quality/cost tradeoffs than a summarizer agent. Per-call routing that ignores role misses this signal.

2. **Conversational systems** (customer service, chatbots): Routing based only on the current message ignores accumulated context — turn count, topic drift, correction loops — that meaningfully changes which model is appropriate.

Additionally, the predictor's learned state (RF weights, LinUCB/LinTS arm parameters) is in-memory only and lost on restart, breaking the "continuous improvement" guarantee.

---

## Design

### Architecture

```
                    ┌─────────────────────────────────────────────┐
Native Python:      │ rs.completion(messages, context=RouteContext)│
                    └──────────────────────┬──────────────────────┘
                                           │
LangChain/DSPy/     ┌──────────────────────▼──────────────────────┐
Anthropic native:   │ ChatRouteSmith / RouteSmithLM /              │
                    │ RouteSmithAnthropic                          │
                    │  agent_role=... track_conversation=True      │
                    │  → builds RouteContext per call              │
                    └──────────────────────┬──────────────────────┘
                                           │
AutoGen/CrewAI/     ┌──────────────────────▼──────────────────────┐
DSPy proxy:         │ Proxy handler reads X-RouteSmith-* headers  │
                    │  → constructs RouteContext → completion()    │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │  Pre-routing filters (capabilities +        │
                    │  business rules) — filter before predictor  │
                    ├─────────────────────────────────────────────┤
                    │  AgentInferencer → fills missing agent_role │
                    │  (system prompt → role + confidence)        │
                    ├─────────────────────────────────────────────┤
                    │  Feature extractor (27 → 35 features)       │
                    ├─────────────────────────────────────────────┤
                    │  Predictor (RF / LinUCB / LinTS)            │
                    │  — loaded from SQLite on init               │
                    │  — serialized to SQLite after retrain       │
                    ├─────────────────────────────────────────────┤
                    │  Storage (SQLite)                           │
                    │  feedback_records (+agent/conv columns)     │
                    │  trajectories (new)                         │
                    │  predictor_state (new)                      │
                    └─────────────────────────────────────────────┘
```

**Key principle:** Business rules and capability filters run *before* the predictor sees the candidate set. The predictor only ever ranks valid models — feedback is always cleanly attributable to a freely chosen model, preventing distribution shift from rule overrides.

---

## Components

### 1. `RouteContext` (new dataclass in `config.py`)

```python
@dataclass
class RouteContext:
    agent_id: str | None = None        # instance ID, e.g. "research_agent_42"
    agent_role: str | None = None      # role type, e.g. "research", "summarizer"
    conversation_id: str | None = None # groups turns; auto-generated if not provided
    turn_index: int | None = None      # 0-based turn number
    metadata: dict[str, Any] = field(default_factory=dict)
```

Passed as `context=RouteContext(...)` to `completion()`. All fields are optional — omitting context entirely falls back to existing 27-feature routing with no degradation.

---

### 2. `ConversationTracker` (new, `feedback/conversation.py`)

Optional helper for callers who want stateful turn tracking. Maintains:
- `turn_count`
- `cumulative_token_estimate` (character count / 4, approximate)
- `correction_count` (detects "no", "wrong", "actually" patterns in user messages)
- `first_user_message` (for topic drift computation)

```python
tracker = ConversationTracker(agent_role="research")

# Each turn:
ctx = tracker.next_context(messages)       # → RouteContext with computed fields
response = rs.completion(messages, context=ctx)
# implicit signals collected automatically

# Optional — only when app has explicit signal:
tracker.record_outcome(ctx.request_id, quality_score=1.0)
```

`record_outcome()` is optional. Implicit signals from `feedback/signals.py` fire automatically on every `completion()` call regardless.

---

### 3. `AgentInferencer` (new, `predictor/agent_inferencer.py`)

Lightweight rule-based classifier. Runs on system prompt (first user message as fallback). No model load, <1ms. Results cached by system-prompt hash.

**Output roles:** `"research"`, `"coding"`, `"summarizer"`, `"qa"`, `"customer_service"`, `"planning"`, `"creative"`, `"general"`

**Graceful degradation contract:**
```python
infer_agent_role(messages) → (role: str | None, confidence: float)
# role=None, confidence=0.0  → keyword density below threshold (0.15); don't use
# role="general", confidence=0.6 → explicitly general-purpose prompt
# role="research", confidence=0.85 → strong keyword match
```

When `role=None`:
- `agent_role_type` = 0 (reserved for "unknown")
- `agent_role_confidence` = 0.0
- Predictor learns to weight `agent_role_type` near zero → degrades to 27-feature baseline

"Unknown" and "general" are distinct. Novel system prompts that don't match any rules return `(None, 0.0)` — never forced into a category.

---

### 4. Feature Vector: 27 → 35 (+8 context features)

| Index | Feature | Description |
|-------|---------|-------------|
| 28 | `turn_index_norm` | `turn / 20`, capped at 1.0 |
| 29 | `conv_token_density` | cumulative tokens / turn count |
| 30 | `correction_rate` | corrections / turns (detects rework loops) |
| 31 | `topic_drift` | divergence from first user message |
| 32 | `agent_role_type` | 0–7 ordinal for role category (0 = unknown) |
| 33 | `agent_role_confidence` | 1.0 if explicit, inferred confidence otherwise |
| 34 | `messages_in_context_norm` | `len(messages) / 50`, capped at 1.0 |
| 35 | `has_agent_context` | 1 if any RouteContext provided, else 0 |

When no `RouteContext` is passed, all 8 features default to 0. Fully backwards compatible — existing trained predictors see the same 27 base features unchanged.

---

### 5. Storage Extensions (`feedback/storage.py`)

**Existing `feedback_records`** — four new nullable columns (old records valid with NULLs):
- `agent_id TEXT`
- `agent_role TEXT`
- `conversation_id TEXT`
- `turn_index INTEGER`

**New `trajectories` table** — groups related calls for observability:
```sql
CREATE TABLE trajectories (
    trajectory_id TEXT PRIMARY KEY,
    conversation_id TEXT,
    agent_id TEXT,
    agent_role TEXT,
    request_ids TEXT,      -- JSON array
    start_time REAL,
    end_time REAL,
    turn_count INTEGER,
    total_cost REAL,
    avg_quality REAL
);
```

**New `predictor_state` table** — survives restarts:
```sql
CREATE TABLE predictor_state (
    predictor_type TEXT PRIMARY KEY,
    serialized_state BLOB,  -- joblib for RF; numpy bytes for LinUCB/LinTS
    updated_at REAL
);
```

**Durability behavior:**
- After each retrain: serialize weights to `predictor_state`
- On `RouteSmith.__init__()`: load from `predictor_state` first, fall back to cold start
- LinUCB/LinTS arm state (A matrices, b vectors) also persisted — exploration state survives restarts

**Historical records are immutable.** Feedback records for removed models are never deleted — they encode query-difficulty signal that informs routing for remaining models.

---

### 6. Model Registration and Removal

**`register_model()`** triggers arm initialization in the active predictor:
- LinUCB/LinTS: fresh arm with `A = I`, `b = 0` (optimistic initialization → high exploration priority)
- RF: cold-start prior from static `quality_score` until `min_samples` reached
- Same behavior whether adding to a fresh system or an existing trained one

**`deregister_model(model_id)`:**
- Removes from registry → immediately excluded from routing
- Drops arm from LinUCB/LinTS state; remaining arms unaffected
- RF: removed model excluded from routing candidates; its historical records remain valid training data
- Persists updated predictor state without the removed arm
- Guards against removing the last model: raises `RouteSmithError`

**Re-adding a previously removed model** → fresh optimistic initialization, same as new. No memory of prior arm state.

**`recommend_model_for_agent()`** filters by `registry.list_models()` — removed models never surface in recommendations even if they have historical records.

---

### 7. Per-Role Reward Functions

Resolution order (first match wins):
1. Tracker-level `reward_fn` (passed to `ConversationTracker`)
2. `config.reward_fns[agent_role]` (per-role mapping)
3. Global `config.reward_fn` / `config.reward_expr`
4. Predictor's internal default reward computation

**`RouteSmithConfig` additions:**
```python
reward_fns: dict[str, Callable] = field(default_factory=dict)
business_rules: list[Callable] = field(default_factory=list)
```

**Example:**
```python
RouteSmithConfig(
    reward_expr="latency_score * 0.3 + quality * 0.7",  # global fallback
    reward_fns={
        "research":         lambda r, m: information_density(r),
        "summarizer":       lambda r, m: compression_ratio(r, m),
        "customer_service": lambda r, m: resolution_signal(r),
    }
)
```

**Runtime registration:**
```python
rs.register_reward_fn("coding", lambda r, m: test_pass_rate(r))
```

---

### 8. `recommend_model_for_agent(agent_role)`

Queries `feedback_records` filtered by `agent_role`, ranks by quality-adjusted cost efficiency, returns top model with confidence based on sample count.

```python
rs.recommend_model_for_agent("research")
# → {
#     "model": "openai/gpt-4o",
#     "confidence": 0.87,
#     "sample_count": 312,
#     "avg_quality": 0.91,
#     "avg_cost_usd": 0.004,
#     "new_models_to_explore": ["anthropic/claude-3-7-sonnet"]  # < min_samples
#   }
```

Returns `None` when `agent_role` is `None`, or when fewer than `min_samples` (default 50) records exist for the role. `new_models_to_explore` surfaces registered models with insufficient data so the caller knows the recommendation is not exhaustive.

---

### 9. Business Rules (Pre-Routing Filters)

Business rules register as pre-routing filters alongside capability filtering — they run before the predictor sees the candidate set:

```
business_rules + filter_by_capabilities() → filtered candidates → predictor ranks → route
```

This keeps feedback clean and attributable. If a rule consistently excludes a model, the predictor's learned model reflects the true constrained action space rather than drifting due to override contamination.

```python
# Signature: (models: list[ModelConfig], context: RouteContext | None) -> list[ModelConfig]
def hipaa_filter(models, context):
    return [m for m in models if "hipaa" in m.capabilities]

RouteSmithConfig(business_rules=[hipaa_filter])
```

Forced model choices (e.g. A/B tests) are flagged as `exploration_override=True` in feedback records, consistent with the existing `ab_test.py` pattern.

---

### 10. Integration-Specific Changes

**`RouteSmith.completion()`** — one new optional param:
```python
def completion(self, messages, ..., context: RouteContext | None = None, **kwargs)
```

**`ChatRouteSmith` / `RouteSmithLM` / `RouteSmithAnthropic`** — three new constructor fields:
```python
agent_role: str | None = None       # explicit role for every call from this instance
conversation_id: str | None = None  # optional; auto-generated when track_conversation=True
track_conversation: bool = False    # enables internal ConversationTracker
reward_fn: Callable | None = None   # per-instance reward override
```

**Proxy server** (`proxy/handler.py`) — reads three new optional headers:
```
X-RouteSmith-Agent-Id: research_agent_42
X-RouteSmith-Agent-Role: research
X-RouteSmith-Conversation-Id: conv_abc123
```
Constructs `RouteContext` before routing. AutoGen `llm_config` and CrewAI `LLM()` both support custom headers — zero friction for proxy-mode users.

---

## What Does Not Change

- `RouteSmith.completion()` with no `context` argument behaves identically to today
- The 27 existing features are unchanged; 8 new features default to 0 when no context is passed
- Existing trained predictor state (if any) remains valid — the new feature dimensions start at 0
- `reward_fn` / `reward_expr` global config continues to work as before
- All existing integration factory methods (`with_openai_models()`, `with_anthropic_models()`, etc.) are unchanged

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `src/routesmith/config.py` | Add `RouteContext`, `reward_fns`, `business_rules` to `RouteSmithConfig` |
| `src/routesmith/client.py` | Add `context` param to `completion()`, `deregister_model()`, `register_reward_fn()`, `recommend_model_for_agent()` |
| `src/routesmith/feedback/conversation.py` | New — `ConversationTracker` |
| `src/routesmith/predictor/agent_inferencer.py` | New — `AgentInferencer` |
| `src/routesmith/predictor/features.py` | Add 8 context features (indices 28–35) |
| `src/routesmith/feedback/storage.py` | New columns on `feedback_records`; new `trajectories` and `predictor_state` tables |
| `src/routesmith/integrations/langchain.py` | Add `agent_role`, `track_conversation`, `reward_fn` fields to `ChatRouteSmith` |
| `src/routesmith/integrations/dspy.py` | Same fields on `RouteSmithLM` |
| `src/routesmith/integrations/anthropic.py` | Same fields on `RouteSmithAnthropic` |
| `src/routesmith/proxy/handler.py` | Read `X-RouteSmith-*` headers, construct `RouteContext` |
