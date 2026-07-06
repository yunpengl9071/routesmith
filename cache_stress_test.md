# Cache Fragmentation Stress Test — RouteSmith

## 1. The Core Tension (Quantified)

```
Architecture goal:  Route each query to optimal model (routing diversity)
Cache goal:         Return cached responses (benefits from routing stability)
```

These goals are in **direct tension**. Let's quantify under real parameters.

### Scenario Parameters
- N = 5 registered models (gpt-4o, gpt-4o-mini, claude-sonnet, deepseek-chat, gemini-flash)
- M = 1000 unique query types in a session
- Repeat probability p = 0.15 (15% of queries are near-duplicates of prior queries)
- Similarity threshold = 0.95 (only near-verbatim matches)

### Hit Rate Simulation

| Caching Strategy | Formula | With p=0.15, s=0.2 | With p=0.15, s=0.5 |
|---|---|---|---|
| Model-unaware (current if wired) | p | 15.0% | 15.0% |
| Model-aware (filter by model_id) | p × (1-s) | 12.0% | 7.5% |
| Model-aware + quality-tolerant fallback | p × (1-s) + p × s × q | 14.4% (q=0.8) | 13.5% (q=0.8) |

Where s = probability router picks a different model on repeat, q = probability cached model's quality ≥ threshold.

**Key insight**: Model-aware caching costs 2-7.5 percentage points vs. model-unaware at p=0.15. But p=0.15 is optimistic — FAQ chatbots see p>0.5.

## 2. Failure Mode Inventory

### A. Permanent Exploration → Permanent Cache Fragmentation

**The bug**: Both bandit predictors perform ongoing exploration.

```
LinTS.predict():  arm.sample(rng, v_sq) → stochastic sample from posterior
                   Different samples on each call even for identical messages
```

**The LinUCB variant**:
```
LinUCB.predict(): ucb_score = mean_pred + alpha * sqrt(x^T A_inv x)
                    The confidence width shrinks with √n, but NEVER reaches zero
                    alpha=1.5 means the exploration bonus is still meaningful at n=1000
```

**Impact**: Even after 10,000 updates, the router may still switch models for the same query type. Each switch fragments the model-aware cache.

**Mitigation**: LinTS converges toward always sampling the best arm (posterior narrows), but only asymptotically. In practice, 3-5 models with similar quality scores will be sampled indefinitely.

### B. Feedback Loop Data Leakage (Most Dangerous)

```
T=1: Query "sort list" → Router picks gpt-4o-mini → cache["sort list": gpt-4o-mini]
T=2: Query "sort list" → Router picks deepseek-chat →  
     cache HIT → returns gpt-4o-mini response
     user rates response 4/5 → feedback attributed to deepseek-chat
     deepseek-chat's posterior gets positive update for gpt-4o-mini's work!
```

**This is feedback poisoning**. The predictor's quality estimates for deepseek-chat converge to gpt-4o-mini's quality, eliminating the routing benefit of having both models.

**Reinforcement cycle**:
1. Router picks wrong model
2. Cache returns other model's response
3. User rates it (quality = actual_cached_model quality)
4. Predictor updates wrong arm with wrong reward
5. Router's estimates converge to homogeneity → all models rated the same
6. All routing decisions become random → **value prop destroyed**

### C. Feature-Vector-Driven Permanent Oscillation

The 27-dim message feature vector includes:
- Text length, word count, average word length
- Has code, has list, has question
- Language indicators, instruction density

If the same user asks slightly different questions with the same intent:
```
Q1: "How do I sort a list in Python?"       → features: [len=30, has_question=True, ...]
Q2: "Write a Python sort function for me"    → features: [len=35, has_question=False, ...]
```

These generate different feature vectors → different UCB scores → possibly different model selections → even if Q1 and Q2 are semantically similar enough for a cache hit (0.96 similarity), they got routed to different models → cache miss with model-aware.

### D. The Exact Hash Trap

If the same prompt is sent twice and the model-aware fix is applied:

```
T=1: messages=[{"role":"user","content":"What's 2+2?"}] → route to gpt-4o-mini
     cache["sha256(messages) | model=gpt-4o-mini"] = response
     
T=2: messages=[{"role":"user","content":"What's 2+2?"}] → route to gpt-4o
     cache lookup: key = sha256(messages) | model=gpt-4o → MISS
     Executes gpt-4o call → duplicates work
```

But with model-UNaware exact hash:
```
T=1: cache["sha256(messages)"] = gpt-4o-mini response
T=2: cache["sha256(messages)"] = HIT → returns gpt-4o-mini response
     Cost tracked as gpt-4o → incorrect cost attribution
     Quality feedback goes to gpt-4o → feedback poisoning
```

**No win either way for exact duplicates under model switching.**

### E. Streaming Bypasses Cache Entirely

```python
def completion_stream(self, ...)  # line ~510
    # No cache check anywhere
    yield from litellm.completion(model=selected_model, ...)
```

Every streaming call is a 100% cache miss. The design doc says "Cache lookup <1ms" but the streaming path doesn't even attempt it.

### F. Race Condition on Concurrent Access

```python
# semantic.py — no locking
def get(self, messages, semantic=True):
    # Two threads simultaneously check cache → both get None
    return None  # both miss
    
def put(self, messages, response, model_id):
    # Both threads call put → two entries for same query+model
    self._exact_cache[query_hash] = entry  # last write wins
```

In async usage (acompletion), two coroutines for the same prompt would both miss cache, both call the LLM, and waste one call.

### G. Temperature/Ignored Sampling Parameters

The cache keys on messages only. It ignores:
- `temperature`
- `top_p`
- `seed`
- `response_format`
- `tools` / `functions`
- `max_tokens`

Two calls with `temperature=0` and `temperature=1.5` would cache-collide, returning a deterministic response for a creative request (or vice versa).

### H. Memory Bloat Under Model Switching

```
With N models and model-aware caching:
- Each unique query can be cached N times (once per model)
- 10,000 entry limit ÷ 5 models = 2,000 unique queries before eviction
- Without model-aware: 10,000 unique queries before eviction
```

Model-aware caching requires 5x the memory for the same effective query coverage.

## 3. Architectural Verification

### Cache Integration Gap

The `SemanticCache` class exists but is **never instantiated or used** in these code paths:

| Method | Cache check? | Cache store? |
|--------|-------------|-------------|
| `completion()` | ❌ | ❌ |
| `acompletion()` | ❌ | ❌ |
| `completion_stream()` | ❌ | ❌ |
| `acompletion_stream()` | ❌ | ❌ |

The `CacheConfig` has `enabled: bool = False` by default. The `with_cache()` builder exists but the cache instance is never constructed in RouteSmith.__init__() or used in completion().

### Where cache SHOULD be called (proposed flow):

```
completion(messages, ...)
  1. If config.cache.enabled:
     entry = cache.get(messages, model_id=selected_model, 
                       min_quality=min_quality)
     if entry: return entry.response with metadata.cache_hit=True
  2. selected_model = router.route(messages, ...)
  3. response = litellm.completion(model=selected_model, ...)
  4. If config.cache.enabled:
     cache.put(messages, response, model_id=selected_model)
  5. return response
```

**But wait** — the cache check must happen BEFORE routing, but routing determines the model_id needed for model-aware lookup. This is a chicken-and-egg problem for model-aware caching.

### Resolution approaches:

**Option 1: Route first, cache second** (simplest)
```
route → check cache for (messages, model_id) → if hit, return → else execute
```
Pro: Cache always gets the same model_id as the route
Con: Pay routing overhead even on cache hits. But routing is <5ms — cheap.

**Option 2: Cache first, fall through to route** (current design doc intent)
```
check cache (model-unaware) → if hit, return → else route → execute → store
```
Pro: Zero overhead on cache hits
Con: Wrong model's response on cache hit (data leakage, feedback poisoning)

**Option 3: Two-level cache** (recommended)
```
Level 1: Check cache for (messages, model_id=any) with quality ≥ threshold
         → hit: return cached response, skip routing entirely
         → miss: route → check Level 2
Level 2: Check cache for (messages, model_id=routed_model)
         → hit: return
         → miss: execute, store
```
Pro: Recovers cross-model hits when quality is interchangeable, avoids feedback poisoning
Con: Slightly more complex

## 4. Impact Summary

| Severity | Issue | Effect |
|----------|-------|--------|
| 🔴 CRITICAL | Feedback poisoning via model-mismatched cache hits | Predictor converges to homogeneous wrong estimates; routing value destroyed |
| 🔴 CRITICAL | Cache not wired into client | Zero cache benefit; money left on table |
| 🟠 HIGH | Bandit exploration permanently fragments model-aware cache | Cache hit rate drops proportional to exploration rate |
| 🟠 HIGH | Exact duplicates get wrong model's response | Wrong cost attribution, wrong quality feedback |
| 🟡 MEDIUM | Streaming bypasses cache entirely | All streaming calls are full-cost LLM calls |
| 🟡 MEDIUM | No concurrency control | Double LLM calls on concurrent cache misses |
| 🟡 MEDIUM | Sampling params ignored in cache key | Deterministic/creative responses mixed |
| 🟢 LOW | Feature-vector oscillation | Minor hit rate reduction; pathological with many similar-but-not-identical queries |
| 🟢 LOW | Memory bloat with model-aware caching | N× entries for N models; eviction covers it |

## 5. Recommended Resolution

### Phase 1: Wire in the cache (critical path)
- Instantiate `SemanticCache` in `RouteSmith.__init__()` when `config.cache.enabled`
- Call `cache.get()`/`cache.put()` in `completion()` and `acompletion()`

### Phase 2: Model-aware with quality-tolerant fallback
- Add `model_id` parameter to `cache.get()` 
- Exact match: key on `(hash(messages), model_id)`
- Semantic match: filter by `model_id`, then fall back to quality-tolerant cross-model match
- Store `quality_score` from registry in `CacheEntry.metadata`

### Phase 3: Concurrency safety
- Add `threading.Lock` to cache get/put
- Or use `asyncio.Lock` for async path (two separate lock objects)

### Phase 4: Sampling parameter awareness
- Add `temperature`, `top_p`, `tools` to cache key hash
