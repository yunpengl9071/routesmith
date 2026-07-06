# Phase 3 — Differentiation: features a hosted black-box router won't build

Three moats. Stickiness matches OpenRouter's one genuinely superior feature, then exceeds it.
Per-role policies and the decision audit log target the production/multi-agent buyers whom
reviewers explicitly steer AWAY from `openrouter/auto`.

Task order: any (all independent).

---

## Task P3.1 — Conversation stickiness (cache-aware pinning)  (size: M)

**The gap:** OpenRouter pins model+provider per conversation once prompt caching kicks in.
RouteSmith currently re-routes every turn, which can destroy provider prompt caches
mid-conversation — a real cost REGRESSION vs. not using RouteSmith. Match: pin per
conversation. Exceed: unpin intelligently on drift/corrections, which the
`ConversationTracker` already detects (`feedback/conversation.py` — read it first; it
computes topic drift and correction counts that reach `RouteContext.metadata` as
`topic_drift` / `correction_count`, consumed in `features.py _extract_context_features`).

**Files:** `src/routesmith/strategy/stickiness.py` (new), `src/routesmith/client.py`,
`src/routesmith/config.py`, `tests/test_stickiness.py` (new)

**Spec — config:** add to `RouteSmithConfig`:
```python
sticky_conversations: bool = True   # Pin model per conversation_id (preserves prompt caches)
sticky_ttl_s: float = 1800.0        # Unpin after 30 min idle
sticky_drift_threshold: float = 0.5 # Re-route when topic_drift exceeds this
```

**Spec — `src/routesmith/strategy/stickiness.py`:**

```python
"""Per-conversation model pinning."""
from __future__ import annotations

import time
from collections import OrderedDict


class ConversationPins:
    """LRU map: conversation_id -> (model_id, last_used_ts). Max 10_000 entries."""

    MAX_ENTRIES = 10_000

    def __init__(self, ttl_s: float = 1800.0) -> None:
        self._ttl = ttl_s
        self._pins: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def get(self, conversation_id: str, now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        entry = self._pins.get(conversation_id)
        if entry is None:
            return None
        model_id, ts = entry
        if now - ts > self._ttl:
            del self._pins[conversation_id]
            return None
        self._pins.move_to_end(conversation_id)
        self._pins[conversation_id] = (model_id, now)
        return model_id

    def pin(self, conversation_id: str, model_id: str, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._pins[conversation_id] = (model_id, now)
        self._pins.move_to_end(conversation_id)
        while len(self._pins) > self.MAX_ENTRIES:
            self._pins.popitem(last=False)

    def unpin(self, conversation_id: str) -> None:
        self._pins.pop(conversation_id, None)
```

**Spec — wiring (`client.py`):**
1. `__init__`: `self._pins = ConversationPins(ttl_s=config.sticky_ttl_s)`.
2. In `completion()`/`acompletion()`, immediately BEFORE calling `self.router.route(...)`,
   and only when `model` (the explicit-model argument) is None:

   ```python
   conv_id = context.conversation_id if context else None
   if conv_id and self.config.sticky_conversations:
       drift = float(context.metadata.get("topic_drift", 0.0))
       corrections_now = int(context.metadata.get("correction_count", 0))
       if drift > self.config.sticky_drift_threshold or corrections_now > self._last_corrections.get(conv_id, 0):
           self._pins.unpin(conv_id)          # user corrected or changed topic: re-route
       pinned = self._pins.get(conv_id)
       if pinned and self.registry.get(pinned) is not None:
           selected_model = pinned
           routing_reason = "sticky"          # surface in metadata
   ```
   (`self._last_corrections: dict[str, int] = {}`, updated after each turn; when a pin is
   used, SKIP `router.route` entirely.)
3. After a SUCCESSFUL completion with a `conversation_id`: `self._pins.pin(conv_id, selected_model)`
   and `self._last_corrections[conv_id] = corrections_now`.
4. Routing metadata gains `"routing_reason": "sticky" | "routed" | "explicit" | "fallback"`.
5. The proxy already forwards `X-RouteSmith-Conversation-Id` into `RouteContext`
   (`handler.py _ROUTESMITH_HEADERS`) — no proxy change needed; add a README note that
   sending this header enables stickiness.

**Tests** (`tests/test_stickiness.py`, litellm mocked, explicit `now=` where the pin store is
driven directly):
- `test_second_turn_reuses_pinned_model` — 2 completions, same conversation_id → router.route
  called ONCE (spy), both responses same model, second metadata `routing_reason == "sticky"`.
- `test_no_conversation_id_no_pinning` — route called twice.
- `test_ttl_expiry_reroutes` — drive `ConversationPins` with now=0 / now=1801 → get returns None.
- `test_drift_breaks_pin` — second turn metadata `topic_drift=0.9` → route called again.
- `test_correction_breaks_pin` — `correction_count` increments → route called again.
- `test_pin_ignored_when_model_deregistered` — pin a model, remove it from registry → routed fresh.
- `test_lru_eviction_at_capacity` — pin MAX_ENTRIES+1 conversations → oldest evicted.
- `test_sticky_disabled_by_config` — flag False → route every turn.

**Acceptance:** tests pass; README "Multi-turn conversations" section explains
pin/unpin rules in one table.

---

## Task P3.2 — Per-agent-role routing policies  (size: M)

Multi-agent frameworks are the fastest-growing LLM spend, and NotDiamond is per-prompt only —
no concept of "my planner needs quality, my summarizer needs cheap". The plumbing
(`RouteContext.agent_role`, `X-RouteSmith-Agent-Role`, per-role reward_fns) exists; policies
close the loop.

**Files:** `src/routesmith/config.py`, `src/routesmith/strategy/router.py`,
`src/routesmith/client.py`, `src/routesmith/cli/yaml_loader.py`, `tests/test_role_policies.py` (new)

**Spec — config:**

```python
@dataclass
class RolePolicy:
    """Routing constraints for one agent role."""

    allowed_models: list[str] | None = None   # None = all registered
    denied_models: list[str] = field(default_factory=list)
    min_quality: float | None = None          # overrides budget.quality_threshold
    max_cost_per_request: float | None = None # overrides budget.max_cost_per_request
```
`RouteSmithConfig` gains `role_policies: dict[str, RolePolicy] = field(default_factory=dict)`.

**Spec — enforcement:**
1. `router.py`: in `_route_direct` and `_route_cascade`, after `_apply_business_rules`, add
   `candidates = self._apply_role_policy(candidates, context)`:

   ```python
   def _apply_role_policy(self, candidates, context):
       role = getattr(context, "agent_role", None) if context else None
       policy = self.config.role_policies.get(role) if role else None
       if policy is None:
           return candidates
       if policy.allowed_models is not None:
           candidates = [c for c in candidates if c.model_id in policy.allowed_models]
       if policy.denied_models:
           candidates = [c for c in candidates if c.model_id not in policy.denied_models]
       if not candidates:
           raise ValueError(f"Role policy for '{role}' filtered out all models")
       return candidates
   ```
2. `client.py`: when resolving `min_quality` (currently
   `min_quality or self.config.budget.quality_threshold`), consult the role policy first:
   explicit arg > `policy.min_quality` > `budget.quality_threshold`. Same precedence for
   `max_cost_per_request` in the P0.4 pre-flight filter.
3. `yaml_loader.py`: parse

   ```yaml
   roles:
     planner: {min_quality: 0.9}
     summarizer: {allowed_models: ["openai/gpt-4o-mini"], min_quality: 0.0}
     coder: {denied_models: ["some/weak-model"]}
   ```

**Tests** (`tests/test_role_policies.py`):
- `test_allowed_models_restricts_routing` — policy allows only mini; route with
  `RouteContext(agent_role="summarizer")` → mini selected always (10 runs).
- `test_denied_models_excluded` / `test_empty_result_raises` /
  `test_no_role_no_policy_applied` / `test_role_min_quality_overrides_global`
  (global threshold 0.0, planner policy 0.99 + priors set so only gpt-4o clears →
  planner gets gpt-4o) / `test_yaml_roles_parsed`.

**Acceptance:** tests pass; new `docs/examples/multi-agent.md` shows a 3-role CrewAI pipeline
with per-role policies + per-role reward_fns in one YAML.

---

## Task P3.3 — Routing decision audit log  (size: M)

Transparency is the anti-black-box weapon: every decision reconstructable — candidates,
scores, filters, winner, reason.

**Files:** `src/routesmith/feedback/storage.py`, `src/routesmith/client.py`,
`src/routesmith/proxy/handler.py`, `src/routesmith/cli/stats.py`,
`tests/test_decision_log.py` (new)

**Spec:**
1. `config.py`: `audit_routing: bool = True`.
2. `storage.py`: new table (create in the same place existing tables are created; bump any
   schema-version constant found there):

   ```sql
   CREATE TABLE IF NOT EXISTS routing_decisions (
       request_id TEXT PRIMARY KEY,
       created_at REAL NOT NULL,
       strategy TEXT NOT NULL,
       agent_role TEXT,
       conversation_id TEXT,
       selected_model TEXT NOT NULL,
       routing_reason TEXT NOT NULL,        -- routed | sticky | explicit | fallback | cascade
       candidates_json TEXT NOT NULL        -- [{"model": .., "predicted_quality": .., "cost_per_1k": ..}]
   );
   CREATE INDEX IF NOT EXISTS idx_decisions_created ON routing_decisions(created_at);
   ```
   Methods: `store_decision(request_id, decision: dict) -> None` and
   `recent_decisions(limit: int = 20) -> list[dict]` (parse candidates_json back to a list).
3. Producing candidate scores: `router.route()` returns only a model id. Add an OPTIONAL
   richer API — `Router.route_with_trace(...) -> tuple[str, list[dict]]` that wraps the
   existing prediction flow and also returns per-candidate
   `{"model", "predicted_quality", "cost_per_1k"}`. Implement by refactoring `_route_direct`
   to build the trace internally; `route()` keeps its exact current signature and behavior
   by delegating to `route_with_trace()[0]`.
4. `client.py`: when `audit_routing` and storage exists, call `store_decision(...)` after
   routing (fire-and-forget; wrap in try/except-log — auditing must never fail a request).
5. Surfacing: `/v1/stats` response gains `"recent_decisions": storage.recent_decisions(20)`;
   CLI `routesmith stats --decisions` pretty-prints them (model, reason, top-2 candidates).

**Tests** (`tests/test_decision_log.py`):
- `test_decision_stored_per_request` — one completion → one row; candidates list contains
  BOTH registered models with numeric `predicted_quality`.
- `test_route_with_trace_matches_route` — same seed → same selection from both APIs.
- `test_sticky_decision_reason_recorded` — pinned turn → `routing_reason == "sticky"`.
- `test_audit_disabled_no_rows` / `test_audit_failure_does_not_break_completion`
  (monkeypatch store_decision to raise → completion still succeeds).
- `test_recent_decisions_order_and_limit` — 25 requests → 20 returned, newest first.

**Acceptance:** tests pass; README "Why did it pick that model?" section shows a real
`/v1/stats` decision excerpt.
