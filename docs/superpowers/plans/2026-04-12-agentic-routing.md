# Agentic Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend RouteSmith with agent-aware routing, conversation context tracking, per-role reward functions, trajectory storage, predictor state durability, and a `recommend_model_for_agent()` API.

**Architecture:** `RouteContext` carries agent identity through `completion()`; `AgentInferencer` fills missing roles from system prompts; 8 new context features extend the 27-dim feature vector; predictor state serializes to SQLite (JSON for bandit arms, joblib bytes for RF) after each retrain and reloads on init. Business rules run as pre-routing filters before the predictor sees candidates.

**Tech Stack:** Python 3.13, SQLite (via existing `FeedbackStorage`), numpy, json (for bandit state serialization), joblib (for RF model), existing routesmith predictor/registry/router stack.

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `src/routesmith/config.py` | Modify | Add `RouteContext`; add `reward_fns`, `business_rules` to `RouteSmithConfig` |
| `src/routesmith/feedback/storage.py` | Modify | New columns on `feedback_records`; new `trajectories` + `predictor_state` tables; `save_predictor_state()`/`load_predictor_state()` |
| `src/routesmith/predictor/agent_inferencer.py` | Create | Keyword classifier: system prompt → `(role, confidence)` |
| `src/routesmith/predictor/features.py` | Modify | `extract()` accepts optional `RouteContext`; adds 8 context features (27→35 dims) |
| `src/routesmith/predictor/lints.py` | Modify | `LinTSPredictor.add_arm()`/`remove_arm()`; JSON state serialization; 35-dim |
| `src/routesmith/predictor/linucb.py` | Modify | `LinUCBPredictor.add_arm()`/`remove_arm()`; JSON state serialization |
| `src/routesmith/predictor/learner.py` | Modify | `AdaptivePredictor.add_arm()`/`remove_arm()`; joblib RF serialization |
| `src/routesmith/feedback/conversation.py` | Create | `ConversationTracker`: stateful turn tracker → emits `RouteContext` |
| `src/routesmith/feedback/collector.py` | Modify | Per-role reward resolution; thread `RouteContext` into stored records |
| `src/routesmith/strategy/router.py` | Modify | `route()` accepts `context`; apply `business_rules` before capability filter |
| `src/routesmith/client.py` | Modify | `context` param on `completion()`; `deregister_model()`; `register_reward_fn()`; `recommend_model_for_agent()`; predictor state load on init |
| `src/routesmith/integrations/langchain.py` | Modify | Add `agent_role`, `track_conversation`, `conversation_id`, `reward_fn` to `ChatRouteSmith` |
| `src/routesmith/integrations/dspy.py` | Modify | Same four fields on `RouteSmithLM` |
| `src/routesmith/integrations/anthropic.py` | Modify | Same four fields on `RouteSmithAnthropic` |
| `src/routesmith/proxy/handler.py` | Modify | Read `X-RouteSmith-*` headers; construct `RouteContext` |

---

## Task 1: `RouteContext` + `RouteSmithConfig` additions

**Files:**
- Modify: `src/routesmith/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_config.py
from routesmith.config import RouteContext, RouteSmithConfig


class TestRouteContext:
    def test_all_optional(self):
        ctx = RouteContext()
        assert ctx.agent_id is None
        assert ctx.agent_role is None
        assert ctx.conversation_id is None
        assert ctx.turn_index is None
        assert ctx.metadata == {}

    def test_partial_construction(self):
        ctx = RouteContext(agent_role="research", turn_index=3)
        assert ctx.agent_role == "research"
        assert ctx.turn_index == 3
        assert ctx.agent_id is None


class TestRouteSmithConfigExtensions:
    def test_reward_fns_default_empty(self):
        config = RouteSmithConfig()
        assert config.reward_fns == {}

    def test_business_rules_default_empty(self):
        config = RouteSmithConfig()
        assert config.business_rules == []

    def test_reward_fns_set(self):
        fn = lambda r, m: 0.9
        config = RouteSmithConfig(reward_fns={"research": fn})
        assert config.reward_fns["research"] is fn
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
.venv/bin/pytest tests/test_config.py::TestRouteContext tests/test_config.py::TestRouteSmithConfigExtensions -v
```
Expected: `ImportError: cannot import name 'RouteContext'`

- [ ] **Step 3: Implement**

In `src/routesmith/config.py`, add `RouteContext` before `RouteSmithConfig`:

```python
@dataclass
class RouteContext:
    """Agent and conversation context for a completion request."""

    agent_id: str | None = None
    agent_role: str | None = None
    conversation_id: str | None = None
    turn_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Add to `RouteSmithConfig` dataclass body (after `reward_expr`):

```python
    # Per-agent-role reward functions. Resolution order:
    # reward_fns[agent_role] → reward_fn/reward_expr → predictor default.
    reward_fns: dict[str, Callable[..., float]] = field(default_factory=dict)

    # Pre-routing filter callables.
    # Signature: (models: list[ModelConfig], context: RouteContext | None) -> list[ModelConfig]
    # Run before capability filtering; predictor only sees the filtered set.
    business_rules: list[Callable[..., list]] = field(default_factory=list)
```

Add `RouteContext` to `__all__` in `src/routesmith/__init__.py` if that file has an explicit `__all__`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv/bin/pytest tests/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/config.py src/routesmith/__init__.py tests/test_config.py
git commit -m "feat(config): add RouteContext dataclass and reward_fns/business_rules to RouteSmithConfig"
```

---

## Task 2: Storage schema extensions

**Files:**
- Modify: `src/routesmith/feedback/storage.py`
- Test: `tests/test_feedback.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_feedback.py
from routesmith.feedback.storage import FeedbackStorage


class TestStorageSchemaExtensions:
    def test_feedback_records_has_agent_columns(self):
        storage = FeedbackStorage(":memory:")
        conn = storage._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback_records)")}
        assert "agent_id" in cols
        assert "agent_role" in cols
        assert "conversation_id" in cols
        assert "turn_index" in cols

    def test_trajectories_table_exists(self):
        storage = FeedbackStorage(":memory:")
        conn = storage._get_conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "trajectories" in tables

    def test_predictor_state_table_exists(self):
        storage = FeedbackStorage(":memory:")
        conn = storage._get_conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "predictor_state" in tables

    def test_store_record_with_agent_context(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record(
            request_id="req1",
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            latency_ms=100.0,
            agent_id="agent_42",
            agent_role="research",
            conversation_id="conv_abc",
            turn_index=2,
        )
        record = storage.get_record("req1")
        assert record["agent_role"] == "research"
        assert record["turn_index"] == 2

    def test_store_record_without_agent_context_still_works(self):
        storage = FeedbackStorage(":memory:")
        storage.store_record(
            request_id="req2",
            model_id="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            latency_ms=50.0,
        )
        record = storage.get_record("req2")
        assert record["agent_id"] is None
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
.venv/bin/pytest tests/test_feedback.py::TestStorageSchemaExtensions -v
```

- [ ] **Step 3: Implement schema changes**

In `FeedbackStorage._create_tables()`, replace the `executescript` content with:

```python
self._conn.executescript("""
    CREATE TABLE IF NOT EXISTS feedback_records (
        request_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        messages_json TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        quality_score REAL,
        user_feedback TEXT,
        metadata_json TEXT,
        created_at REAL NOT NULL,
        agent_id TEXT,
        agent_role TEXT,
        conversation_id TEXT,
        turn_index INTEGER
    );

    CREATE TABLE IF NOT EXISTS outcome_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        signal_name TEXT NOT NULL,
        signal_value REAL NOT NULL,
        raw_value_json TEXT,
        created_at REAL NOT NULL,
        FOREIGN KEY (request_id) REFERENCES feedback_records(request_id)
    );

    CREATE TABLE IF NOT EXISTS trajectories (
        trajectory_id TEXT PRIMARY KEY,
        conversation_id TEXT,
        agent_id TEXT,
        agent_role TEXT,
        request_ids TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL,
        turn_count INTEGER NOT NULL DEFAULT 0,
        total_cost REAL NOT NULL DEFAULT 0.0,
        avg_quality REAL
    );

    CREATE TABLE IF NOT EXISTS predictor_state (
        predictor_type TEXT PRIMARY KEY,
        serialized_state BLOB NOT NULL,
        updated_at REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_signals_request
        ON outcome_signals(request_id);
    CREATE INDEX IF NOT EXISTS idx_records_model
        ON feedback_records(model_id);
    CREATE INDEX IF NOT EXISTS idx_records_agent_role
        ON feedback_records(agent_role);
    CREATE INDEX IF NOT EXISTS idx_records_conversation
        ON feedback_records(conversation_id);
""")
# Idempotent migration for existing databases that lack the new columns
for col, coltype in [
    ("agent_id", "TEXT"),
    ("agent_role", "TEXT"),
    ("conversation_id", "TEXT"),
    ("turn_index", "INTEGER"),
]:
    try:
        self._conn.execute(
            f"ALTER TABLE feedback_records ADD COLUMN {col} {coltype}"
        )
        self._conn.commit()
    except Exception:
        pass  # Column already exists
```

Update `store_record()` signature and INSERT:

```python
def store_record(
    self,
    request_id: str,
    model_id: str,
    messages: list[dict[str, str]],
    latency_ms: float,
    quality_score: float | None = None,
    user_feedback: str | None = None,
    metadata: dict[str, Any] | None = None,
    agent_id: str | None = None,
    agent_role: str | None = None,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> None:
    conn = self._get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO feedback_records
           (request_id, model_id, messages_json, latency_ms,
            quality_score, user_feedback, metadata_json, created_at,
            agent_id, agent_role, conversation_id, turn_index)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            request_id, model_id, json.dumps(messages), latency_ms,
            quality_score, user_feedback,
            json.dumps(metadata) if metadata else None,
            time.time(),
            agent_id, agent_role, conversation_id, turn_index,
        ),
    )
    conn.commit()
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_feedback.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/feedback/storage.py tests/test_feedback.py
git commit -m "feat(storage): add agent/conversation columns, trajectories and predictor_state tables"
```

---

## Task 3: Predictor state persistence in storage

**Files:**
- Modify: `src/routesmith/feedback/storage.py`
- Test: `tests/test_feedback.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_feedback.py
import json


class TestPredictorStatePersistence:
    def test_save_and_load_predictor_state(self):
        storage = FeedbackStorage(":memory:")
        # State stored as JSON bytes (no pickle)
        state = {"arms": [], "d": 35, "t": 0}
        blob = json.dumps(state).encode()
        storage.save_predictor_state("lints", blob)
        loaded = storage.load_predictor_state("lints")
        assert loaded is not None
        recovered = json.loads(loaded.decode())
        assert recovered["d"] == 35

    def test_load_missing_predictor_returns_none(self):
        storage = FeedbackStorage(":memory:")
        assert storage.load_predictor_state("nonexistent") is None

    def test_save_overwrites_existing(self):
        storage = FeedbackStorage(":memory:")
        storage.save_predictor_state("lints", b"v1")
        storage.save_predictor_state("lints", b"v2")
        assert storage.load_predictor_state("lints") == b"v2"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_feedback.py::TestPredictorStatePersistence -v
```

- [ ] **Step 3: Implement**

Add to `FeedbackStorage`:

```python
def save_predictor_state(self, predictor_type: str, serialized_state: bytes) -> None:
    """Persist serialized predictor state to SQLite."""
    conn = self._get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO predictor_state
           (predictor_type, serialized_state, updated_at)
           VALUES (?, ?, ?)""",
        (predictor_type, serialized_state, time.time()),
    )
    conn.commit()

def load_predictor_state(self, predictor_type: str) -> bytes | None:
    """Load serialized predictor state. Returns None if not found."""
    conn = self._get_conn()
    row = conn.execute(
        "SELECT serialized_state FROM predictor_state WHERE predictor_type = ?",
        (predictor_type,),
    ).fetchone()
    return row[0] if row else None
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_feedback.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/feedback/storage.py tests/test_feedback.py
git commit -m "feat(storage): add save_predictor_state/load_predictor_state"
```

---

## Task 4: `AgentInferencer`

**Files:**
- Create: `src/routesmith/predictor/agent_inferencer.py`
- Create: `tests/test_agent_inferencer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_inferencer.py
from routesmith.predictor.agent_inferencer import AgentInferencer, AGENT_ROLES


class TestAgentInferencer:
    def setup_method(self):
        self.inferencer = AgentInferencer()

    def test_research_prompt(self):
        messages = [
            {"role": "system", "content": "You are a research assistant. Analyze papers and summarize findings."},
            {"role": "user", "content": "What is the latest on transformer architectures?"},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "research"
        assert confidence > 0.3

    def test_coding_prompt(self):
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Help users debug and implement algorithms."},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "coding"
        assert confidence > 0.3

    def test_customer_service_prompt(self):
        messages = [
            {"role": "system", "content": "You are a customer support agent for Acme Corp. Resolve customer issues politely."},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "customer_service"
        assert confidence > 0.3

    def test_unknown_prompt_returns_none(self):
        messages = [
            {"role": "system", "content": "xyz123 qwerty foo bar baz"},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role is None
        assert confidence == 0.0

    def test_empty_messages_returns_none(self):
        role, confidence = self.inferencer.infer([])
        assert role is None
        assert confidence == 0.0

    def test_general_assistant_prompt(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        role, confidence = self.inferencer.infer(messages)
        assert role == "general"
        assert confidence > 0.0

    def test_caches_by_system_prompt_hash(self):
        messages = [
            {"role": "system", "content": "You help with code and debugging Python programs."},
        ]
        result1 = self.inferencer.infer(messages)
        result2 = self.inferencer.infer(messages)
        assert result1 == result2
        assert len(self.inferencer._cache) == 1

    def test_role_ordinal_known_roles(self):
        for role in AGENT_ROLES:
            assert AgentInferencer.role_ordinal(role) >= 1

    def test_role_ordinal_none_returns_zero(self):
        assert AgentInferencer.role_ordinal(None) == 0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_agent_inferencer.py -v
```

- [ ] **Step 3: Implement `src/routesmith/predictor/agent_inferencer.py`**

```python
"""Lightweight agent role inference from system prompts."""

from __future__ import annotations

import hashlib

# Ordered list of roles. Index 0 is reserved for unknown (role=None).
# Indices 1-8 correspond to these roles (used by role_ordinal()).
AGENT_ROLES: list[str] = [
    "research",
    "coding",
    "summarizer",
    "qa",
    "customer_service",
    "planning",
    "creative",
    "general",
]

_ROLE_KEYWORDS: dict[str, frozenset[str]] = {
    "research": frozenset([
        "research", "analyze", "analysis", "findings", "papers", "literature",
        "study", "investigate", "evidence", "scholar", "academic", "survey",
        "hypothesis", "data", "experiment", "citation",
    ]),
    "coding": frozenset([
        "code", "coding", "programming", "python", "javascript", "typescript",
        "debug", "implement", "algorithm", "function", "class", "developer",
        "software", "bug", "compile", "syntax", "script", "repository",
    ]),
    "summarizer": frozenset([
        "summarize", "summary", "condense", "shorten", "brief",
        "key", "highlight", "extract", "distill", "compress",
    ]),
    "qa": frozenset([
        "answer", "question", "faq", "quiz", "knowledge", "factual",
        "accurate", "correct", "truth", "fact",
    ]),
    "customer_service": frozenset([
        "customer", "support", "service", "issue", "complaint",
        "resolve", "ticket", "refund", "account", "order", "product",
        "satisfaction", "escalate", "politely", "assist",
    ]),
    "planning": frozenset([
        "plan", "planning", "schedule", "roadmap", "task", "project",
        "milestone", "timeline", "organize", "strategy", "objective",
        "goal", "workflow", "prioritize",
    ]),
    "creative": frozenset([
        "creative", "write", "story", "poem", "fiction", "imagine",
        "character", "narrative", "dialogue", "compose", "draft",
        "essay", "blog", "lyric", "brainstorm",
    ]),
}

_CONFIDENCE_THRESHOLD = 0.15


class AgentInferencer:
    """Infer agent role from system prompt using keyword density.

    Returns (role, confidence) where role is None and confidence is 0.0
    when keyword density falls below threshold — never forces a category.
    Results are cached by system-prompt hash; inference is <1ms.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str | None, float]] = {}

    def infer(self, messages: list[dict]) -> tuple[str | None, float]:
        """Infer agent role from messages.

        Uses system prompt if present; falls back to first user message.
        """
        if not messages:
            return None, 0.0

        text = self._extract_text(messages)
        if not text:
            return None, 0.0

        cache_key = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._classify(text)
        self._cache[cache_key] = result
        return result

    def _extract_text(self, messages: list[dict]) -> str:
        for msg in messages:
            if msg.get("role") == "system":
                return str(msg.get("content", "")).lower()
        for msg in messages:
            if msg.get("role") == "user":
                return str(msg.get("content", "")).lower()
        return ""

    def _classify(self, text: str) -> tuple[str | None, float]:
        # Special case: "helpful assistant" or "general purpose" phrases
        if "helpful assistant" in text or "general purpose" in text:
            return "general", 0.7

        words = set(text.split())
        word_count = max(len(words), 1)
        best_role: str | None = None
        best_score = 0.0

        for role, keywords in _ROLE_KEYWORDS.items():
            matches = len(words & keywords)
            score = min(1.0, matches / word_count * 8)
            if score > best_score:
                best_score = score
                best_role = role

        if best_score < _CONFIDENCE_THRESHOLD:
            return None, 0.0

        return best_role, best_score

    @staticmethod
    def role_ordinal(role: str | None) -> int:
        """Return 1-based ordinal for role, or 0 for unknown/None."""
        if role is None:
            return 0
        try:
            return AGENT_ROLES.index(role) + 1
        except ValueError:
            return 0
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_agent_inferencer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/predictor/agent_inferencer.py tests/test_agent_inferencer.py
git commit -m "feat(predictor): add AgentInferencer — keyword-based agent role classifier"
```

---

## Task 5: Feature vector extensions (27 → 35 dims)

**Files:**
- Modify: `src/routesmith/predictor/features.py`
- Modify: `src/routesmith/predictor/lints.py` (update hardcoded `d=27` → `d=35`)
- Test: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_features.py
import pytest
from routesmith.config import RouteContext
from routesmith.predictor.features import FeatureExtractor
from routesmith.registry.models import ModelRegistry


def _make_registry():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    return reg


class TestContextFeatures:
    def test_no_context_gives_35_features_zeros_for_new_dims(self):
        extractor = FeatureExtractor(_make_registry())
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=None)
        assert len(fv.features) == 35
        assert all(f == 0.0 for f in fv.features[27:])

    def test_context_with_explicit_role_sets_confidence_1(self):
        extractor = FeatureExtractor(_make_registry())
        ctx = RouteContext(agent_role="research", turn_index=2)
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=ctx)
        assert len(fv.features) == 35
        assert fv.features[32] == pytest.approx(1.0)  # agent_role_confidence
        assert fv.features[34] == pytest.approx(1.0)  # has_agent_context

    def test_turn_index_normalized(self):
        extractor = FeatureExtractor(_make_registry())
        ctx = RouteContext(turn_index=10)
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=ctx)
        assert fv.features[27] == pytest.approx(0.5)  # 10/20

    def test_turn_index_capped_at_one(self):
        extractor = FeatureExtractor(_make_registry())
        ctx = RouteContext(turn_index=100)
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o", context=ctx)
        assert fv.features[27] == pytest.approx(1.0)

    def test_feature_names_length(self):
        extractor = FeatureExtractor(_make_registry())
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o")
        assert len(fv.feature_names) == 35

    def test_backward_compat_no_context_arg(self):
        """extract() with no context kwarg still returns 35 features with zeros."""
        extractor = FeatureExtractor(_make_registry())
        fv = extractor.extract([{"role": "user", "content": "hello"}], "gpt-4o")
        assert len(fv.features) == 35
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_features.py::TestContextFeatures -v
```

- [ ] **Step 3: Implement feature extensions in `features.py`**

Add to `FeatureExtractor` class body:

```python
CONTEXT_FEATURE_NAMES = [
    "turn_index_norm",           # 27
    "conv_token_density",        # 28
    "correction_rate",           # 29
    "topic_drift",               # 30
    "agent_role_type",           # 31
    "agent_role_confidence",     # 32
    "messages_in_context_norm",  # 33
    "has_agent_context",         # 34
]

ALL_FEATURE_NAMES = (
    MESSAGE_FEATURE_NAMES + MODEL_FEATURE_NAMES + INTERACTION_FEATURE_NAMES
    + CONTEXT_FEATURE_NAMES
)
```

Update `extract()` signature and body:

```python
def extract(
    self,
    messages: list[dict[str, str]],
    model_id: str,
    context: "RouteContext | None" = None,
) -> FeatureVector:
    """Extract features from messages, model metadata, and optional context.

    Returns a 35-dimensional feature vector. The last 8 dims are all 0.0
    when context=None, preserving backward compatibility.
    """
    msg_features = self._extract_message_features(messages)
    model_features = self._extract_model_features(model_id)
    interaction_features = self._extract_interaction_features(msg_features, model_features)
    context_features = self._extract_context_features(messages, context)
    return FeatureVector(
        features=msg_features + model_features + interaction_features + context_features,
        feature_names=list(self.ALL_FEATURE_NAMES),
    )
```

Add `_extract_context_features()`:

```python
def _extract_context_features(
    self,
    messages: list[dict[str, str]],
    context: "RouteContext | None",
) -> list[float]:
    """Extract 8 context features. All 0.0 when context is None."""
    from routesmith.predictor.agent_inferencer import AgentInferencer

    if context is None:
        return [0.0] * 8

    turn = context.turn_index or 0
    turn_index_norm = min(1.0, turn / 20.0)

    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    token_estimate = total_chars / 4.0  # approximate chars-to-tokens
    conv_token_density = token_estimate / max(turn + 1, 1) / 1000.0

    correction_count = float(context.metadata.get("correction_count", 0))
    correction_rate = correction_count / max(turn + 1, 1)

    topic_drift = float(context.metadata.get("topic_drift", 0.0))

    role = context.agent_role
    agent_role_type = float(AgentInferencer.role_ordinal(role))

    if role is not None and not context.metadata.get("role_inferred"):
        agent_role_confidence = 1.0
    else:
        agent_role_confidence = float(context.metadata.get("role_confidence", 0.0))

    messages_in_context_norm = min(1.0, len(messages) / 50.0)
    has_agent_context = 1.0

    return [
        turn_index_norm,
        conv_token_density,
        correction_rate,
        topic_drift,
        agent_role_type,
        agent_role_confidence,
        messages_in_context_norm,
        has_agent_context,
    ]
```

Add `TYPE_CHECKING` guard at the top of `features.py`:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from routesmith.config import RouteContext
```

- [ ] **Step 4: Update `LinTSPredictor` in `lints.py`**

Change `d = 27` to `d = 35` and update `_features()`:

```python
d = 35  # matches FeatureExtractor full output (27 base + 8 context)
```

```python
def _features(self, messages: list[dict], model_id: str, context=None) -> np.ndarray:
    fv = self._extractor.extract(messages, model_id, context=context)
    x = np.array(fv.features[:35], dtype=np.float64)
    if len(x) < 35:
        x = np.pad(x, (0, 35 - len(x)))
    return x
```

Update `predict()` and `update()` in `LinTSPredictor` to accept and pass `context=None`:

```python
def predict(self, messages: list[dict], model_ids: list[str], context=None) -> list:
    results = []
    for model_id in model_ids:
        arm_idx = self._arm_index.get(model_id)
        ...
        x = self._features(messages, model_id, context=context)
        ...
```

```python
def update(self, messages: list[dict], model_id: str, actual_quality: float,
           reward_override: float | None = None, context=None) -> None:
    ...
    x = self._features(messages, model_id, context=context)
    ...
```

Update `LinUCBPredictor._get_context()` to accept and pass `context`:

```python
def _get_context(self, messages, model_id, context=None) -> np.ndarray:
    fv = self._extractor.extract(messages, model_id, context=context)
    ...
```

Update `LinUCBPredictor.predict()` and `update()` signatures with `context=None`.

- [ ] **Step 5: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_features.py tests/test_lints.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/routesmith/predictor/features.py src/routesmith/predictor/lints.py src/routesmith/predictor/linucb.py tests/test_features.py
git commit -m "feat(features): expand feature vector from 27 to 35 dims with 8 context features"
```

---

## Task 6: `LinTSPredictor` arm add/remove + JSON state persistence

**Files:**
- Modify: `src/routesmith/predictor/lints.py`
- Test: `tests/test_lints.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_lints.py
import json
from routesmith.predictor.lints import LinTSPredictor, LinTSArm
from routesmith.registry.models import ModelRegistry


def _make_lints():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    reg.register("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.7)
    return LinTSPredictor(registry=reg)


class TestLinTSArmLifecycle:
    def test_add_new_arm(self):
        p = _make_lints()
        p.add_arm("claude-haiku")
        assert "claude-haiku" in p._arm_index
        assert len(p._arm_names) == 3
        assert p._router.n_arms == 3

    def test_add_existing_arm_is_noop(self):
        p = _make_lints()
        p.add_arm("gpt-4o")
        assert len(p._arm_names) == 2

    def test_remove_arm(self):
        p = _make_lints()
        p.remove_arm("gpt-4o-mini")
        assert "gpt-4o-mini" not in p._arm_index
        assert len(p._arm_names) == 1
        assert p._router.n_arms == 1

    def test_remove_arm_reindexes_remaining(self):
        p = _make_lints()
        p.add_arm("claude-haiku")
        # arms: gpt-4o=0, gpt-4o-mini=1, claude-haiku=2
        p.remove_arm("gpt-4o-mini")
        # arms: gpt-4o=0, claude-haiku=1
        assert p._arm_index["claude-haiku"] == 1
        assert p._arm_index["gpt-4o"] == 0

    def test_remove_nonexistent_arm_is_noop(self):
        p = _make_lints()
        p.remove_arm("nonexistent")
        assert len(p._arm_names) == 2

    def test_serialize_deserialize_roundtrip(self):
        p = _make_lints()
        msgs = [{"role": "user", "content": "hello"}]
        p.update(msgs, "gpt-4o", actual_quality=0.9)
        blob = p.serialize_state()
        assert isinstance(blob, bytes)
        # Verify it's valid JSON (not pickle)
        state = json.loads(blob.decode())
        assert "router_state" in state

        p2 = _make_lints()
        p2.load_state(blob)
        assert p2._router._t == p._router._t

    def test_load_state_dimension_mismatch_cold_starts(self):
        """If stored d != current d, load_state does nothing (cold start)."""
        p = _make_lints()
        # Manually craft a state with wrong dimension
        bad_state = json.dumps({
            "router_state": {"n_arms": 2, "d": 5, "v_sq": 1.0, "t": 0,
                             "arms": []},
            "arm_names": ["gpt-4o", "gpt-4o-mini"],
            "arm_index": {"gpt-4o": 0, "gpt-4o-mini": 1},
            "total_updates": 0,
        }).encode()
        p2 = _make_lints()
        original_t = p2._router._t
        p2.load_state(bad_state)
        assert p2._router._t == original_t  # unchanged — cold start
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_lints.py::TestLinTSArmLifecycle -v
```

- [ ] **Step 3: Implement**

Add to `LinTSPredictor` in `lints.py`:

```python
def add_arm(self, model_id: str) -> None:
    """Add a new arm with optimistic initialization (A=I, b=0)."""
    if model_id in self._arm_index:
        return
    new_idx = len(self._arm_names)
    self._arm_names.append(model_id)
    self._arm_index[model_id] = new_idx
    self._router.arms.append(LinTSArm(d=self._router.d))
    self._router.n_arms += 1

def remove_arm(self, model_id: str) -> None:
    """Remove an arm and reindex remaining arms."""
    if model_id not in self._arm_index:
        return
    idx = self._arm_index.pop(model_id)
    self._arm_names.pop(idx)
    self._router.arms.pop(idx)
    self._router.n_arms -= 1
    # Reindex arms that shifted down
    for name in self._arm_names[idx:]:
        self._arm_index[name] -= 1

def serialize_state(self) -> bytes:
    """Serialize predictor state to JSON bytes (no pickle)."""
    import json
    state = {
        "router_state": self._router.get_state(),
        "arm_names": self._arm_names,
        "arm_index": self._arm_index,
        "total_updates": self._total_updates,
    }
    return json.dumps(state).encode()

def load_state(self, blob: bytes) -> None:
    """Load predictor state from JSON bytes.

    If stored feature dimension differs from current, skips load (cold start)
    rather than crashing on mismatched array shapes.
    """
    import json
    state = json.loads(blob.decode())
    router_state = state["router_state"]
    stored_d = router_state.get("d", self._router.d)
    if stored_d != self._router.d:
        return  # dimension mismatch — cold start
    self._router.load_state(router_state)
    self._arm_names = state["arm_names"]
    self._arm_index = state["arm_index"]
    self._total_updates = state.get("total_updates", 0)
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_lints.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/predictor/lints.py tests/test_lints.py
git commit -m "feat(lints): add add_arm/remove_arm and JSON serialize_state/load_state to LinTSPredictor"
```

---

## Task 7: `LinUCBPredictor` arm add/remove + JSON state persistence

**Files:**
- Modify: `src/routesmith/predictor/linucb.py`
- Create: `tests/test_linucb.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_linucb.py
import json
import pytest
from routesmith.predictor.linucb import LinUCBPredictor
from routesmith.registry.models import ModelRegistry


def _make_linucb():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    reg.register("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.7)
    return LinUCBPredictor(registry=reg)


class TestLinUCBArmLifecycle:
    def test_new_arm_predicted_after_add(self):
        p = _make_linucb()
        p.add_arm("claude-haiku")
        msgs = [{"role": "user", "content": "test"}]
        results = p.predict(msgs, ["gpt-4o", "claude-haiku"])
        assert any(r.model_id == "claude-haiku" for r in results)

    def test_add_existing_arm_preserves_state(self):
        p = _make_linucb()
        msgs = [{"role": "user", "content": "hi"}]
        p.update(msgs, "gpt-4o", 0.8)
        count_before = p._arms.get("gpt-4o", {}).get("count", 0)
        p.add_arm("gpt-4o")
        assert p._arms.get("gpt-4o", {}).get("count", 0) == count_before

    def test_remove_arm(self):
        p = _make_linucb()
        # Trigger lazy init
        p.predict([{"role": "user", "content": "hi"}], ["gpt-4o-mini"])
        p.remove_arm("gpt-4o-mini")
        assert "gpt-4o-mini" not in p._arms

    def test_remove_nonexistent_is_noop(self):
        p = _make_linucb()
        p.remove_arm("nonexistent")  # should not raise

    def test_serialize_deserialize_roundtrip(self):
        p = _make_linucb()
        msgs = [{"role": "user", "content": "test"}]
        p.update(msgs, "gpt-4o", 0.85)
        blob = p.serialize_state()
        assert isinstance(blob, bytes)
        state = json.loads(blob.decode())
        assert "arms" in state

        p2 = _make_linucb()
        p2.load_state(blob)
        assert p2._total_updates == p._total_updates
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_linucb.py -v
```

- [ ] **Step 3: Implement**

Add to `LinUCBPredictor` in `linucb.py`:

```python
def add_arm(self, model_id: str) -> None:
    """Signal intent to add a new arm.

    LinUCBPredictor uses dict-based lazy arm init — the arm is
    initialized automatically by _ensure_arm() on first predict/update.
    This method is a no-op if the arm already has state.
    """
    pass  # arm auto-initialized on first predict via _ensure_arm

def remove_arm(self, model_id: str) -> None:
    """Remove arm state for a deregistered model."""
    self._arms.pop(model_id, None)

def serialize_state(self) -> bytes:
    """Serialize arm states to JSON bytes (no pickle)."""
    import json
    state = {
        "total_updates": self._total_updates,
        "d": self._d,
        "arms": {
            model_id: {
                "A": arm["A"].tolist(),
                "b": arm["b"].tolist(),
                "count": arm["count"],
            }
            for model_id, arm in self._arms.items()
        },
    }
    return json.dumps(state).encode()

def load_state(self, blob: bytes) -> None:
    """Load arm states from JSON bytes.

    Skips load if stored feature dimension differs (cold start on mismatch).
    """
    import json
    import numpy as np
    state = json.loads(blob.decode())
    stored_d = state.get("d")
    if stored_d is not None and self._d is not None and stored_d != self._d:
        return  # dimension mismatch — cold start
    self._total_updates = state.get("total_updates", 0)
    if stored_d is not None:
        self._d = stored_d
    for model_id, arm_data in state.get("arms", {}).items():
        A = np.array(arm_data["A"], dtype=np.float64)  # noqa: N806
        self._arms[model_id] = {
            "A": A,
            "b": np.array(arm_data["b"], dtype=np.float64),
            "A_inv": np.linalg.inv(A),
            "count": arm_data["count"],
        }
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_linucb.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/predictor/linucb.py tests/test_linucb.py
git commit -m "feat(linucb): add add_arm/remove_arm and JSON serialize_state/load_state to LinUCBPredictor"
```

---

## Task 8: `AdaptivePredictor` arm add/remove + joblib RF persistence

**Files:**
- Modify: `src/routesmith/predictor/learner.py`
- Test: `tests/test_learner.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_learner.py
import json
import pytest
from routesmith.predictor.learner import AdaptivePredictor
from routesmith.registry.models import ModelRegistry


def _make_adaptive():
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    reg.register("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.7)
    return AdaptivePredictor(registry=reg)


class TestAdaptiveArmLifecycle:
    def test_add_arm_adds_ema_prior(self):
        p = _make_adaptive()
        p.add_arm("claude-haiku", quality_score=0.75)
        assert "claude-haiku" in p._ema_priors
        assert p._ema_priors["claude-haiku"] == pytest.approx(0.75)

    def test_add_existing_arm_is_noop(self):
        p = _make_adaptive()
        original = p._ema_priors["gpt-4o"]
        p.add_arm("gpt-4o", quality_score=0.1)
        assert p._ema_priors["gpt-4o"] == original  # not overwritten

    def test_remove_arm_removes_ema_prior(self):
        p = _make_adaptive()
        p.remove_arm("gpt-4o-mini")
        assert "gpt-4o-mini" not in p._ema_priors

    def test_remove_nonexistent_is_noop(self):
        p = _make_adaptive()
        p.remove_arm("nonexistent")  # should not raise

    def test_serialize_deserialize_cold_start_phase(self):
        p = _make_adaptive()
        blob = p.serialize_state()
        assert isinstance(blob, bytes)
        # Metadata portion is JSON
        state = json.loads(blob.decode().split("\n")[0] if b"\n" in blob else blob.decode())
        # If blob isn't pure JSON (has joblib segment), just check it's bytes
        p2 = _make_adaptive()
        p2.load_state(blob)
        assert p2._phase == p._phase
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_learner.py::TestAdaptiveArmLifecycle -v
```

- [ ] **Step 3: Implement**

Add to `AdaptivePredictor` in `learner.py`:

```python
def add_arm(self, model_id: str, quality_score: float = 0.8) -> None:
    """Register a new model arm with a cold-start EMA prior.

    No-op if model already has a prior (preserves learned value).
    """
    if model_id not in self._ema_priors:
        self._ema_priors[model_id] = quality_score

def remove_arm(self, model_id: str) -> None:
    """Remove EMA prior for a deregistered model.

    Historical training records in storage are unaffected.
    """
    self._ema_priors.pop(model_id, None)

def serialize_state(self) -> bytes:
    """Serialize predictor state to bytes.

    Format: JSON for metadata + optional joblib bytes for RF model.
    Uses a delimiter line to separate the two sections.
    joblib is only used for scikit-learn RF objects (no general pickle).
    """
    import json
    import io
    meta = {
        "phase": self._phase,
        "update_count": self._update_count,
        "ema_priors": self._ema_priors,
        "has_rf_model": self._model is not None,
    }
    meta_bytes = json.dumps(meta).encode()
    if self._model is not None:
        try:
            import joblib
            buf = io.BytesIO()
            joblib.dump(self._model, buf)
            # Delimiter: meta length as 4-byte big-endian int
            meta_len = len(meta_bytes).to_bytes(4, "big")
            return meta_len + meta_bytes + buf.getvalue()
        except Exception:
            pass  # RF serialization failed; store meta only
    meta_len = len(meta_bytes).to_bytes(4, "big")
    return meta_len + meta_bytes

def load_state(self, blob: bytes) -> None:
    """Load predictor state from bytes produced by serialize_state()."""
    import json
    import io
    if len(blob) < 4:
        return
    meta_len = int.from_bytes(blob[:4], "big")
    meta_bytes = blob[4:4 + meta_len]
    rf_bytes = blob[4 + meta_len:]
    meta = json.loads(meta_bytes.decode())
    self._phase = meta.get("phase", "cold_start")
    self._update_count = meta.get("update_count", 0)
    self._ema_priors.update(meta.get("ema_priors", {}))
    if rf_bytes and meta.get("has_rf_model"):
        try:
            import joblib
            self._model = joblib.load(io.BytesIO(rf_bytes))
        except Exception:
            self._model = None
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_learner.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/predictor/learner.py tests/test_learner.py
git commit -m "feat(learner): add add_arm/remove_arm and joblib+JSON serialize_state/load_state"
```

---

## Task 9: `ConversationTracker`

**Files:**
- Create: `src/routesmith/feedback/conversation.py`
- Create: `tests/test_conversation_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_conversation_tracker.py
from routesmith.feedback.conversation import ConversationTracker
from routesmith.config import RouteContext


class TestConversationTracker:
    def test_first_turn_has_index_zero(self):
        tracker = ConversationTracker(agent_role="research")
        ctx = tracker.next_context([{"role": "user", "content": "Hello"}])
        assert ctx.turn_index == 0
        assert ctx.agent_role == "research"
        assert ctx.conversation_id is not None

    def test_turn_index_increments(self):
        tracker = ConversationTracker()
        ctx1 = tracker.next_context([{"role": "user", "content": "Hi"}])
        ctx2 = tracker.next_context([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])
        assert ctx2.turn_index == ctx1.turn_index + 1

    def test_consistent_conversation_id(self):
        tracker = ConversationTracker(conversation_id="conv_abc")
        ctx1 = tracker.next_context([{"role": "user", "content": "Hi"}])
        ctx2 = tracker.next_context([{"role": "user", "content": "Hi"}])
        assert ctx1.conversation_id == "conv_abc"
        assert ctx2.conversation_id == "conv_abc"

    def test_correction_count_increments(self):
        tracker = ConversationTracker()
        tracker.next_context([{"role": "user", "content": "What is the capital of France?"}])
        tracker.next_context([{"role": "user", "content": "No, that's wrong. Try again."}])
        assert tracker._correction_count == 1

    def test_metadata_includes_correction_count(self):
        tracker = ConversationTracker()
        tracker.next_context([{"role": "user", "content": "Hi"}])
        ctx = tracker.next_context([{"role": "user", "content": "Actually that is incorrect."}])
        assert "correction_count" in ctx.metadata

    def test_auto_generated_conversation_id(self):
        tracker = ConversationTracker()
        ctx = tracker.next_context([{"role": "user", "content": "Hi"}])
        assert isinstance(ctx.conversation_id, str)
        assert len(ctx.conversation_id) > 0

    def test_record_outcome_does_not_raise(self):
        tracker = ConversationTracker()
        ctx = tracker.next_context([{"role": "user", "content": "Hi"}])
        tracker.record_outcome(ctx.metadata["request_id"], quality_score=0.9)
        tracker.record_outcome("nonexistent_id")
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_conversation_tracker.py -v
```

- [ ] **Step 3: Implement `src/routesmith/feedback/conversation.py`**

```python
"""Stateful conversation tracker for multi-turn agent sessions."""

from __future__ import annotations

import uuid
from typing import Any

from routesmith.config import RouteContext

_CORRECTION_WORDS = frozenset([
    "no", "wrong", "incorrect", "actually", "wait",
    "not right", "try again", "mistake",
])


class ConversationTracker:
    """Tracks turns in a conversation and emits RouteContext per call.

    Optional — callers who don't need stateful tracking can pass a
    RouteContext directly to completion() instead.

    Args:
        agent_role: Explicit agent role. If not provided, AgentInferencer
            attempts inference from the system prompt on each turn.
        agent_id: Optional agent instance identifier.
        conversation_id: Fixed conversation ID. Auto-generated if None.
        reward_fn: Optional per-instance reward function override.
    """

    def __init__(
        self,
        agent_role: str | None = None,
        agent_id: str | None = None,
        conversation_id: str | None = None,
        reward_fn: Any = None,
    ) -> None:
        self._agent_role = agent_role
        self._agent_id = agent_id
        self._conversation_id = conversation_id or uuid.uuid4().hex[:16]
        self.reward_fn = reward_fn

        self._turn_count = 0
        self._cumulative_chars = 0
        self._correction_count = 0
        self._first_user_content: str | None = None

    def next_context(self, messages: list[dict]) -> RouteContext:
        """Compute RouteContext for the next turn.

        Call immediately before each completion() call, passing the current
        full message list. Pass the returned context to completion(context=...).
        """
        # Record first user message for topic drift
        for msg in messages:
            if msg.get("role") == "user" and self._first_user_content is None:
                self._first_user_content = str(msg.get("content", ""))
                break

        # Detect correction in the latest user message (skip turn 0)
        if self._turn_count > 0:
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user:
                content_lower = str(last_user.get("content", "")).lower()
                if any(word in content_lower for word in _CORRECTION_WORDS):
                    self._correction_count += 1

        self._cumulative_chars += sum(
            len(str(m.get("content", ""))) for m in messages
        )

        # Topic drift: word overlap between first and current user message
        topic_drift = 0.0
        last_user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if self._first_user_content and last_user_msg and self._turn_count > 0:
            first_words = set(self._first_user_content.lower().split())
            curr_words = set(str(last_user_msg.get("content", "")).lower().split())
            if first_words:
                overlap = len(first_words & curr_words) / len(first_words)
                topic_drift = 1.0 - min(1.0, overlap)

        request_id = uuid.uuid4().hex[:16]

        ctx = RouteContext(
            agent_id=self._agent_id,
            agent_role=self._agent_role,
            conversation_id=self._conversation_id,
            turn_index=self._turn_count,
            metadata={
                "correction_count": self._correction_count,
                "topic_drift": topic_drift,
                "cumulative_token_estimate": self._cumulative_chars / 4.0,
                "request_id": request_id,
            },
        )

        self._turn_count += 1
        return ctx

    def record_outcome(
        self,
        request_id: str,
        quality_score: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Optional: record outcome for a completed turn.

        Implicit signals are collected automatically by FeedbackCollector
        on every completion() call. Call this only when the application
        has an explicit quality signal (task completion, user rating, etc.).
        Does nothing by default — callers wire this to rs.record_outcome()
        if they need the explicit signal to reach the predictor.
        """
        pass

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def conversation_id(self) -> str:
        return self._conversation_id
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_conversation_tracker.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/feedback/conversation.py tests/test_conversation_tracker.py
git commit -m "feat(feedback): add ConversationTracker for stateful multi-turn context tracking"
```

---

## Task 10: Per-role reward functions in `FeedbackCollector`

**Files:**
- Modify: `src/routesmith/feedback/collector.py`
- Test: `tests/test_feedback.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_feedback.py
from routesmith.feedback.collector import FeedbackCollector
from routesmith.config import RouteSmithConfig


class TestPerRoleRewardFns:
    def test_resolve_by_role(self):
        fn = lambda r, m: 0.95
        collector = FeedbackCollector(RouteSmithConfig(reward_fns={"research": fn}))
        assert collector.resolve_reward_fn(agent_role="research") is fn

    def test_falls_back_to_global(self):
        global_fn = lambda r, m: 0.5
        collector = FeedbackCollector(RouteSmithConfig(reward_fn=global_fn))
        assert collector.resolve_reward_fn(agent_role="coding") is global_fn

    def test_returns_none_when_nothing_configured(self):
        collector = FeedbackCollector(RouteSmithConfig())
        assert collector.resolve_reward_fn(agent_role="research") is None

    def test_role_fn_takes_priority_over_global(self):
        role_fn = lambda r, m: 0.99
        global_fn = lambda r, m: 0.5
        collector = FeedbackCollector(RouteSmithConfig(
            reward_fn=global_fn, reward_fns={"research": role_fn}
        ))
        assert collector.resolve_reward_fn(agent_role="research") is role_fn

    def test_none_role_falls_back_to_global(self):
        global_fn = lambda r, m: 0.5
        collector = FeedbackCollector(RouteSmithConfig(reward_fn=global_fn))
        assert collector.resolve_reward_fn(agent_role=None) is global_fn
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_feedback.py::TestPerRoleRewardFns -v
```

- [ ] **Step 3: Implement**

Add to `FeedbackCollector` in `collector.py`:

```python
def resolve_reward_fn(self, agent_role: str | None = None):
    """Resolve reward function with per-role priority.

    Resolution: config.reward_fns[agent_role] → config.reward_fn → None
    """
    if agent_role and agent_role in getattr(self.config, "reward_fns", {}):
        return self.config.reward_fns[agent_role]
    return getattr(self.config, "reward_fn", None)
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_feedback.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/feedback/collector.py tests/test_feedback.py
git commit -m "feat(feedback): add resolve_reward_fn with per-role priority to FeedbackCollector"
```

---

## Task 11: Business rules pre-filter in `Router`

**Files:**
- Modify: `src/routesmith/strategy/router.py`
- Test: `tests/test_router.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_router.py
import pytest
from routesmith.config import RouteSmithConfig, RouteContext, RoutingStrategy
from routesmith.registry.models import ModelRegistry
from routesmith.strategy.router import Router


def _make_router(rules):
    reg = ModelRegistry()
    reg.register("gpt-4o", cost_per_1k_input=0.005, cost_per_1k_output=0.015, quality_score=0.9)
    reg.register("gpt-4o-mini", cost_per_1k_input=0.00015, cost_per_1k_output=0.0006, quality_score=0.7)
    return Router(RouteSmithConfig(business_rules=rules), reg)


class TestBusinessRules:
    def test_rule_filters_candidate(self):
        def exclude_expensive(models, ctx):
            return [m for m in models if m.cost_per_1k_input < 0.001]

        router = _make_router([exclude_expensive])
        selected = router.route(
            [{"role": "user", "content": "hi"}],
            strategy=RoutingStrategy.DIRECT,
        )
        assert selected == "gpt-4o-mini"

    def test_rule_receives_context(self):
        seen = []

        def capture(models, ctx):
            seen.append(ctx)
            return models

        router = _make_router([capture])
        ctx = RouteContext(agent_role="research")
        router.route(
            [{"role": "user", "content": "hi"}],
            strategy=RoutingStrategy.DIRECT,
            context=ctx,
        )
        assert len(seen) == 1
        assert seen[0].agent_role == "research"

    def test_multiple_rules_applied_in_order(self):
        def rule1(models, ctx):
            return [m for m in models if m.cost_per_1k_input < 0.01]

        def rule2(models, ctx):
            return [m for m in models if m.quality_score >= 0.7]

        router = _make_router([rule1, rule2])
        selected = router.route(
            [{"role": "user", "content": "hi"}],
            strategy=RoutingStrategy.DIRECT,
        )
        assert selected in ("gpt-4o", "gpt-4o-mini")

    def test_rule_removing_all_models_raises(self):
        router = _make_router([lambda m, c: []])
        with pytest.raises(ValueError, match="business rules"):
            router.route(
                [{"role": "user", "content": "hi"}],
                strategy=RoutingStrategy.DIRECT,
            )
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_router.py::TestBusinessRules -v
```

- [ ] **Step 3: Implement**

In `src/routesmith/strategy/router.py`, add `context` param to `route()` and insert the business rules block before capability filtering:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from routesmith.config import RouteContext


def route(
    self,
    messages: list[dict[str, str]],
    strategy: RoutingStrategy | None = None,
    max_cost: float | None = None,
    min_quality: float | None = None,
    required_capabilities: set[str] | None = None,
    context: "RouteContext | None" = None,
) -> str:
    strategy = strategy or self.config.default_strategy
    models = self.registry.list_models()

    # Apply business rules before capability filtering
    for rule in getattr(self.config, "business_rules", []):
        models = rule(models, context)
        if not models:
            raise ValueError(
                "All models were filtered out by business rules. "
                "Check your RouteSmithConfig.business_rules."
            )

    # Capability filtering (existing logic)
    if required_capabilities:
        models = [m for m in models if required_capabilities.issubset(m.capabilities)]
        if not models:
            raise ValueError(
                f"No models satisfy required capabilities: {required_capabilities}"
            )

    model_ids = [m.model_id for m in models]
    # ... rest of existing route() logic, passing context to predictor.predict()
```

Also pass `context` through to `self.predictor.predict(messages, model_ids, context=context)` where the predictor is called.

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_router.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/strategy/router.py tests/test_router.py
git commit -m "feat(router): add business_rules pre-filter and context param to route()"
```

---

## Task 12: Wire `RouteContext` through `completion()` + predictor state on init

**Files:**
- Modify: `src/routesmith/client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_client.py
from unittest.mock import MagicMock, patch
from routesmith import RouteSmith
from routesmith.config import RouteContext, RouteSmithConfig


def _mock_litellm_response():
    return MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        id="resp_1",
        model="gpt-4o-mini",
    )


def _make_rs():
    rs = RouteSmith()
    rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                     cost_per_1k_output=0.015, quality_score=0.9)
    rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                     cost_per_1k_output=0.0006, quality_score=0.7)
    return rs


class TestCompletionWithContext:
    @patch("litellm.completion")
    def test_accepts_context_param(self, mock_litellm):
        mock_litellm.return_value = _mock_litellm_response()
        rs = _make_rs()
        ctx = RouteContext(agent_role="research", turn_index=2)
        response = rs.completion(
            messages=[{"role": "user", "content": "hello"}],
            context=ctx,
        )
        assert response is not None

    @patch("litellm.completion")
    def test_without_context_still_works(self, mock_litellm):
        mock_litellm.return_value = _mock_litellm_response()
        rs = _make_rs()
        response = rs.completion(messages=[{"role": "user", "content": "hello"}])
        assert response is not None

    @patch("litellm.completion")
    def test_agent_role_inferred_when_missing(self, mock_litellm):
        mock_litellm.return_value = _mock_litellm_response()
        rs = _make_rs()
        msgs = [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "hello"},
        ]
        ctx = RouteContext()  # no agent_role
        rs.completion(messages=msgs, context=ctx)  # should not raise
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_client.py::TestCompletionWithContext -v
```

- [ ] **Step 3: Implement**

In `src/routesmith/client.py`:

1. Add import: `from routesmith.config import ..., RouteContext`

2. Add `context: RouteContext | None = None` to `completion()` signature.

3. After capability detection and before routing, infer role when missing:

```python
if context is not None and context.agent_role is None:
    if not hasattr(self, "_agent_inferencer"):
        from routesmith.predictor.agent_inferencer import AgentInferencer
        self._agent_inferencer = AgentInferencer()
    role, confidence = self._agent_inferencer.infer(messages)
    if role is not None:
        context = RouteContext(
            agent_id=context.agent_id,
            agent_role=role,
            conversation_id=context.conversation_id,
            turn_index=context.turn_index,
            metadata={
                **context.metadata,
                "role_inferred": True,
                "role_confidence": confidence,
            },
        )
```

4. Pass `context` to `self.router.route(...)`.

5. Pass agent fields to `store_record` call (check where feedback is stored in `completion()` and add the four fields from context if context is not None).

6. Resolve per-role reward_fn:

```python
reward_fn = self.feedback.resolve_reward_fn(
    agent_role=context.agent_role if context else None
)
```

Pass `reward_fn` as `reward_override` when calling `predictor.update()` (via `record_outcome`).

7. In `__init__()`, after creating the router, add:

```python
if self.config.feedback_storage_path:
    self._load_predictor_state()
```

8. Add `_load_predictor_state()`:

```python
def _load_predictor_state(self) -> None:
    """Load persisted predictor weights from storage on startup."""
    blob = self.feedback._storage.load_predictor_state(self.config.predictor_type)
    if blob is not None:
        predictor = self.router.predictor
        if hasattr(predictor, "load_state"):
            try:
                predictor.load_state(blob)
            except Exception:
                pass  # corrupt or incompatible state; cold start
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_client.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/client.py tests/test_client.py
git commit -m "feat(client): add context param to completion(), agent role inference, predictor state load on init"
```

---

## Task 13: `deregister_model()` + `register_model()` predictor arm wiring + state save

**Files:**
- Modify: `src/routesmith/client.py`
- Modify: `src/routesmith/registry/models.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_client.py
class TestModelLifecycle:
    def test_deregister_removes_from_registry(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                         cost_per_1k_output=0.0006, quality_score=0.7)
        rs.deregister_model("gpt-4o-mini")
        ids = [m.model_id for m in rs.registry.list_models()]
        assert "gpt-4o-mini" not in ids

    def test_deregister_last_model_raises(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        with pytest.raises(ValueError, match="last registered model"):
            rs.deregister_model("gpt-4o")

    def test_deregister_nonexistent_is_noop(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.deregister_model("nonexistent")
        assert len(rs.registry.list_models()) == 1

    def test_register_model_adds_predictor_arm(self):
        rs = RouteSmith(config=RouteSmithConfig(predictor_type="lints"))
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("claude-haiku", cost_per_1k_input=0.00025,
                         cost_per_1k_output=0.00125, quality_score=0.75)
        predictor = rs.router.predictor
        assert "claude-haiku" in predictor._arm_index
```

Add `import pytest` at top of `tests/test_client.py` if missing.

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_client.py::TestModelLifecycle -v
```

- [ ] **Step 3: Add `deregister()` to `ModelRegistry` in `registry/models.py`**

Check if `deregister` exists; if not, add:

```python
def deregister(self, model_id: str) -> None:
    """Remove a model from the registry."""
    self._models.pop(model_id, None)
```

(Verify the internal storage attribute name — use `grep` for `self._models` or `self._registry` in `models.py` and match what exists.)

- [ ] **Step 4: Add `deregister_model()` and `_persist_predictor_state()` to `RouteSmith`**

```python
def deregister_model(self, model_id: str) -> None:
    """Remove a model from routing.

    Raises ValueError if it's the last registered model.
    Historical feedback records are preserved; predictor arm is retired.
    """
    if self.registry.get(model_id) is None:
        return
    if len(self.registry.list_models()) <= 1:
        raise ValueError(
            f"Cannot deregister '{model_id}': it is the last registered model."
        )
    self.registry.deregister(model_id)
    predictor = self.router.predictor
    if hasattr(predictor, "remove_arm"):
        predictor.remove_arm(model_id)
    self._persist_predictor_state()

def _persist_predictor_state(self) -> None:
    """Serialize current predictor state to storage (non-fatal on failure)."""
    if not self.config.feedback_storage_path:
        return
    predictor = self.router.predictor
    if hasattr(predictor, "serialize_state"):
        try:
            blob = predictor.serialize_state()
            self.feedback._storage.save_predictor_state(
                self.config.predictor_type, blob
            )
        except Exception:
            pass
```

Update `register_model()`: after `self.registry.register(...)`, add:

```python
predictor = self.router.predictor
if hasattr(predictor, "add_arm"):
    predictor.add_arm(model_id, quality_score=quality_score)
```

In `completion()`, after feedback recording, trigger periodic persistence:

```python
updates = getattr(self.router.predictor, "_total_updates", 0)
if updates > 0 and updates % 50 == 0:
    self._persist_predictor_state()
```

- [ ] **Step 5: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_client.py::TestModelLifecycle -v
```

- [ ] **Step 6: Commit**

```bash
git add src/routesmith/client.py src/routesmith/registry/models.py tests/test_client.py
git commit -m "feat(client): add deregister_model(), arm wiring on register_model(), predictor state persistence"
```

---

## Task 14: `recommend_model_for_agent()`

**Files:**
- Modify: `src/routesmith/client.py`
- Modify: `src/routesmith/feedback/storage.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_client.py
class TestRecommendModelForAgent:
    def test_returns_none_for_none_role(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        assert rs.recommend_model_for_agent(None) is None

    def test_returns_none_when_insufficient_samples(self):
        rs = RouteSmith()
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        assert rs.recommend_model_for_agent("research") is None

    def test_returns_recommendation_with_enough_samples(self):
        rs = RouteSmith(config=RouteSmithConfig(feedback_storage_path=":memory:"))
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                         cost_per_1k_output=0.0006, quality_score=0.7)
        storage = rs.feedback._storage
        for i in range(55):
            storage.store_record(
                request_id=f"req_{i}",
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "research question"}],
                latency_ms=200.0,
                quality_score=0.9,
                agent_role="research",
            )
        result = rs.recommend_model_for_agent("research", min_samples=50)
        assert result is not None
        assert result["model"] == "gpt-4o"
        assert result["sample_count"] >= 50
        assert "confidence" in result
        assert "new_models_to_explore" in result

    def test_new_models_to_explore_includes_models_with_no_data(self):
        rs = RouteSmith(config=RouteSmithConfig(feedback_storage_path=":memory:"))
        rs.register_model("gpt-4o", cost_per_1k_input=0.005,
                         cost_per_1k_output=0.015, quality_score=0.9)
        rs.register_model("gpt-4o-mini", cost_per_1k_input=0.00015,
                         cost_per_1k_output=0.0006, quality_score=0.7)
        storage = rs.feedback._storage
        for i in range(55):
            storage.store_record(
                request_id=f"req_{i}",
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                latency_ms=100.0,
                quality_score=0.9,
                agent_role="research",
            )
        result = rs.recommend_model_for_agent("research", min_samples=50)
        assert "gpt-4o-mini" in result["new_models_to_explore"]
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_client.py::TestRecommendModelForAgent -v
```

- [ ] **Step 3: Add `get_records_by_agent_role()` to `FeedbackStorage`**

```python
def get_records_by_agent_role(
    self,
    agent_role: str,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Fetch feedback records with quality scores for a given agent role."""
    conn = self._get_conn()
    rows = conn.execute(
        """SELECT model_id, quality_score, latency_ms
           FROM feedback_records
           WHERE agent_role = ? AND quality_score IS NOT NULL
           ORDER BY created_at DESC LIMIT ?""",
        (agent_role, limit),
    ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Add `recommend_model_for_agent()` to `RouteSmith`**

```python
def recommend_model_for_agent(
    self,
    agent_role: str | None,
    min_samples: int = 50,
) -> dict | None:
    """Return the historically best model for an agent role.

    Returns None when agent_role is None or fewer than min_samples
    quality records exist for the role.

    Returns a dict with:
        model: str — recommended model_id
        confidence: float — 0-1, based on sample count
        sample_count: int — records for the recommended model
        avg_quality: float
        avg_cost_usd: float
        new_models_to_explore: list[str] — registered models with < min_samples data
    """
    if agent_role is None:
        return None

    records = self.feedback._storage.get_records_by_agent_role(agent_role)
    if len(records) < min_samples:
        return None

    from collections import defaultdict
    model_quality: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["quality_score"] is not None:
            model_quality[r["model_id"]].append(float(r["quality_score"]))

    registered = {m.model_id: m for m in self.registry.list_models()}
    best_model = None
    best_efficiency = -1.0
    model_stats: dict[str, dict] = {}

    for model_id, qualities in model_quality.items():
        if model_id not in registered:
            continue
        model = registered[model_id]
        avg_quality = sum(qualities) / len(qualities)
        avg_cost = (model.cost_per_1k_input + model.cost_per_1k_output) / 2
        efficiency = avg_quality / max(avg_cost * 1000, 1e-6)
        model_stats[model_id] = {
            "avg_quality": avg_quality,
            "avg_cost_usd": avg_cost,
            "sample_count": len(qualities),
        }
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_model = model_id

    if best_model is None:
        return None

    new_models_to_explore = [
        m.model_id for m in self.registry.list_models()
        if len(model_quality.get(m.model_id, [])) < min_samples
        and m.model_id != best_model
    ]
    total_samples = sum(len(q) for q in model_quality.values())
    confidence = min(1.0, total_samples / (min_samples * 3))
    stats = model_stats[best_model]

    return {
        "model": best_model,
        "confidence": round(confidence, 3),
        "sample_count": stats["sample_count"],
        "avg_quality": round(stats["avg_quality"], 3),
        "avg_cost_usd": round(stats["avg_cost_usd"], 6),
        "new_models_to_explore": new_models_to_explore,
    }
```

- [ ] **Step 5: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_client.py::TestRecommendModelForAgent -v
```

- [ ] **Step 6: Run all client tests**

```bash
.venv/bin/pytest tests/test_client.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/routesmith/client.py src/routesmith/feedback/storage.py tests/test_client.py
git commit -m "feat(client): add recommend_model_for_agent() and register_reward_fn()"
```

---

## Task 15: `register_reward_fn()` on `RouteSmith`

**Files:**
- Modify: `src/routesmith/client.py`
- Test: `tests/test_client.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_client.py
class TestRegisterRewardFn:
    def test_adds_to_config(self):
        rs = RouteSmith()
        fn = lambda r, m: 0.95
        rs.register_reward_fn("research", fn)
        assert rs.config.reward_fns["research"] is fn

    def test_overrides_existing(self):
        fn1 = lambda r, m: 0.8
        fn2 = lambda r, m: 0.95
        rs = RouteSmith(config=RouteSmithConfig(reward_fns={"research": fn1}))
        rs.register_reward_fn("research", fn2)
        assert rs.config.reward_fns["research"] is fn2
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_client.py::TestRegisterRewardFn -v
```

- [ ] **Step 3: Implement**

Add to `RouteSmith`:

```python
def register_reward_fn(self, agent_role: str, fn: "Callable[..., float]") -> None:
    """Register a per-role reward function at runtime.

    Takes priority over the global reward_fn/reward_expr for this role.
    """
    self.config.reward_fns[agent_role] = fn
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_client.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/client.py tests/test_client.py
git commit -m "feat(client): add register_reward_fn() for runtime per-role reward registration"
```

---

## Task 16: LangChain integration updates

**Files:**
- Modify: `src/routesmith/integrations/langchain.py`
- Test: `tests/test_langchain_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_langchain_integration.py
from routesmith.integrations.langchain import ChatRouteSmith
from routesmith.feedback.conversation import ConversationTracker


class TestChatRouteSmithAgentContext:
    def test_agent_role_field(self):
        llm = ChatRouteSmith(agent_role="research")
        assert llm.agent_role == "research"

    def test_track_conversation_creates_tracker(self):
        llm = ChatRouteSmith(track_conversation=True, agent_role="coding")
        assert llm._tracker is not None
        assert isinstance(llm._tracker, ConversationTracker)
        assert llm._tracker._agent_role == "coding"

    def test_no_tracker_when_false(self):
        llm = ChatRouteSmith(track_conversation=False)
        assert llm._tracker is None

    def test_reward_fn_field(self):
        fn = lambda r, m: 0.9
        llm = ChatRouteSmith(reward_fn=fn)
        assert llm.reward_fn is fn
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_langchain_integration.py::TestChatRouteSmithAgentContext -v
```

- [ ] **Step 3: Implement**

In `src/routesmith/integrations/langchain.py`, add four new Pydantic fields to `ChatRouteSmith`:

```python
agent_role: str | None = None
conversation_id: str | None = None
track_conversation: bool = False
reward_fn: Any = None
```

Update `__init__()` to create the tracker:

```python
def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self._tracker = None
    if self.track_conversation:
        from routesmith.feedback.conversation import ConversationTracker
        self._tracker = ConversationTracker(
            agent_role=self.agent_role,
            conversation_id=self.conversation_id,
            reward_fn=self.reward_fn,
        )
```

Add `_build_context()` helper:

```python
def _build_context(self, messages_dicts: list[dict]) -> "RouteContext | None":
    from routesmith.config import RouteContext
    if self._tracker is not None:
        return self._tracker.next_context(messages_dicts)
    if self.agent_role is not None:
        return RouteContext(agent_role=self.agent_role, conversation_id=self.conversation_id)
    return None
```

In `_generate()`, before calling `self.routesmith.completion(...)`, compute context:

```python
msgs_dicts = _langchain_messages_to_dicts(messages)
ctx = self._build_context(msgs_dicts)
response = self.routesmith.completion(
    messages=msgs_dicts,
    context=ctx,
    ...
)
```

Apply the same `_build_context` call in `_agenerate()`, `_stream()`, and `_astream()`.

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_langchain_integration.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/routesmith/integrations/langchain.py tests/test_langchain_integration.py
git commit -m "feat(langchain): add agent_role, track_conversation, reward_fn to ChatRouteSmith"
```

---

## Task 17: DSPy + Anthropic integration updates

**Files:**
- Modify: `src/routesmith/integrations/dspy.py`
- Modify: `src/routesmith/integrations/anthropic.py`
- Test: `tests/test_dspy_integration.py`
- Test: `tests/test_anthropic_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_dspy_integration.py
from routesmith.integrations.dspy import RouteSmithLM
from routesmith.feedback.conversation import ConversationTracker


class TestRouteSmithLMAgentContext:
    def test_agent_role_stored(self):
        lm = RouteSmithLM(agent_role="coding")
        assert lm.agent_role == "coding"

    def test_track_conversation_creates_tracker(self):
        lm = RouteSmithLM(track_conversation=True, agent_role="qa")
        assert isinstance(lm._tracker, ConversationTracker)

    def test_no_tracker_by_default(self):
        lm = RouteSmithLM()
        assert lm._tracker is None
```

```python
# Add to tests/test_anthropic_integration.py
from routesmith.integrations.anthropic import RouteSmithAnthropic
from routesmith.feedback.conversation import ConversationTracker


class TestRouteSmithAnthropicAgentContext:
    def test_agent_role_stored(self):
        client = RouteSmithAnthropic(agent_role="summarizer")
        assert client.agent_role == "summarizer"

    def test_track_conversation_creates_tracker(self):
        client = RouteSmithAnthropic(track_conversation=True, agent_role="research")
        assert isinstance(client._tracker, ConversationTracker)

    def test_no_tracker_by_default(self):
        client = RouteSmithAnthropic()
        assert client._tracker is None
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_dspy_integration.py::TestRouteSmithLMAgentContext tests/test_anthropic_integration.py::TestRouteSmithAnthropicAgentContext -v
```

- [ ] **Step 3: Implement `RouteSmithLM`** in `dspy.py`

Add four params to `__init__()`:

```python
def __init__(
    self,
    routesmith: RouteSmith | None = None,
    config: RouteSmithConfig | None = None,
    agent_role: str | None = None,
    conversation_id: str | None = None,
    track_conversation: bool = False,
    reward_fn: Any = None,
) -> None:
    self._rs = routesmith or RouteSmith(config=config)
    self.history: list[dict[str, Any]] = []
    self.agent_role = agent_role
    self._tracker = None
    if track_conversation:
        from routesmith.feedback.conversation import ConversationTracker
        self._tracker = ConversationTracker(
            agent_role=agent_role,
            conversation_id=conversation_id,
            reward_fn=reward_fn,
        )
```

In `__call__()`, build context before `completion()`:

```python
from routesmith.config import RouteContext
ctx = None
if self._tracker is not None:
    ctx = self._tracker.next_context(messages)
elif self.agent_role is not None:
    ctx = RouteContext(agent_role=self.agent_role)
response = self._rs.completion(messages=messages, context=ctx, **kwargs)
```

- [ ] **Step 4: Implement `RouteSmithAnthropic`** in `anthropic.py`

Add four params to `RouteSmithAnthropic.__init__()`:

```python
def __init__(
    self,
    routesmith: RouteSmith | None = None,
    config: RouteSmithConfig | None = None,
    agent_role: str | None = None,
    conversation_id: str | None = None,
    track_conversation: bool = False,
    reward_fn: Any = None,
) -> None:
    self._rs = routesmith or RouteSmith(config=config)
    self.agent_role = agent_role
    self._tracker = None
    if track_conversation:
        from routesmith.feedback.conversation import ConversationTracker
        self._tracker = ConversationTracker(
            agent_role=agent_role,
            conversation_id=conversation_id,
            reward_fn=reward_fn,
        )
    self.messages = _MessagesResource(self._rs, self._tracker, agent_role)
```

Update `_MessagesResource`:

```python
class _MessagesResource:
    def __init__(self, rs: RouteSmith, tracker=None, agent_role=None) -> None:
        self._rs = rs
        self._tracker = tracker
        self._agent_role = agent_role

    def create(self, *, model="auto", messages, max_tokens=1024,
               system=None, temperature=None, **kwargs) -> Message:
        from routesmith.config import RouteContext
        openai_messages: list[dict[str, Any]] = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(_anthropic_to_openai_messages(messages))

        ctx = None
        if self._tracker is not None:
            ctx = self._tracker.next_context(openai_messages)
        elif self._agent_role is not None:
            ctx = RouteContext(agent_role=self._agent_role)

        extra: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            extra["temperature"] = temperature

        response = self._rs.completion(messages=openai_messages, context=ctx, **extra)
        selected = getattr(self._rs._last_routing_metadata, "model_selected", model)
        return _litellm_to_anthropic_message(response, selected)
```

- [ ] **Step 5: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_dspy_integration.py tests/test_anthropic_integration.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/routesmith/integrations/dspy.py src/routesmith/integrations/anthropic.py tests/test_dspy_integration.py tests/test_anthropic_integration.py
git commit -m "feat(integrations): add agent_role and track_conversation to RouteSmithLM and RouteSmithAnthropic"
```

---

## Task 18: Proxy handler — `X-RouteSmith-*` headers

**Files:**
- Modify: `src/routesmith/proxy/handler.py`
- Test: `tests/test_proxy.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_proxy.py
from routesmith.proxy.handler import extract_route_context_from_headers
from routesmith.config import RouteContext


class TestProxyRouteContextHeaders:
    def test_extracts_all_fields(self):
        headers = {
            "x-routesmith-agent-id": "agent_42",
            "x-routesmith-agent-role": "research",
            "x-routesmith-conversation-id": "conv_abc",
        }
        ctx = extract_route_context_from_headers(headers)
        assert ctx is not None
        assert ctx.agent_id == "agent_42"
        assert ctx.agent_role == "research"
        assert ctx.conversation_id == "conv_abc"

    def test_returns_none_when_no_routesmith_headers(self):
        assert extract_route_context_from_headers({}) is None
        assert extract_route_context_from_headers({"content-type": "application/json"}) is None

    def test_partial_headers(self):
        ctx = extract_route_context_from_headers({"x-routesmith-agent-role": "summarizer"})
        assert ctx is not None
        assert ctx.agent_role == "summarizer"
        assert ctx.agent_id is None

    def test_case_insensitive(self):
        ctx = extract_route_context_from_headers({"X-RouteSmith-Agent-Role": "coding"})
        assert ctx is not None
        assert ctx.agent_role == "coding"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv/bin/pytest tests/test_proxy.py::TestProxyRouteContextHeaders -v
```

- [ ] **Step 3: Implement**

Add to `src/routesmith/proxy/handler.py`:

```python
from routesmith.config import RouteContext

_ROUTESMITH_HEADERS = {
    "x-routesmith-agent-id": "agent_id",
    "x-routesmith-agent-role": "agent_role",
    "x-routesmith-conversation-id": "conversation_id",
}


def extract_route_context_from_headers(
    headers: dict[str, str],
) -> RouteContext | None:
    """Build RouteContext from X-RouteSmith-* HTTP headers.

    Returns None if no RouteSmith headers are present.
    Header matching is case-insensitive.
    """
    normalized = {k.lower(): v for k, v in headers.items()}
    kwargs: dict = {}
    for header, field in _ROUTESMITH_HEADERS.items():
        if header in normalized:
            kwargs[field] = normalized[header]
    return RouteContext(**kwargs) if kwargs else None
```

In the proxy request handler, call `extract_route_context_from_headers()` on incoming request headers and pass the result as `context=ctx` to `rs.completion()`. (Locate the existing `completion()` call in `handler.py` and thread `ctx` through it.)

- [ ] **Step 4: Run — expect PASS**

```bash
.venv/bin/pytest tests/test_proxy.py -v
```

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short
```

All tests should pass. Fix any regressions before committing.

- [ ] **Step 6: Commit**

```bash
git add src/routesmith/proxy/handler.py tests/test_proxy.py
git commit -m "feat(proxy): extract RouteContext from X-RouteSmith-* headers"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short
```

Expected: all tests pass including the original 255.

- [ ] **Run linter**

```bash
.venv/bin/ruff check src/
```

Fix any reported issues before marking complete.

- [ ] **Commit any cleanup**

```bash
git add -p
git commit -m "chore: cleanup after agentic routing implementation"
```
