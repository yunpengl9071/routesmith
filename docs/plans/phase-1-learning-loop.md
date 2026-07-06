# Phase 1 — Close the learning loop

**Why this phase IS the product:** OpenRouter's Auto Router is pre-trained by NotDiamond and
never learns from the customer's traffic. RouteSmith's entire competitive thesis is "learns
YOUR workload" — but today, in the flagship drop-in deployment (the proxy), learning is
unreachable: there is no feedback endpoint, implicit signals never update the predictor, and
the default `feedback_sample_rate=0.1` means 90% of requests can't even receive stored
explicit feedback. This phase makes the thesis true.

**Exit criteria:** G4 (`tests/test_convergence.py`) and G6 (`tests/test_proxy_feedback_e2e.py`) green.

Task order: P1.1 → P1.2 → P1.3 → P1.4 → P1.5. (P1.3 is independent of P1.2.)

---

## Task P1.1 — Record all requests; sample the *evaluation*, not the *recording*  (size: S)

The FK trap: `record_outcome()` can only persist an explicit signal if the request was
recorded, but only 10% are (`feedback_sample_rate: float = 0.1`, config.py line ~97). A user
who dutifully calls `record_outcome` loses 90% of their feedback.

**Files:** `src/routesmith/config.py`, `src/routesmith/cli/yaml_loader.py`,
`tests/test_feedback.py` (extend)

**Spec:**
1. In `config.py`, change `feedback_sample_rate: float = 0.1` → `1.0`. Update comment:
   `# Fraction of requests to record (1.0 = all; evaluation is sampled separately)`.
2. Add a new field directly below it: `judge_sample_rate: float = 0.05`
   `# Fraction of recorded requests scored by the LLM judge (see JudgeConfig)`.
   (Consumed in P1.4.)
3. `yaml_loader.py`: parse `feedback.judge_sample_rate` if present (follow the existing
   pattern for `feedback_sample_rate` — grep `sample_rate` in that file).

**Tests:**
- `test_feedback.py::test_default_sample_rate_records_all` (new) — 20 completions with
  default config → 20 records in storage (query via the same accessor existing feedback
  tests use).
- `test_feedback.py::test_record_outcome_always_persists_signal` (new) — completion then
  `record_outcome(request_id, score=0.9)` → an explicit signal row exists for that
  request_id. Repeat 10× in a loop; all 10 persist.

**Acceptance:** both tests pass. Memory note: a record is small (~1 KB); 100k requests ≈
100 MB SQLite — acceptable; do not build rotation in this task.

---

## Task P1.2 — Proxy feedback endpoint `POST /v1/feedback`  (size: M)

The single biggest loop-closure gap: proxy users have no way to submit outcomes.

**Files:** `src/routesmith/proxy/server.py`, `src/routesmith/proxy/handler.py`,
`tests/test_proxy_feedback.py` (new), `README.md`

**Spec — handler (`handler.py`):** add method to `RequestHandler`:

```python
async def handle_feedback(self, body: bytes) -> tuple[dict, int]:
    """Process an outcome report. Returns (response_dict, http_status).

    Request JSON:
      request_id  str, required — from routesmith_metadata.request_id
      score       float in [0,1], optional
      success     bool, optional (exactly one of score/success required)
    """
    import json
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, 400

    request_id = data.get("request_id")
    score = data.get("score")
    success = data.get("success")
    if not isinstance(request_id, str) or not request_id:
        return {"error": {"message": "'request_id' (string) is required",
                          "type": "invalid_request_error"}}, 400
    if (score is None) == (success is None):
        return {"error": {"message": "Provide exactly one of 'score' or 'success'",
                          "type": "invalid_request_error"}}, 400
    if score is not None and not (isinstance(score, (int, float)) and 0.0 <= score <= 1.0):
        return {"error": {"message": "'score' must be a number in [0, 1]",
                          "type": "invalid_request_error"}}, 400

    found = self.routesmith.record_outcome(request_id=request_id, score=score, success=success)
    if not found:
        return {"error": {"message": f"Unknown request_id: {request_id}",
                          "type": "not_found"}}, 404
    return {"status": "ok", "request_id": request_id}, 200
```

First **verify `record_outcome`'s actual signature** (client.py, `def record_outcome(`, line
~659) and its return value (the audit found `found = self.feedback.record_outcome(...)` —
confirm the client method returns that bool; if it currently returns `None`, change it to
return the bool as part of this task).

**Spec — server (`server.py`):** in `_route_request`, before the 404 fallthrough:

```python
if path == "/v1/feedback" and method == "POST":
    result, status = await self.handler.handle_feedback(body)
    await self._send_json(writer, result, status)
    return
```

Add `404` handling already exists; ensure `_send_json` status-text map covers 404 (it does).

**Docs:** README gains a "Closing the loop over HTTP" section:

```bash
curl -s localhost:9119/v1/chat/completions -d '{"model":"auto","messages":[...]}' \
  | jq -r .routesmith_metadata.request_id   # → e.g. "3f2a..."
curl -s localhost:9119/v1/feedback -d '{"request_id":"3f2a...","score":0.9}'
```

**Tests** (`tests/test_proxy_feedback.py` — follow `tests/test_proxy.py`'s existing pattern
for driving the handler directly, litellm mocked):
- `test_feedback_updates_predictor` — completion via `handler.handle_completion`; extract
  `routesmith_metadata["request_id"]` from the response dict; call `handle_feedback` with
  score 1.0 → status 200 AND the predictor's update counter incremented
  (`rs.router.predictor._total_updates == 1` for lints/linucb).
- `test_feedback_unknown_request_id_404`.
- `test_feedback_requires_exactly_one_of_score_success` — neither → 400; both → 400.
- `test_feedback_score_out_of_range_400` — score 1.5 → 400.
- `test_feedback_invalid_json_400`.

**Acceptance:** all tests pass; endpoint documented.

---

## Task P1.3 — Feed implicit signals into the predictor  (size: M)

Signals (refusal/empty/error/truncation) are extracted (`feedback/signals.py`) and stored,
but never reach `predictor.update()`. Wire the **negative** signals only — never synthesize
positive rewards from silence, or the bandit self-congratulates on unverified output.

**Files:** `src/routesmith/config.py`, `src/routesmith/feedback/signals.py`,
`src/routesmith/client.py`, `tests/test_implicit_learning.py` (new)

**Spec:**
1. `config.py`: add to `RouteSmithConfig` (near `feedback_enabled`):
   `implicit_feedback_enabled: bool = True`
   `# Feed negative implicit signals (refusal/empty/error/truncation) to the predictor`.
2. `signals.py`: add a module-level pure function (read the file first; use the real signal
   name constants it defines):

   ```python
   # Quality ascribed to a response exhibiting each implicit failure signal.
   IMPLICIT_QUALITY = {
       "error": 0.05,
       "refusal": 0.05,
       "empty": 0.05,
       "truncation": 0.40,
   }

   def implicit_quality(signals) -> float | None:
       """Worst implied quality across triggered signals; None if no negative signal."""
       triggered = [IMPLICIT_QUALITY[s.name] for s in signals
                    if s.name in IMPLICIT_QUALITY and s.value >= 0.5]
       return min(triggered) if triggered else None
   ```
   Adapt attribute access (`s.name` / `s.value`) to the actual signal dataclass in that file
   — check how `SignalExtractor` returns signals (list of objects vs dicts) and match it.
3. `client.py` `completion()` success path: the collector already extracts signals during
   `self.feedback.record(...)`. Obtain the extracted signals — if `record()` does not return
   them, extend it to return the created record (non-breaking: currently callers ignore the
   return value; verify with `grep -rn "feedback.record(" src/`). Then:

   ```python
   if self.config.implicit_feedback_enabled:
       iq = implicit_quality(record.signals)   # adapt to the record's real field name
       if iq is not None:
           self.router.predictor.update(
               messages, selected_model, actual_quality=iq, context=context
           )
   ```
   Mirror in `acompletion()`.
4. Guard against **double counting**: if the user later calls `record_outcome()` explicitly
   for the same request, that is a second (better-informed) update — acceptable for bandits;
   document this in the `implicit_feedback_enabled` field comment. Do NOT build dedup in
   this task.

**Tests** (`tests/test_implicit_learning.py`, litellm mocked):
- `test_refusal_triggers_negative_update` — fake response content
  `"I'm sorry, I can't help with that."` (must match a pattern in `_REFUSAL_PATTERNS` —
  read `signals.py` and use one verbatim) → predictor update called once with
  `actual_quality <= 0.05` (spy: `monkeypatch` `predictor.update`).
- `test_truncation_triggers_mild_negative` — `finish_reason="length"` → update with 0.40.
- `test_clean_response_no_implicit_update` — normal response → predictor.update NOT called.
- `test_implicit_disabled_no_update` — config flag False, refusal content → not called.
- `test_implicit_quality_pure_function` — direct unit tests of the mapping incl. empty list → None.

**Acceptance:** tests pass; README "Learning" section documents which signals update the
bandit and with what weights (copy the `IMPLICIT_QUALITY` table).

---

## Task P1.4 — Built-in LLM-as-judge evaluator  (size: L)

The only non-heuristic quality source, and the feature that makes the router self-improving
with ZERO integration work — the wedge NotDiamond can't copy on customer traffic.
`collector.set_quality_evaluator()` (feedback/collector.py, `def set_quality_evaluator`)
exists; nothing implements it.

**Files:** `src/routesmith/feedback/judge.py` (new), `src/routesmith/config.py`,
`src/routesmith/client.py`, `src/routesmith/cli/yaml_loader.py`, `tests/test_judge.py` (new)

**Spec — config (`config.py`):**

```python
@dataclass
class JudgeConfig:
    """LLM-as-judge sampling evaluator. Off by default (costs money)."""

    enabled: bool = False
    judge_model: str = "openai/gpt-4o-mini"  # any litellm model id; keep it cheap
    sample_rate: float = 0.05                # fraction of requests scored
    timeout_s: float = 20.0
```

Add `judge: JudgeConfig = field(default_factory=JudgeConfig)` to `RouteSmithConfig`.
(`judge_sample_rate` from P1.1 is superseded: keep the config field for YAML compat but have
it populate `judge.sample_rate`; note this in the field comment.)

**Spec — `src/routesmith/feedback/judge.py`:**

```python
"""LLM-as-judge quality evaluator."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a strict grader of AI assistant responses. Score how well the response "
    "answers the user's request: correctness, completeness, instruction-following. "
    'Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, "reason": "<max 15 words>"}'
)

_SCORE_RE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')


class LLMJudge:
    def __init__(self, model: str, timeout_s: float = 20.0) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def score(self, messages: list[dict], response_text: str) -> float | None:
        """Return quality in [0,1], or None on any failure (never raises)."""
        try:
            import litellm
            user_prompt = self._render(messages, response_text)
            result = litellm.completion(
                model=self.model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=80,
                timeout=self.timeout_s,
            )
            text = result.choices[0].message.content or ""
            return self._parse(text)
        except Exception as e:  # judge failures must never break serving
            logger.warning(f"Judge scoring failed: {e}")
            return None

    @staticmethod
    def _render(messages: list[dict], response_text: str) -> str:
        # Last user message is the request; truncate both to keep judge cost bounded.
        last_user = next((m.get("content", "") for m in reversed(messages)
                          if m.get("role") == "user"), "")
        return (f"USER REQUEST:\n{last_user[:4000]}\n\n"
                f"ASSISTANT RESPONSE:\n{response_text[:4000]}\n\nGrade it.")

    @staticmethod
    def _parse(text: str) -> float | None:
        try:
            score = float(json.loads(text)["score"])
        except Exception:
            m = _SCORE_RE.search(text)
            if not m:
                return None
            score = float(m.group(1))
        return max(0.0, min(1.0, score))
```

**Spec — wiring (`client.py`):**
1. `__init__`: if `config.judge.enabled`: `self._judge = LLMJudge(config.judge.judge_model,
   config.judge.timeout_s)`; also `self._judge_rng = random.Random(config.predictor.seed)`
   (deterministic sampling for tests); else `self._judge = None`.
2. `completion()` success path, after feedback recording:

   ```python
   if self._judge is not None and self._judge_rng.random() < self.config.judge.sample_rate:
       text = response.choices[0].message.content or ""
       jscore = self._judge.score(messages, text)
       if jscore is not None:
           self.record_outcome(request_id, score=jscore)  # reuses the whole explicit path
   ```
   Note: this makes the judge synchronous in-request. Acceptable for v0.3; a background
   queue is a Phase 4 concern. Document the added latency (`~1 judge call on sample_rate of
   requests`) in README.
3. `yaml_loader.py`: parse a `judge:` block (`enabled`, `model` → `judge_model`,
   `sample_rate`).

**Tests** (`tests/test_judge.py`, litellm mocked):
- `test_parse_clean_json` / `test_parse_json_in_prose` / `test_parse_garbage_returns_none` /
  `test_parse_clamps_range` (input `{"score": 1.7}` → 1.0).
- `test_judge_never_raises` — litellm mock raises → `score()` returns None.
- `test_judge_updates_predictor_when_sampled` — `sample_rate=1.0`, judge mock returns
  `'{"score": 0.9}'` → predictor update called with quality 0.9 (via record_outcome path).
- `test_judge_not_called_below_sample` — `sample_rate=0.0` → judge litellm call count 0.
- `test_judge_disabled_by_default` — `RouteSmithConfig().judge.enabled is False`.

**Acceptance:** all tests pass; README gains a "Self-improving mode" section with the YAML
snippet enabling the judge (5% sampling, gpt-4o-mini) and a cost note (≈ judge-model price ×
sample_rate per request).

---

## Task P1.5 — Convergence + E2E proof tests (the measurable goals)  (size: M)

These tests ARE goals G4 and G6. They pin the product promise in CI.

**Files:** `tests/test_convergence.py` (new), `tests/test_proxy_feedback_e2e.py` (new)

**Spec — `test_convergence.py`:**

```python
def test_lints_converges_on_synthetic_workload():
    """G4: >=80% optimal-arm selection within 150 feedback events."""
```
Construct `LinTSPredictor` over a 2-model registry (seed 42). Two query archetypes:
`MATH = [{"role":"user","content":"solve the integral of x^2 dx"}]`,
`CHAT = [{"role":"user","content":"write a short friendly greeting"}]`.
Ground truth: `gpt-4o` reward 0.95 on MATH / 0.55 on CHAT; `gpt-4o-mini` 0.45 on MATH /
0.90 on CHAT. Training loop, 150 iterations alternating archetypes: call
`predict(msgs, both_ids)`, take top-ranked as the selection, `update()` it with its ground-truth
reward. Evaluation loop, 100 iterations (50 per archetype): count selections matching the
optimal arm. `assert correct >= 80`.
Also `test_linucb_converges_on_synthetic_workload` — same harness, LinUCB, same threshold.

**Spec — `test_proxy_feedback_e2e.py` (G6):** end-to-end over handler + server-level JSON:
1. Build `RouteSmith` (litellm mocked) + `RequestHandler`.
2. `handle_completion` for a request → response dict has `routesmith_metadata.request_id`.
3. `handle_feedback` with that id, score 1.0 → 200.
4. Assert predictor `_total_updates` incremented AND a persisted explicit signal exists.
5. `test_e2e_feedback_shifts_routing` (slow-ish, still fast with mocks): 60 completion+feedback
   rounds rewarding gpt-4o-mini at 0.95 while gpt-4o gets 0.3 → over the next 20 routed
   requests (min_quality=0.0), gpt-4o-mini is selected ≥ 15 times.

**Acceptance:** all tests deterministic (fixed seeds), each file < 5 s runtime, green in CI.
