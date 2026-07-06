# Phase 5 — Ease of use + agentic platform coverage

Goal G1 (quickstart ≤ 6 commands) and G9 (8 platforms, runnable examples, CI smoke).
NotDiamond wins on "it's just a model id". We counter with: one CLI command to a running
router, native adapters for every major agent framework, and — uniquely — an
**Anthropic-native endpoint** so Claude Code / Claude Agent SDK route through RouteSmith by
setting one env var.

Task order: P5.1 → P5.2 (largest value) → P5.3 → P5.4 → P5.5 → P5.6.

Platform coverage matrix this phase completes:

| Platform | Mechanism | Task |
|---|---|---|
| LangChain / LangGraph | `ChatRouteSmith` (exists) + LangGraph example | P5.3 |
| CrewAI | existing adapter + example | P5.3 |
| AutoGen | existing adapter + example | P5.3 |
| DSPy | existing `RouteSmithLM` + example | P5.3 |
| OpenAI Agents SDK | proxy `base_url` + example | P5.4 |
| Pydantic AI | OpenAI-compatible provider → proxy + example | P5.4 |
| LlamaIndex | `OpenAILike` → proxy + example | P5.4 |
| Claude Code / Claude Agent SDK / Anthropic SDK | **NEW `/v1/messages` endpoint** + `ANTHROPIC_BASE_URL` | P5.2 |

---

## Task P5.1 — `examples/` directory + CI smoke harness  (size: M)

**Files:** `examples/` (new), `tests/test_examples_smoke.py` (new), `README.md`

**Spec:**
1. Every example is a standalone script ≤ 70 lines, top-of-file comment stating required
   extras + env vars, and a `main()` guarded by `if __name__ == "__main__":`. Examples read
   keys from env and NEVER hardcode secrets.
2. Create:
   - `examples/quickstart_python.py` — register 2 OpenRouter models, one `completion()`,
     print `stats` + `routing metadata`; then `record_outcome(score=1.0)`.
   - `examples/quickstart_proxy.sh` — `routesmith init --yes` (see P5.6), `routesmith serve &`,
     curl a completion, curl feedback with the returned request_id (jq), curl `/v1/stats`.
   - `examples/multi_agent_roles.py` — three `RouteContext(agent_role=...)` calls
     (planner/coder/summarizer) with role policies from P3.2.
3. `tests/test_examples_smoke.py` — for each `.py` example:
   `importlib.util.spec_from_file_location` + `module exec` with litellm mocked and
   `pytest.importorskip` for optional frameworks; assert `main` exists and, for the two
   quickstart examples, actually call `main()` (mocked) and assert no exception.
4. README: "Examples" table linking every file with a one-line description.

**Acceptance:** smoke test green with NO optional extras installed (skips reported, not
failures); green with extras when present.

---

## Task P5.2 — Anthropic-native proxy endpoint `POST /v1/messages`  (size: L)

**The differentiator:** any Anthropic-SDK tool — Claude Code, Claude Agent SDK, raw
`anthropic` clients — routes through RouteSmith by setting `ANTHROPIC_BASE_URL=http://localhost:9119`
(+ dummy `ANTHROPIC_API_KEY` when the proxy runs authless). No OpenRouter equivalent exists
for BYO-routing in Anthropic-native tools.

**Files:** `src/routesmith/proxy/anthropic_compat.py` (new), `src/routesmith/proxy/server.py`
(route registration), `tests/test_anthropic_endpoint.py` (new)

**Spec — request translation (`anthropic_compat.py`):**

```python
def anthropic_to_internal(data: dict) -> tuple[list[dict], dict]:
    """Anthropic Messages API request -> (openai_style_messages, kwargs).

    Handles:
      system: str | [{"type":"text","text":...}]  -> messages[0] = {"role":"system",...}
      messages[*].content: str  -> passthrough
      messages[*].content: [blocks] -> concatenate text blocks with "\n";
          non-text blocks (images, tool_use, tool_result): v1 REJECTS with a clear
          error listing the unsupported block type (tool support is a follow-up task).
      max_tokens (required by Anthropic)      -> kwargs["max_tokens"]
      temperature / top_p / stop_sequences    -> kwargs (stop_sequences -> stop)
      stream                                  -> handled by caller
      model                                   -> "auto"-routed when it is NOT a registered
          model id (Claude Code sends real Anthropic ids like "claude-sonnet-4-..."; when
          unknown to the registry, treat as "auto" so RouteSmith decides — THIS IS THE POINT).
    Raises ValueError with a precise message on malformed input.
    """
```

```python
def internal_to_anthropic(response, request_model: str) -> dict:
    """litellm/OpenAI-style response -> Anthropic Messages API response dict:

    {
      "id": "msg_<request_id>", "type": "message", "role": "assistant",
      "model": <actual routed model id>,
      "content": [{"type": "text", "text": <content>}],
      "stop_reason": map(finish_reason),   # stop->end_turn, length->max_tokens,
                                           # tool_calls->tool_use, else->end_turn
      "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}
    }
    NOTE: integrations/anthropic.py already contains a module-level stop-reason map —
    grep "end_turn" there and REUSE it (import, don't duplicate).
    """
```

**Streaming:** Anthropic SSE event sequence, minimum viable set:
`message_start` → repeated `content_block_delta` (`{"type":"text_delta","text": ...}`) →
`message_delta` (with `stop_reason`) → `message_stop`, each as
`event: <name>\ndata: <json>\n\n`. Emit `content_block_start`/`content_block_stop` around the
deltas for spec compliance. Read Anthropic's public streaming docs section if any field is
in doubt; keep to text-only blocks in v1.

**Server:** register `POST /v1/messages`. Auth from P4.1 applies; additionally accept the
Anthropic convention `x-api-key` header as the bearer token equivalent. Response includes
`routesmith_metadata` under a top-level `"routesmith_metadata"` key (Anthropic clients
ignore unknown fields) so feedback via `/v1/feedback` still works.

**Tests** (`tests/test_anthropic_endpoint.py`, litellm mocked):
- `test_basic_translation_roundtrip` — Anthropic-shape request in → 200; response has
  `type=="message"`, `content[0].text` set, `usage.input_tokens==10`.
- `test_system_prompt_translated` / `test_content_blocks_concatenated` /
  `test_image_block_rejected_clearly` (400, message names `"image"`).
- `test_missing_max_tokens_400` (Anthropic requires it).
- `test_unknown_model_id_gets_routed` — `model="claude-sonnet-4-20250514"` not in registry →
  response `model` is one of the registered models; metadata reason `"routed"`.
- `test_registered_model_id_respected` — explicitly registered id → that model used.
- `test_stop_reason_mapping` — finish_reason length → `max_tokens`.
- `test_streaming_event_sequence` — collected SSE events in exact order
  `message_start, content_block_start, content_block_delta+, content_block_stop,
  message_delta, message_stop`.
- `test_x_api_key_header_accepted_for_auth`.

**Docs:** README section "Use with Claude Code / Anthropic SDK":

```bash
routesmith serve &
export ANTHROPIC_BASE_URL=http://localhost:9119
export ANTHROPIC_API_KEY=dummy   # or a configured RouteSmith key
# Anthropic-SDK apps now route through RouteSmith
```

**Acceptance:** all tests pass; a manual smoke with the real `anthropic` Python SDK
(`client.messages.create`) against the local proxy is documented in the PR description.

---

## Task P5.3 — Agent-framework examples: LangGraph, CrewAI, AutoGen, DSPy  (size: M)

Adapters exist (`integrations/langchain.py` with `agent_role` support, `crewai.py`,
`autogen.py`, `dspy.py`); what's missing is copy-paste proof for agent GRAPHS with per-role
routing.

**Files:** `examples/langgraph_agents.py`, `examples/crewai_crew.py`,
`examples/autogen_pair.py`, `examples/dspy_pipeline.py`, docs

**Spec:**
1. `langgraph_agents.py` — 2-node LangGraph (`planner` → `executor`), each node with its own
   `ChatRouteSmith(agent_role=..., track_conversation=True)` over ONE shared RouteSmith
   instance; after the run, print `rs.stats` and per-role model recommendations
   (`recommend_model_for_agent("planner")`).
2. `crewai_crew.py` — 2-agent crew via the existing
   `routesmith_crewai_chat_model()` against the proxy; header-based roles
   (`X-RouteSmith-Agent-Role`) — grep `crewai.py` for the extra-headers mechanism; if none
   exists, add `default_headers` passthrough to the adapter as part of this task.
3. `autogen_pair.py` / `dspy_pipeline.py` — same pattern with the existing adapters.
4. Each example paired with an entry in `tests/test_examples_smoke.py`
   (importorskip the framework).

**Acceptance:** smoke green; docs/examples index updated; each example ≤ 70 lines.

---

## Task P5.4 — OpenAI Agents SDK, Pydantic AI, LlamaIndex examples  (size: S)

All three speak OpenAI-compatible `base_url` — no adapter code needed, only proof + docs.

**Files:** `examples/openai_agents_sdk.py`, `examples/pydantic_ai_agent.py`,
`examples/llamaindex_engine.py`, smoke-test entries

**Spec:** each: point the framework's OpenAI client at `http://localhost:9119/v1` with
`api_key="dummy"`, `model="routesmith/auto"`; one trivial agent/task; print which model
actually served (`routesmith_metadata` or `/v1/stats`). Include per-conversation stickiness
header where the framework supports default headers.

**Acceptance:** smoke entries green (importorskip); README matrix rows checked off.

---

## Task P5.5 — `routesmith quickstart` CLI (G1)  (size: M)

**Files:** `src/routesmith/cli/quickstart.py` (new), `src/routesmith/cli/main.py` (register
subcommand), `scripts/verify_quickstart.sh` (new), `tests/test_cli_quickstart.py` (new)

**Spec:**
1. `routesmith quickstart [--port 9119] [--yes]`:
   1. Detect provider keys in env (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`,
      `ANTHROPIC_API_KEY`); if none, print the three export lines and exit 1.
   2. Generate `routesmith.yaml` (reuse `cli/init.py` logic; `--yes` skips prompts, picks
      the detected provider, registers a sensible default pool — for OpenRouter: top-5 by
      the P2.1 prior table that the key can serve).
   3. Start the server (foreground), first printing a copy-paste block: curl completion,
      curl feedback, Python `OpenAI(base_url=...)` snippet, and the `ANTHROPIC_BASE_URL`
      snippet from P5.2.
2. `scripts/verify_quickstart.sh` — THE G1 measurement. From a clean checkout:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   # G1: <= 6 commands from clean venv to a routed request.
   python -m venv .qs-venv                     # 1
   . .qs-venv/bin/activate                     # 2
   pip install -e ".[proxy]" -q                # 3
   export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-test-key-mocked}
   routesmith quickstart --yes --port 9123 &   # 4
   sleep 3
   curl -sf localhost:9123/health              # 5
   curl -sf localhost:9123/v1/chat/completions \
     -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}' \
     | grep -q routesmith_metadata             # 6  (with RS_MOCK_LITELLM=1, see below)
   kill %1
   ```
   To keep CI key-free, support env `RS_MOCK_LITELLM=1` which makes the client return a
   canned response instead of calling litellm (implement as a small check in
   `client.completion` gated on the env var, clearly marked "testing only").
3. `tests/test_cli_quickstart.py` — `--yes` writes a valid YAML (parse it back), exits 1
   with helpful text when no keys in env.

**Acceptance:** `bash scripts/verify_quickstart.sh` exits 0 in CI (mocked mode) — G1 MET.

---

## Task P5.6 — Packaging & docs polish  (size: S)

**Files:** `pyproject.toml`, `README.md`

**Spec:**
1. Extras: add `integrations = []` meta-extra listing nothing itself but documented as the
   umbrella (`langchain-core`, `dspy`, etc. stay user-installed — adapters degrade with clear
   ImportError messages, which already exist); add `all = [proxy + cache + embeddings + redis
   deps]` (enumerate explicitly; extras cannot reference each other in all build backends —
   copy the lists).
2. README top: 30-second pitch table "RouteSmith vs OpenRouter Auto Router" —
   rows: learns from your traffic (✅/❌), self-hosted (✅/❌), custom model pool (✅/❌),
   custom rewards & per-role policies (✅/❌), decision audit (✅/❌), budget caps (✅/❌),
   conversation stickiness (✅/✅), pre-trained cold start (✅ after P2.1/✅).
   Keep it factual; link OpenRouter's docs for their column.
3. Verify every README code block against the shipped API one final time
   (`scripts/check_claims.sh` extended with any new greps).

**Acceptance:** `pip install routesmith[all]` resolves in a fresh venv; check_claims green;
G9 matrix complete in README.
