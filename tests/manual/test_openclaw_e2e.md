# OpenClaw + RouteSmith End-to-End Test Checklist

## Prerequisites

- [ ] RouteSmith installed (`pip install -e ".[dev]"` or `uv pip install -e ".[dev]"`)
- [ ] API keys configured in `.env` (e.g. `GROQ_API_KEY`, `OPENAI_API_KEY`)
- [ ] `routesmith.yaml` has at least one model registered
- [ ] `curl` and `jq` available on the proxy host
- [ ] OpenClaw installed on the target device

## Step 1: Start the Proxy

```bash
# From the routesmith repo root
routesmith serve --port 9119
```

Verify it's running:
```bash
curl http://localhost:9119/health | jq
# Expected: {"status": "healthy", "registered_models": <N>}
```

## Step 2: Run Automated Proxy Tests

```bash
bash tests/manual/test_openclaw_integration.sh
```

All 8 tests should show **PASSED**:

| # | Test | What it checks |
|---|------|---------------|
| 1 | Health check | `/health` returns `"healthy"` |
| 2 | List models | `/v1/models` includes `routesmith/auto` |
| 3 | Non-streaming completion | `/v1/chat/completions` returns content |
| 4 | Streaming completion | SSE stream ends with `data: [DONE]` |
| 5 | Explicit model bypass | Specific model name is forwarded |
| 6 | RouteSmith extensions | `routesmith_min_quality` accepted |
| 7 | Stats | `request_count >= 3` after tests |
| 8 | Error handling | Missing `messages` returns 400 |

## Step 3: Configure OpenClaw

1. Copy `tests/manual/openclaw_provider_config.json` to the OpenClaw machine
2. Merge it into `~/.openclaw/openclaw.json`:

```bash
# Backup existing config
cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.bak

# Merge (requires jq)
jq -s '.[0] * .[1]' \
  ~/.openclaw/openclaw.json \
  openclaw_provider_config.json \
  > /tmp/merged.json

mv /tmp/merged.json ~/.openclaw/openclaw.json
```

3. Apply the config:

```bash
openclaw gateway config.apply --file ~/.openclaw/openclaw.json
```

## Step 4: Verify in OpenClaw

- [ ] Run `/models` in OpenClaw — `routesmith/auto` should appear
- [ ] Run `/model routesmith` to switch to RouteSmith
- [ ] Send a simple message (e.g. "What is 2+2?") and verify a response comes back
- [ ] Send a longer message and verify streaming works (tokens appear incrementally)
- [ ] Check proxy stats: `curl http://<proxy-host>:9119/v1/stats | jq`

## Step 5: Advanced Tests

- [ ] Send a message requiring code generation — verify formatting is preserved
- [ ] Test with `routesmith_min_quality: 0.9` in the request — should route to a higher-quality model
- [ ] Restart the proxy and verify OpenClaw reconnects

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Connection refused | Proxy not running or wrong port | Check `routesmith serve` is running; verify port matches config `baseUrl` |
| No models in `/models` | No models registered | Ensure `routesmith.yaml` has models or register them programmatically |
| 500 errors on completion | Missing or invalid API keys | Check `.env` for the correct provider API key |
| Streaming hangs | Firewall or proxy buffering | Try disabling response buffering; check network path |
| OpenClaw doesn't show model | Config not merged/applied | Re-run `config.apply`; check JSON syntax in merged config |
| Model mismatch | Wrong model ID in config | Ensure OpenClaw config uses `"id": "auto"` (maps to `routesmith/auto` internally) |
| Slow responses | Model routing overhead | Check `routesmith serve` logs for routing decisions; verify network latency to LLM provider |
