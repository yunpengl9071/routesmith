# RouteSmith Testing Sandbox

This guide helps you test RouteSmith's proxy server and routing with real API calls.

## Getting API Keys

You need at least ONE API key to test. Here are your options:

### Option 1: Groq (Recommended - FREE)
- **Sign up**: https://console.groq.com/
- **Get key**: Dashboard → API Keys → Create
- **Free tier**: 30 requests/minute, no credit card required
- **Set env**: `export GROQ_API_KEY=gsk_...`

### Option 2: OpenAI (Pay-per-use)
- **Sign up**: https://platform.openai.com/
- **Get key**: API Keys → Create new secret key
- **Cost**: ~$0.00015/1K tokens for gpt-4o-mini
- **Set env**: `export OPENAI_API_KEY=sk-...`

### Option 3: Anthropic (Pay-per-use)
- **Sign up**: https://console.anthropic.com/
- **Get key**: API Keys → Create Key
- **Cost**: ~$0.003/1K tokens for Claude Haiku
- **Set env**: `export ANTHROPIC_API_KEY=sk-ant-...`

### Option 4: Google Gemini (FREE tier)
- **Sign up**: https://makersuite.google.com/
- **Get key**: Get API Key
- **Free tier**: 60 requests/minute
- **Set env**: `export GOOGLE_API_KEY=...`

---

## Quick Start Setup

```bash
# 1. Navigate to project
cd /Users/yliulupo/Apps/routesmith

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install with proxy dependencies
uv pip install -e ".[dev,proxy]"

# 4. Set your API key (pick one)
export GROQ_API_KEY=gsk_your_key_here
# OR
export OPENAI_API_KEY=sk-your_key_here
# OR
export ANTHROPIC_API_KEY=sk-ant-your_key_here

# 5. Create config file
cp routesmith.yaml.example routesmith.yaml
```

---

## Test 1: Basic Proxy Server

### Start the server:
```bash
routesmith serve --port 9119 --config routesmith.yaml
```

Expected output:
```
RouteSmith proxy server starting on http://127.0.0.1:9119
Registered models: 6
Endpoints:
  POST http://127.0.0.1:9119/v1/chat/completions
  GET  http://127.0.0.1:9119/v1/models
  GET  http://127.0.0.1:9119/v1/stats
  GET  http://127.0.0.1:9119/health
Press Ctrl+C to stop
```

### Test endpoints (in another terminal):

```bash
# Health check
curl http://localhost:9119/health
# Expected: {"status": "healthy", "registered_models": 6}

# List models
curl http://localhost:9119/v1/models
# Expected: {"object": "list", "data": [...]}

# Check stats
curl http://localhost:9119/v1/stats
# Expected: {"request_count": 0, "total_cost_usd": 0, ...}
```

---

## Test 2: Chat Completion with Routing

### Using auto-routing (RouteSmith picks best model):
```bash
curl -X POST http://localhost:9119/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}]
  }'
```

Expected response includes:
```json
{
  "id": "chatcmpl-...",
  "model": "gpt-4o-mini",
  "choices": [{"message": {"content": "4"}}],
  "routesmith_metadata": {
    "model_selected": "gpt-4o-mini",
    "routing_reason": "cheapest model with quality >= 0.8",
    "estimated_cost_usd": 0.000012,
    "cost_savings_usd": 0.000088
  }
}
```

### Using specific model (bypass routing):
```bash
curl -X POST http://localhost:9119/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### With RouteSmith parameters:
```bash
curl -X POST http://localhost:9119/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "routesmith_min_quality": 0.9,
    "routesmith_max_cost": 0.01
  }'
```

---

## Test 3: Check Statistics

After running some requests:
```bash
routesmith stats
```

Expected output:
```
╭─────────────────────────────────────────────╮
│         RouteSmith Cost Report              │
├─────────────────────────────────────────────┤
│  Requests:                       5          │
│  Actual Cost:        $     0.0001           │
│  Without Routing:    $     0.0010           │
│  You Saved:          $0.0009 (90.0%)        │
├─────────────────────────────────────────────┤
│  Registered Models:              6          │
│  Feedback Samples:               1          │
╰─────────────────────────────────────────────╯
```

---

## Test 4: Streaming

```bash
curl -X POST http://localhost:9119/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

Expected: SSE stream with `data: {...}` chunks ending with `data: [DONE]`

---

## Test 5: Python Client

Create `test_sandbox.py`:
```python
#!/usr/bin/env python3
"""Test RouteSmith proxy with Python."""

import httpx

BASE_URL = "http://localhost:9119"

# Test completion with auto-routing
response = httpx.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "auto",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    },
    timeout=30.0,
)

data = response.json()
print(f"Model used: {data.get('model')}")
print(f"Response: {data['choices'][0]['message']['content']}")

if "routesmith_metadata" in data:
    meta = data["routesmith_metadata"]
    print(f"Cost: ${meta['estimated_cost_usd']:.6f}")
    print(f"Saved: ${meta['cost_savings_usd']:.6f}")

# Check stats
stats = httpx.get(f"{BASE_URL}/v1/stats").json()
print(f"\nTotal requests: {stats['request_count']}")
print(f"Total cost: ${stats['total_cost_usd']:.6f}")
print(f"Savings: {stats['savings_percent']:.1f}%")
```

Run:
```bash
python test_sandbox.py
```

---

## Test 6: Groq-Only Config (Free Testing)

If you only have a Groq API key, use this minimal config:

Create `routesmith-groq.yaml`:
```yaml
routing:
  strategy: direct

budget:
  quality_threshold: 0.7

models:
  - id: groq/llama-3.1-70b-versatile
    cost_per_1k_input: 0.00059
    cost_per_1k_output: 0.00079
    quality_score: 0.88

  - id: groq/llama-3.1-8b-instant
    cost_per_1k_input: 0.00005
    cost_per_1k_output: 0.00008
    quality_score: 0.75
```

Run:
```bash
export GROQ_API_KEY=gsk_your_key
routesmith serve --config routesmith-groq.yaml
```

---

## Test 7: Unit Tests (No API Keys)

Run mocked tests:
```bash
pytest tests/ -v
```

---

## OpenClaw Integration (Future)

Once OpenClaw supports custom providers, configure:

`~/.openclaw/openclaw.json`:
```json
{
  "models": {
    "providers": {
      "routesmith": {
        "baseUrl": "http://localhost:9119/v1",
        "api": "openai-completions",
        "apiKey": "dummy"
      }
    }
  },
  "agent": {
    "model": "routesmith/auto"
  }
}
```

---

## Troubleshooting

### "No models registered"
- Check your `routesmith.yaml` exists and has models defined
- Verify YAML syntax is correct

### "Connection refused"
- Make sure server is running: `routesmith serve`
- Check port is correct (default 9119)

### API errors
- Verify API key is set: `echo $OPENAI_API_KEY`
- Check key has credits/quota

### "litellm error"
- Model name might be wrong
- Check litellm docs for correct model IDs

---

## Cost Estimates

| Model | Input $/1K | Output $/1K | Quality |
|-------|------------|-------------|---------|
| gpt-4o | $0.005 | $0.015 | 0.95 |
| gpt-4o-mini | $0.00015 | $0.0006 | 0.85 |
| claude-3-5-sonnet | $0.003 | $0.015 | 0.93 |
| claude-3-5-haiku | $0.001 | $0.005 | 0.80 |
| groq/llama-3.1-70b | $0.00059 | $0.00079 | 0.88 |
| groq/llama-3.1-8b | $0.00005 | $0.00008 | 0.75 |

A simple "What is 2+2?" query with gpt-4o-mini costs ~$0.00001

---

## Next Steps

1. Run several queries and watch the stats accumulate
2. Try different `routesmith_min_quality` values to see routing change
3. Compare costs between models for the same query
4. Check `rs.stats` to see total savings
