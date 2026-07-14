#!/usr/bin/env bash
# Quickstart: RouteSmith proxy via CLI.
#
# Usage:
#   export OPENROUTER_API_KEY=sk-or-...
#   bash examples/quickstart_proxy.sh
#
# Requires: routesmith-llm[proxy]

set -euo pipefail

PORT=${PORT:-9119}

echo "=== Generating config ==="
routesmith init --output /tmp/routesmith.yaml --force 2>/dev/null || {
    echo "No API key detected. Creating minimal config..."
    cat > /tmp/routesmith.yaml <<'YAML'
models:
  - model_id: openai/gpt-4o-mini
    cost_per_1k_input: 0.15
    cost_per_1k_output: 0.60
    quality_score: 0.85
  - model_id: openai/gpt-4o
    cost_per_1k_input: 2.50
    cost_per_1k_output: 10.00
    quality_score: 0.95
YAML
}

echo "=== Starting proxy ==="
routesmith serve --config /tmp/routesmith.yaml --port "$PORT" &
PID=$!
sleep 3

echo "=== Health check ==="
curl -sf "http://localhost:$PORT/health"

echo "=== Completion ==="
RESP=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
    -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"max_tokens":50}')
echo "$RESP" | python3 -m json.tool
REQUEST_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('routesmith_metadata',{}).get('request_id',''))")

if [ -n "$REQUEST_ID" ]; then
    echo "=== Feedback ==="
    curl -sf "http://localhost:$PORT/v1/feedback" \
        -H "Content-Type: application/json" \
        -d "{\"request_id\":\"$REQUEST_ID\",\"score\":1.0}"
    echo
fi

echo "=== Stats ==="
curl -sf "http://localhost:$PORT/v1/stats" | python3 -m json.tool

kill "$PID" 2>/dev/null || true
echo "=== Done ==="
