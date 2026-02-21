#!/usr/bin/env bash
#
# OpenClaw + RouteSmith Integration Tests
#
# Phase A: Proxy-only curl tests (no OpenClaw needed)
# Phase B: OpenClaw manual instructions (printed at end)
#
# Usage:
#   1. Start the proxy:  routesmith serve --port 9119
#   2. Run this script:  bash tests/manual/test_openclaw_integration.sh
#
# Requirements: curl, jq

set -euo pipefail

BASE_URL="${ROUTESMITH_URL:-http://localhost:9119}"
PASS=0
FAIL=0

# ── Colors ──────────────────────────────────────────────────────────────────

green() { printf '\033[32m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }
bold()  { printf '\033[1m%s\033[0m' "$*"; }

pass() { PASS=$((PASS + 1)); echo "  $(green PASSED) — $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  $(red FAILED) — $1${2:+ ($2)}"; }

# ── Helpers ─────────────────────────────────────────────────────────────────

# curl wrapper that captures both body and HTTP status code
# Usage: http_get /path   or   http_post /path '{"json":"body"}'
http_get() {
  local path="$1"
  curl -s -w '\n%{http_code}' "${BASE_URL}${path}" 2>/dev/null
}

http_post() {
  local path="$1" body="$2"
  curl -s -w '\n%{http_code}' \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer dummy" \
    "${BASE_URL}${path}" \
    -d "${body}" 2>/dev/null
}

# Split curl output into body + status
split_response() {
  local response="$1"
  BODY=$(echo "$response" | sed '$d')
  STATUS=$(echo "$response" | tail -1)
}

# ════════════════════════════════════════════════════════════════════════════
#  Phase A — Proxy-only curl tests
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "$(bold '═══ Phase A: RouteSmith Proxy Tests ═══')"
echo "Target: ${BASE_URL}"
echo ""

# ── Test 1: Health check ────────────────────────────────────────────────────

echo "$(bold '1. Health check')"
split_response "$(http_get /health)"
if [ "$STATUS" = "200" ] && echo "$BODY" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
  pass "GET /health returns healthy"
else
  fail "GET /health" "status=$STATUS body=$BODY"
fi

# ── Test 2: List models ─────────────────────────────────────────────────────

echo "$(bold '2. List models')"
split_response "$(http_get /v1/models)"
if [ "$STATUS" = "200" ] && echo "$BODY" | jq -e '.data[] | select(.id == "routesmith/auto")' > /dev/null 2>&1; then
  pass "GET /v1/models contains routesmith/auto"
else
  fail "GET /v1/models" "status=$STATUS"
fi

# ── Test 3: Non-streaming completion ─────────────────────────────────────────

echo "$(bold '3. Non-streaming completion')"
split_response "$(http_post /v1/chat/completions '{
  "model": "auto",
  "messages": [{"role": "user", "content": "Say hello in one word."}],
  "max_tokens": 32
}')"
CONTENT=$(echo "$BODY" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
if [ "$STATUS" = "200" ] && [ -n "$CONTENT" ]; then
  pass "Non-streaming completion returned content"
else
  fail "Non-streaming completion" "status=$STATUS content=$CONTENT"
fi

# ── Test 4: Streaming completion ─────────────────────────────────────────────

echo "$(bold '4. Streaming completion')"
STREAM_OUTPUT=$(curl -s -N \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  "${BASE_URL}/v1/chat/completions" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Say hi."}],
    "stream": true,
    "max_tokens": 16
  }' 2>/dev/null || true)

if echo "$STREAM_OUTPUT" | grep -q 'data: \[DONE\]'; then
  pass "Streaming completion ends with data: [DONE]"
else
  fail "Streaming completion" "Missing [DONE] marker"
fi

# ── Test 5: Explicit model bypass ────────────────────────────────────────────

echo "$(bold '5. Explicit model bypass')"
split_response "$(http_post /v1/chat/completions '{
  "model": "groq/llama-3.3-70b-versatile",
  "messages": [{"role": "user", "content": "Reply with the word ok."}],
  "max_tokens": 16
}')"
USED_MODEL=$(echo "$BODY" | jq -r '.model // empty' 2>/dev/null)
if [ "$STATUS" = "200" ] && [ -n "$USED_MODEL" ]; then
  pass "Explicit model bypass responded (model=$USED_MODEL)"
else
  fail "Explicit model bypass" "status=$STATUS"
fi

# ── Test 6: RouteSmith extensions ────────────────────────────────────────────

echo "$(bold '6. RouteSmith extensions')"
split_response "$(http_post /v1/chat/completions '{
  "model": "auto",
  "messages": [{"role": "user", "content": "Say ok."}],
  "routesmith_min_quality": 0.9,
  "max_tokens": 16
}')"
if [ "$STATUS" = "200" ]; then
  pass "routesmith_min_quality accepted without error"
else
  fail "RouteSmith extensions" "status=$STATUS body=$BODY"
fi

# ── Test 7: Stats after requests ─────────────────────────────────────────────

echo "$(bold '7. Stats after requests')"
split_response "$(http_get /v1/stats)"
REQ_COUNT=$(echo "$BODY" | jq -r '.request_count // 0' 2>/dev/null)
if [ "$STATUS" = "200" ] && [ "$REQ_COUNT" -ge 3 ] 2>/dev/null; then
  pass "Stats show request_count=$REQ_COUNT (>= 3)"
else
  fail "Stats check" "status=$STATUS request_count=$REQ_COUNT"
fi

# ── Test 8: Error handling ───────────────────────────────────────────────────

echo "$(bold '8. Error handling')"
split_response "$(http_post /v1/chat/completions '{
  "model": "auto"
}')"
HAS_ERROR=$(echo "$BODY" | jq -e '.error' > /dev/null 2>&1 && echo "yes" || echo "no")
if [ "$STATUS" = "400" ] && [ "$HAS_ERROR" = "yes" ]; then
  pass "Malformed request returns 400 with error object"
else
  fail "Error handling" "status=$STATUS has_error=$HAS_ERROR"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "$(bold '═══ Results ═══')"
echo "  $(green "$PASS passed"), $([ "$FAIL" -gt 0 ] && red "$FAIL failed" || echo "$FAIL failed")"
echo ""

# ════════════════════════════════════════════════════════════════════════════
#  Phase B — OpenClaw Manual Instructions
# ════════════════════════════════════════════════════════════════════════════

echo "$(bold '═══ Phase B: OpenClaw Integration Steps ═══')"
echo ""
echo "Follow these steps on a machine with OpenClaw installed:"
echo ""
echo "1. Copy the provider config to your OpenClaw machine:"
echo "   scp tests/manual/openclaw_provider_config.json openclaw-host:~/"
echo ""
echo "2. Merge into your OpenClaw config:"
echo "   jq -s '.[0] * .[1]' ~/.openclaw/openclaw.json ~/openclaw_provider_config.json > /tmp/merged.json"
echo "   cp ~/.openclaw/openclaw.json ~/.openclaw/openclaw.json.bak"
echo "   mv /tmp/merged.json ~/.openclaw/openclaw.json"
echo ""
echo "3. Apply the config:"
echo "   openclaw gateway config.apply --file ~/.openclaw/openclaw.json"
echo ""
echo "4. Verify the model appears:"
echo "   Run /models in OpenClaw — look for 'routesmith/auto'"
echo ""
echo "5. Switch to RouteSmith:"
echo "   /model routesmith"
echo ""
echo "6. Send test messages and verify responses come back."
echo ""
echo "7. Check proxy stats from the RouteSmith host:"
echo "   curl http://localhost:9119/v1/stats | jq"
echo ""

# Exit with failure if any tests failed
[ "$FAIL" -eq 0 ] || exit 1
