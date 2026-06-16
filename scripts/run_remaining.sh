#!/usr/bin/env bash
# scripts/run_remaining.sh
# Runs everything AFTER the static phase (which is already running separately).
# Recovery: each strategy saves after every query — safe to kill and re-run.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

LOG_DIR="benchmark/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_remaining_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== RouteSmith: Remaining Experiments ==="
log "Phases: routellm, ts_cat, linucb, lints, exp2, ablations"
log "Recovery: kill anytime and re-run — resumes from last saved query"
log "Log: $LOG"
log ""

# Wait for static phase to finish (avoid writing same result files simultaneously)
log "Waiting for static phase to complete..."
for i in $(seq 1 120); do
    # Static phase writes 6 result files when done:
    # static_strong_mmlu, static_weak_mmlu, random_mmlu (3 MMLU)
    # static_strong_gsm8k, static_weak_gsm8k, random_gsm8k (3 GSM8K)
    COUNT=$(ls benchmark/results/static_*_results.json benchmark/results/random_*_results.json 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 6 ]; then
        # Also check query counts in the files
        MMLU_COUNT=$(.venv/bin/python3 -c "
import json, glob
files = glob.glob('benchmark/results/static_strong_mmlu_results.json')
if files:
    print(len(json.load(open(files[0]))))
else:
    print(0)
" 2>/dev/null)
        if [ "${MMLU_COUNT:-0}" -ge 600 ]; then
            log "Static phase complete (${MMLU_COUNT} MMLU queries done). Proceeding."
            break
        fi
    fi
    if [ $i -eq 1 ]; then
        log "  (checking every 60s, up to 120 min)"
    fi
    sleep 60
done

log ""
log "=== Phase: routellm (3 thresholds × MMLU + GSM8K) ==="
.venv/bin/python3 -m benchmark.experiments.exp1_binary routellm 2>&1 | tee -a "$LOG"
log "routellm done."

log ""
log "=== Phase: ts_cat (5 seeds × MMLU + GSM8K) ==="
.venv/bin/python3 -m benchmark.experiments.exp1_binary ts_cat 2>&1 | tee -a "$LOG"
log "ts_cat done."

log ""
log "=== Phase: linucb (5 seeds × MMLU + GSM8K) ==="
.venv/bin/python3 -m benchmark.experiments.exp1_binary linucb 2>&1 | tee -a "$LOG"
log "linucb done."

log ""
log "=== Phase: lints (5 seeds × MMLU + GSM8K) ==="
.venv/bin/python3 -m benchmark.experiments.exp1_binary lints 2>&1 | tee -a "$LOG"
log "lints done."

log ""
log "=== Experiment 1 Summary ==="
.venv/bin/python3 -m benchmark.resume 2>&1 | tee -a "$LOG" || true

log ""
log "=== Experiment 2: 5-arm multi-model routing ==="
.venv/bin/python3 -m benchmark.experiments.exp2_multimodel 2>&1 | tee -a "$LOG"
log "exp2 done."

log ""
log "=== Ablation Experiments ==="
.venv/bin/python3 -m benchmark.experiments.ablations 2>&1 | tee -a "$LOG"
log "ablations done."

log ""
log "=== ALL EXPERIMENTS COMPLETE ==="
log "Results in benchmark/results/"
log "Next: make figures && bash paper/build.sh"
