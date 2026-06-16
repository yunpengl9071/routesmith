#!/usr/bin/env bash
# scripts/run_experiments.sh — Run all benchmark experiments with recovery
#
# Recovery: Each strategy saves results incrementally after every query.
# Re-running this script skips already-completed queries automatically.
# Kill at any time and re-run to resume.
#
# Usage:
#   bash scripts/run_experiments.sh [phase]
#
# Phases (run all if not specified):
#   exp1_static    — Static-Strong, Static-Weak, Random
#   exp1_routellm  — RouteLLM-SW (3 thresholds)
#   exp1_ts_cat    — TS-Cat (5 seeds × MMLU + GSM8K)
#   exp1_linucb    — LinUCB-27d (5 seeds × MMLU + GSM8K)
#   exp1_lints     — LinTS-27d (5 seeds × MMLU + GSM8K)
#   exp2           — 5-arm multi-model LinTS (3 seeds × MMLU)
#   ablations      — Feature dims, warm-start, beta sensitivity
#   status         — Print current results summary only

set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

LOG_DIR="benchmark/logs"
mkdir -p "$LOG_DIR"
PHASE="${1:-all}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${PHASE}_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_phase() {
    local phase="$1"
    log "Starting phase: $phase"
    .venv/bin/python3 -m benchmark.experiments.exp1_binary "$phase" 2>&1 | tee -a "$LOG_FILE"
    log "Phase $phase complete."
}

log "=== RouteSmith Experiment Runner ==="
log "Phase: $PHASE"
log "Log: $LOG_FILE"
log "Recovery: experiments resume from last saved query automatically"
log ""

case "$PHASE" in
    exp1_static)
        run_phase "static"
        ;;
    exp1_routellm)
        run_phase "routellm"
        ;;
    exp1_ts_cat)
        run_phase "ts_cat"
        ;;
    exp1_linucb)
        run_phase "linucb"
        ;;
    exp1_lints)
        run_phase "lints"
        ;;
    exp2)
        log "Starting Experiment 2: 5-arm multi-model routing..."
        .venv/bin/python3 -m benchmark.experiments.exp2_multimodel 2>&1 | tee -a "$LOG_FILE"
        log "Experiment 2 complete."
        ;;
    ablations)
        log "Starting ablation experiments..."
        .venv/bin/python3 -m benchmark.experiments.ablations 2>&1 | tee -a "$LOG_FILE"
        log "Ablations complete."
        ;;
    status)
        log "Current experiment status:"
        .venv/bin/python3 -m benchmark.resume 2>&1 | tee -a "$LOG_FILE"
        ;;
    all)
        log "Running ALL experiments in sequence."
        log "Estimated time: 3-6 hours. Kill and re-run to resume from any point."
        log ""

        for phase in static routellm ts_cat linucb lints; do
            run_phase "$phase"
            log ""
        done

        log "=== Exp 1 Summary ==="
        .venv/bin/python3 -m benchmark.experiments.exp1_binary status 2>&1 | tee -a "$LOG_FILE" || true

        log ""
        log "Starting Experiment 2: 5-arm multi-model routing..."
        .venv/bin/python3 -m benchmark.experiments.exp2_multimodel 2>&1 | tee -a "$LOG_FILE"
        log "Experiment 2 complete."

        log ""
        log "Starting ablation experiments..."
        .venv/bin/python3 -m benchmark.experiments.ablations 2>&1 | tee -a "$LOG_FILE"
        log "Ablations complete."

        log ""
        log "=== ALL EXPERIMENTS COMPLETE ==="
        log "Results in benchmark/results/"
        log "Run: make figures  to regenerate all figures"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: exp1_static, exp1_routellm, exp1_ts_cat, exp1_linucb, exp1_lints, exp2, ablations, status, all"
        exit 1
        ;;
esac

log "Done. Log saved to $LOG_FILE"
