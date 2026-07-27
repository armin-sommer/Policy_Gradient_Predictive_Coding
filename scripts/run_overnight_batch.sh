#!/usr/bin/env bash
# Unattended overnight batch: gradient-clip probe + the two open knob cells.
#
# Runs sequentially, never prompts, and keeps going if one block fails so a
# single bad config cannot cost you the night. Every block tees its own log.
#
# Usage (on the pod, inside tmux):
#   bash scripts/run_overnight_batch.sh              # both blocks, seeds 1 2 3
#   bash scripts/run_overnight_batch.sh "1 2 3 4 5"  # more seeds
#
# Blocks:
#   A  gradclip  : 1 clip-free measurement run, then auto-bracket the clip around
#                  the measured p999/p99/p99-3 tail  -> 1 + 3x|seeds| runs
#   B  knob_fill : log_std_min (Adam mode) + value_learning_rate (SGD mode)
#                  -> 4x|seeds| runs
# ~7 min per run: with 3 seeds that is ~22 runs ~= 2.5 h.
set -uo pipefail   # NOT -e: a failed block must not abort the night

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"

SEEDS="${1:-1 2 3}"
STAMP="$(date +%Y%m%d_%H%M)"
LOGDIR="$REPO_ROOT/results/overnight_batch_$STAMP"
mkdir -p "$LOGDIR"

echo "=== overnight batch $STAMP | seeds: $SEEDS ==="
echo "block logs -> $LOGDIR"
START=$(date +%s)

run_block () {   # run_block <name> <cmd...>
    local name="$1"; shift
    echo ""
    echo "=== [$(date +%H:%M)] block $name starting ==="
    if "$@" 2>&1 | tee "$LOGDIR/${name}.log"; then
        echo "=== block $name done ==="
    else
        echo "!!! block $name FAILED (continuing) -- see $LOGDIR/${name}.log"
    fi
}

# --- A. gradient-clip probe (Marco's check, Euclidean 1/sigma^2 retained) -----
run_block gradclip \
    python scripts/run_gradclip_probe.py --stage auto --seeds $SEEDS

# --- B. the two open knob cells ---------------------------------------------
run_block knob_fill \
    python scripts/run_pcpg_knob_fill.py --seeds $SEEDS --skip-complete

# --- summarize both result trees --------------------------------------------
for d in results/gradclip_probe results/knob_fill_smin_vlr; do
    [ -d "$d" ] || continue
    echo ""
    echo "=== summarizing $d ==="
    python scripts/analyze_pcpg_logs.py --results-dir "$d" 2>&1 \
        | tee -a "$LOGDIR/summary.log"
done

MINS=$(( ($(date +%s) - START) / 60 ))
echo ""
echo "=== batch done in ${MINS} min ==="
echo "pull everything to your Mac (do this BEFORE terminating the pod):"
echo "  runpodctl send results/gradclip_probe results/knob_fill_smin_vlr $LOGDIR"
