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
STOP_GRACE="${STOP_GRACE:-60}"   # seconds to cancel the auto-stop
STAMP="$(date +%Y%m%d_%H%M)"
LOGDIR="$REPO_ROOT/results/overnight_batch_$STAMP"
mkdir -p "$LOGDIR"

echo "=== overnight batch $STAMP | seeds: $SEEDS ==="
echo "block logs -> $LOGDIR"

# --- preflight: verify auto-stop BEFORE burning a night of GPU ---------------
# Finding out at 3am that the stop was unauthorized is the expensive failure.
if [ "${AUTOSTOP:-1}" = "1" ]; then
    if runpodctl get pod "${RUNPOD_POD_ID:-}" >/dev/null 2>&1; then
        echo "auto-stop: OK (pod ${RUNPOD_POD_ID} will stop when done)"
    else
        echo ""
        echo "!!! ============================================================"
        echo "!!! AUTO-STOP WILL NOT WORK -- the pod would keep BILLING."
        echo "!!! runpodctl cannot reach the API (missing/invalid API key)."
        echo "!!!"
        echo "!!! Fix, in THIS shell, then re-run:"
        echo "!!!     export RUNPOD_API_KEY=<your-key>     # console/user/settings"
        echo "!!!     runpodctl get pod \$RUNPOD_POD_ID     # must not say Unauthorized"
        echo "!!!"
        echo "!!! Continuing in 20s anyway (Ctrl-C to abort and fix)."
        echo "!!! ============================================================"
        sleep 20
    fi
fi
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

# --- push results so the pod is disposable ----------------------------------
# Pushes to a DEDICATED branch (results/overnight-<stamp>), never to the feature
# branch: the pod's results/ tree diverges from your Mac's, and pushing there
# would recreate the untracked-file collision. Merge/cherry-pick at your leisure.
#
# Auth: needs a token-bearing remote, which YOU set up once on the pod:
#     git remote set-url origin https://<YOUR_TOKEN>@github.com/armin-sommer/Policy_Gradient_Predictive_Coding.git
# (a fine-grained PAT with Contents:read+write on this repo is enough).
# Without it this block is skipped and the runpodctl fallback is printed.
RESULT_BRANCH="results/overnight-$STAMP"
# Always return to the branch we started on. Leaving the pod checked out on a
# throwaway results branch strands the next `git pull` ("no tracking information").
START_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
restore_branch () {
    [ -n "$START_BRANCH" ] || return 0
    [ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" = "$START_BRANCH" ] && return 0
    git checkout -q "$START_BRANCH" 2>/dev/null \
        && echo "returned to branch $START_BRANCH" \
        || echo "!!! could not return to $START_BRANCH -- run: git checkout $START_BRANCH"
}
push_results () {
    local rc=0
    git config user.email "${GIT_AUTHOR_EMAIL:-pod@runpod.local}"
    git config user.name  "${GIT_AUTHOR_NAME:-runpod batch}"
    git checkout -q -b "$RESULT_BRANCH" || return 1
    git add -f results/gradclip_probe results/knob_fill_smin_vlr "$LOGDIR" 2>/dev/null
    if git commit -q -m "Overnight batch $STAMP: gradclip probe + knob fill results"; then
        git push -q origin "$RESULT_BRANCH" || rc=1
    else
        echo "nothing new to commit"; rc=1
    fi
    restore_branch          # <-- always, success or failure
    return $rc
}
trap restore_branch EXIT    # <-- also on Ctrl-C or an unexpected exit

stop_pod () {
    # Stops (does NOT terminate) this pod -> GPU billing ends, /workspace volume
    # and everything in it survives for the next start.
    local pid="${RUNPOD_POD_ID:-}"
    if ! command -v runpodctl >/dev/null 2>&1; then
        echo "!!! runpodctl not on PATH -- cannot auto-stop. Stop the pod in the console."
        return 1
    fi
    if [ -z "$pid" ]; then
        echo "!!! RUNPOD_POD_ID not set -- cannot auto-stop. Stop the pod in the console."
        return 1
    fi
    echo ""
    echo "=== auto-stop: stopping pod $pid in ${STOP_GRACE}s -- Ctrl-C to cancel ==="
    sleep "$STOP_GRACE"
    runpodctl stop pod "$pid"
}

if git ls-remote --exit-code origin >/dev/null 2>&1 && push_results; then
    echo ""
    echo "=== results pushed to branch: $RESULT_BRANCH ==="
    echo "SAFE TO TERMINATE THE POD."
    echo "on your Mac:  git fetch origin && git checkout $RESULT_BRANCH"
else
    echo ""
    echo "!!! could not push (no write credentials on this pod, or nothing to commit)."
    echo "!!! if a commit was made it is on the LOCAL branch $RESULT_BRANCH;"
    echo "!!! you are back on $START_BRANCH and the files are still on disk at:"
    echo "      $REPO_ROOT/results/{gradclip_probe,knob_fill_smin_vlr}"
    echo ""
    echo "  ==> STOP the pod (safe: /workspace persists, billed storage only)."
    echo "  ==> DO NOT TERMINATE -- terminate deletes the volume and these runs."
    echo ""
    echo "  Restart the pod later and pull with:"
    echo "    runpodctl send results/gradclip_probe results/knob_fill_smin_vlr $LOGDIR"
    echo "    (then on your Mac: runpodctl receive <code>)"
fi

# --- stop the pod so billing ends -------------------------------------------
# On by default: the whole point of an overnight batch is not paying for idle
# GPU after it finishes. AUTOSTOP=0 disables. Stop != terminate: the volume and
# all results survive and are there when you start the pod again.
if [ "${AUTOSTOP:-1}" = "1" ]; then
    stop_pod || echo "!!! AUTO-STOP FAILED -- pod is still running and BILLING."
else
    echo ""
    echo "AUTOSTOP=0 -> pod left running (still billing)."
fi
