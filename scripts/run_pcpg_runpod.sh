#!/usr/bin/env bash
# One-shot PCPG RunPod driver: setup -> verify -> sweep -> summarize -> package.
#
# Usage:  bash scripts/run_pcpg_runpod.sh [ENV] [TIERS] [SEEDS] [extra sweep args...]
#   defaults: halfcheetah  "bench"  "1 2 3"
# Examples:
#   bash scripts/run_pcpg_runpod.sh                                   # halfcheetah bench, seeds 1 2 3
#   bash scripts/run_pcpg_runpod.sh halfcheetah bench 1 --total-steps 50000 --no-save   # quick smoke
#   bash scripts/run_pcpg_runpod.sh halfcheetah "bench sota" "1 2 3"  # both tiers
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

ENV="${1:-halfcheetah}"
TIERS="${2:-bench}"
SEEDS="${3:-1 2 3}"
EXTRA=("${@:4}")

# 1. setup (skip if jpc + brax already import, so reruns are fast)
python -c "import jpc, brax" 2>/dev/null || bash "$SCRIPT_DIR/setup_runpod_pcpg.sh"

# 2. verify the env builds on this GPU before committing to a long run
python "$REPO_ROOT/scripts/verify_mujoco.py" --check env --env "$ENV"

# 3. sweep (resumable: --skip-complete skips runs whose log already has TRAINING END)
python "$REPO_ROOT/scripts/run_mujoco_pcpg_sweep.py" \
    --env "$ENV" --tiers $TIERS --seeds $SEEDS --skip-complete "${EXTRA[@]}"

# 4. summarize -> CSV + PNG learning curves, saved next to the logs
RESULTS="$REPO_ROOT/results/mujoco_pcpg_${ENV}"
python "$REPO_ROOT/scripts/summarize_mujoco.py" --results-dir "$RESULTS"

# 5. print the retrieval command (Terminate wipes the pod -- pull first)
echo
echo "=== done. pull results to your Mac: ==="
echo "  (on pod)  runpodctl send $RESULTS"
echo "  (on mac)  runpodctl receive <code>"
