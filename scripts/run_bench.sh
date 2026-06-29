#!/usr/bin/env bash
# Run PPO + TRPO on HalfCheetah + Hopper across seeds, using the per-(env,algo)
# SOTA configs (configs/mujoco_<env>_<algo>.yaml). Sequential and RESUMABLE:
# any run whose log already contains "TRAINING END" is skipped, so you can
# re-launch after an interruption and it picks up where it left off.
#
# Run it inside tmux so it survives disconnect:
#   tmux new -s bench
#   bash scripts/run_bench.sh
#   (detach: Ctrl+B then D   |   reattach: tmux attach -t bench)
#
# Override the grid via env vars, e.g.:
#   SEEDS="1 2 3" ENVS="hopper" ALGOS="trpo" bash scripts/run_bench.sh
set -u
cd "$(dirname "$0")/.."

ENVS="${ENVS:-halfcheetah hopper}"
ALGOS="${ALGOS:-ppo trpo}"
SEEDS="${SEEDS:-1 2 3}"
OUT="results/mujoco"
mkdir -p "$OUT"

total=0; ran=0; skipped=0; failed=0
start=$(date +%s)

for env in $ENVS; do
  for algo in $ALGOS; do
    cfg="configs/mujoco_${env}_${algo}.yaml"
    if [[ ! -f "$cfg" ]]; then
      echo "!! MISSING $cfg — skipping all $env/$algo runs"; continue
    fi
    for seed in $SEEDS; do
      total=$((total+1))
      log="$OUT/${env}_${algo}_sota_seed${seed}.log"
      if [[ -f "$log" ]] && grep -q "TRAINING END" "$log"; then
        echo "== SKIP  $env $algo seed$seed (already complete)"; skipped=$((skipped+1)); continue
      fi
      echo ""
      echo "===================================================================="
      echo "== RUN   $env $algo seed$seed   ($(date '+%H:%M:%S'))  -> $log"
      echo "===================================================================="
      python scripts/run_train.py --config "$cfg" --overrides "seed=$seed" \
        2>&1 | tee "$log"
      if grep -q "TRAINING END" "$log"; then ran=$((ran+1)); else
        echo "!! $env $algo seed$seed did NOT finish (no TRAINING END) — see $log"
        failed=$((failed+1))
      fi
    done
  done
done

mins=$(( ($(date +%s) - start) / 60 ))
echo ""
echo "ALL DONE in ${mins} min — ran=$ran skipped=$skipped failed=$failed total=$total"
echo "logs in $OUT/<env>_<algo>_sota_seed<seed>.log"
echo "plot any with: python scripts/plot_curve.py $OUT/<env>_<algo>_sota_seed1.log --title '...'"
