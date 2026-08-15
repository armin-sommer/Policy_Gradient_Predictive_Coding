#!/usr/bin/env bash
set -u

root=/workspace/PCPG_joint_ablation
out="$root/results/joint_error_neuron_ablation_30m"
config="$root/configs/mujoco_halfcheetah_pc_actor_critic_parameter_local_precision_tr.yaml"
mkdir -p "$out"

pids=()
for seed in 1 2 3; do
  gpu=$((seed - 1))
  stem="halfcheetah_local_precision_tr_30000000_seed${seed}"
  log="$out/${stem}.log"
  if [[ -f "$log" ]] && ! grep -q 'TRAINING END' "$log"; then
    mv "$log" "$out/${stem}.interrupted_joint_shutdown.log"
  fi
  if [[ -f "$out/${stem}.status.json" ]]; then
    mv "$out/${stem}.status.json" \
      "$out/${stem}.interrupted_joint_shutdown.status.json"
  fi
  echo "start precision seed $seed on gpu $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$root/src" \
    python "$root/scripts/run_train.py" \
      --config "$config" --no-save --overrides \
      train.total_steps=30000000 seed="$seed" \
      train.eval_every=10 train.num_eval_episodes=10 \
      > "$log" 2>&1 &
  pids+=("$!")
done

result=0
for pid in "${pids[@]}"; do
  wait "$pid" || result=1
done
exit "$result"
