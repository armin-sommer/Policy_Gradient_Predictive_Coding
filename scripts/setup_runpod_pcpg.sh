#!/usr/bin/env bash
# RunPod setup for MuJoCo backprop + Gaussian PCPG (Brax/mjx + JAX 0.4.38 + jpc).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

bash "$SCRIPT_DIR/setup_runpod_mujoco.sh"

pip install --ignore-installed blinker equinox diffrax optax flax wandb matplotlib pyrallis \
    "git+https://github.com/thebuckleylab/jpc"

pip install -e "$REPO_ROOT" --no-deps 2>/dev/null || {
    # hatch may reject git deps; install src on PYTHONPATH via editable fallback
    pip install hatchling
    pip install -e "$REPO_ROOT" || true
}

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -c "import jpc; import brax; print('jpc + brax ok')"
python "$REPO_ROOT/scripts/verify_mujoco.py" --check env --env hopper

echo "PCPG setup complete. Example:"
echo "  PYTHONPATH=$REPO_ROOT/src python $REPO_ROOT/scripts/run_mujoco_pcpg_sweep.py --tiers bench --seeds 1 --total-steps 50000 --no-save"
