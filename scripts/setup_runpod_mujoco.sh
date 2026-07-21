#!/usr/bin/env bash
# One-shot RunPod setup for the MuJoCo backprop baselines (Brax/mjx + JAX 0.4.38).
#
# Use a CUDA 12.x pod with a NON-Blackwell GPU (A100 / L40S / RTX 4090; not
# RTX 5090 / B200 / "RTX PRO"). Run it from anywhere:
#     bash scripts/setup_runpod_mujoco.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# 1. System libs MuJoCo links against (needed to import, even headless).
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux

# 2. JAX (pinned 0.4.38 for the backprop pmap), Brax/mjx env, YAML config parser.
#    --ignore-installed blinker: the OS-installed blinker has no RECORD file and
#    can't be uninstalled by pip, which would otherwise abort the whole install.
pip install --ignore-installed blinker \
    "jax[cuda12]==0.4.38" jaxlib==0.4.38 brax mujoco-mjx pyyaml

# 3. jaxlib 0.4.38 is built for CUDA 12.4, but modern pods pull CUDA 12.8/12.9
#    nvidia wheels, which segfault on the first matmul. Force the matching 12.4
#    libraries. (This breaks torch's CUDA, which we don't use.)
pip install --force-reinstall --no-deps \
    nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-cupti-cu12==12.4.127 \
    nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 \
    nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.2.1.3 \
    nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 \
    nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.4.127

# 4. Sanity: GPU visible + the wiring checks pass (dist/net/env).
python -c "import jax; print('devices:', jax.devices())"
python "$REPO_ROOT/scripts/verify_mujoco.py"

echo "Setup complete. Smoke-train with:"
echo "  python $REPO_ROOT/scripts/run_train.py --config $REPO_ROOT/configs/mujoco_halfcheetah.yaml --overrides agent.algorithm=ppo train.total_steps=100000"
