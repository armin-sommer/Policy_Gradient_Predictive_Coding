# PCPG — Predictive Coding Policy Gradients

On-policy policy gradient algorithms in JAX/Flax for investigating predictive coding in reinforcement learning.

## Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| REINFORCE | `src/algorithms/reinforce.py` | Vanilla policy gradient with Monte Carlo returns |
| PPO | `src/algorithms/ppo.py` | Proximal Policy Optimization (clipped surrogate) |
| TRPO | `src/algorithms/trpo.py` | Trust Region Policy Optimization (conjugate gradient + line search) |
| PCPG | `src/algorithms/pcpg.py` | Predictive Coding Policy Gradients (WIP) |

PPO, TRPO, and REINFORCE are adapted from [PolicyGradientsJax](https://github.com/Matt00n/PolicyGradientsJax).

## Project Structure

```
src/
  algorithms/    # Policy gradient algorithms
  networks/      # Shared MLP, distributions, policy interface (exact copy from PolicyGradientsJax)
  env/           # Vectorized env wrappers, normalization, evaluation
  utils/
configs/         # Experiment configs (YAML)
scripts/         # Training and evaluation entry points
```

## Setup

```bash
pip install -e .
```

For GPU (requires CUDA 12 on the host — RunPod "PyTorch 2.x" base images ship this):
```bash
pip install -e ".[gpu]"
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

## Running

```bash
# train
python scripts/run_train.py --config configs/default.yaml
python scripts/run_train.py --config configs/default.yaml --overrides agent.algorithm=trpo seed=7

# eval (after a checkpoint lands in outputs/checkpoints/)
python scripts/run_eval.py \
    --config configs/default.yaml \
    --checkpoint outputs/checkpoints/Exp_ppo_procgen__coinrun__42.params \
    --num-episodes 50
```

`run_train.py` reads the YAML, overrides the inline `Config` class on the selected algorithm module, and calls its `main()`. Checkpoints are written to `outputs/checkpoints/` unless `--no-save` is passed.

## Dependencies

JAX, Flax, Optax, procgen-mirror, gym3, pyyaml, wandb.

## Notes

- integrarte Distrax from GoogleDeeping - replace the distributions
- PPO, TRPO, and REINFORCE implementation taken from https://github.com/Matt00n/PolicyGradientsJax 
