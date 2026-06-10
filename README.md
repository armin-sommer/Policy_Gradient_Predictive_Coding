# PCPG — Predictive Coding Policy Gradients

On-policy policy gradient algorithms in JAX/Flax for investigating predictive coding in reinforcement learning.

## Algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| REINFORCE | `src/backprop_algorithms/reinforce.py` | Vanilla policy gradient with Monte Carlo returns |
| PPO | `src/backprop_algorithms/ppo.py` | Proximal Policy Optimization (clipped surrogate) |
| Cleanba PPO | `src/backprop_algorithms/cleanba_ppo.py` | Cleanba/CleanRL-style PPO baseline (GAE once per iteration, flattened minibatches, clipped value loss, Adam eps 1e-5) |
| TRPO | `src/backprop_algorithms/trpo.py` | Trust Region Policy Optimization (conjugate gradient + line search) |
| PCPG | `src/backprop_algorithms/pcpg.py` | Predictive Coding Policy Gradients (WIP) |

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

After cloning, enable the repo's git hooks (strips AI agent attribution from commit messages):
```bash
git config core.hooksPath .githooks
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

## Bandit test environment (NPG vs vanilla PG)

`src/env/bandit.py` implements a 2-armed bandit: 1 state (constant observation), 2 actions, 1-step episodes. Arm 0 pays 1.0, arm 1 pays 0.9 (configurable via `env.arm_means`).

```bash
# Cleanba PPO baseline on the bandit
python scripts/run_train.py --config configs/bandit.yaml

# REINFORCE (vanilla PG) vs TRPO (natural PG) comparison + plot
python scripts/run_bandit_comparison.py --seed 0
```

The comparison uses an adversarial initialization (`agent.policy_init_logit_bias: [0.0, 4.0]`, i.e. the policy starts at pi(optimal arm) ~ 2%). In this setup the natural policy gradient provably beats the vanilla policy gradient for any seed and step size:

- With a softmax policy the vanilla PG gradient on the logit gap is pi(1-pi) * gap, which is ~0.002 at the adversarial init — vanilla PG is stuck on a plateau of length ~1/pi updates.
- NPG preconditions with the inverse Fisher information (1/(pi(1-pi)) for a 2-arm softmax), which exactly cancels the vanishing factor: constant progress in logit space per update, independent of the current policy.
- Rewards are deterministic, so the gap is purely geometric (Mei et al. 2020: softmax PG converges O(1/t) with plateau-dependent constants; NPG converges linearly).

## Dependencies

JAX, Flax, Optax, procgen-mirror, gym3, pyyaml, wandb.

## Notes

- integrarte Distrax from GoogleDeeping - replace the distributions
- PPO, TRPO, and REINFORCE implementation taken from https://github.com/Matt00n/PolicyGradientsJax 
