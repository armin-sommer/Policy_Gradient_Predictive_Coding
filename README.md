# PCPG — Predictive Coding Policy Gradients

On-policy policy gradients in JAX, comparing backprop-trained policies against predictive-coding-trained ones ([jpc](https://github.com/thebuckleylab/jpc)).

## Structure

```
src/
  backprop_algorithms/   # REINFORCE, PPO, Cleanba PPO, TRPO (from PolicyGradientsJax / CleanRL)
  pc_algorithms/         # PC-REINFORCE, PC actor-critic (jpc, no backprop)
  networks/              # MLP/CNN, distributions (from PolicyGradientsJax)
  env/                   # Procgen wrapper + 2-armed bandit
configs/                 # YAML configs
scripts/                 # run_train.py, run_eval.py, run_bandit_comparison.py, run_bandit_multi_seed.py
results/                 # committed plots, logs, CSVs
```

Each algorithm file has an inline `Config` and a `main()`; `run_train.py` maps the YAML onto `Config` and dispatches via `agent.algorithm`.

## Setup

```bash
pip install -e .
git config core.hooksPath .githooks   # once per clone
```

Use JAX 0.4.38 + Flax 0.10.2 + Optax 0.2.4 (jpc needs JAX <= 0.5.2, the pmap code breaks on newer JAX anyway). For GPU: `pip install -e ".[gpu]"` (CUDA 12).

## Running

```bash
# Procgen
python scripts/run_train.py --config configs/default.yaml
python scripts/run_train.py --config configs/default.yaml --overrides agent.algorithm=trpo seed=7

# bandit comparison (plot + logs go to results/bandit_seed{seed}/)
python scripts/run_bandit_comparison.py --seed 0
python scripts/run_bandit_multi_seed.py          # seeds 1-6, 8-21
python scripts/summarize_bandit_seeds.py         # -> results/bandit_multi_seed/

# eval a checkpoint
python scripts/run_eval.py --config configs/default.yaml \
    --checkpoint outputs/checkpoints/<name>.params --num-episodes 50
```

## Results

2-armed bandit (arm means 1.0 / 0.9), softmax policy started adversarially at pi(optimal) ~ 2%. Vanilla PG has a vanishing gradient there (pi(1-pi)·gap); natural PG cancels it via the Fisher.

![bandit comparison](results/bandit_seed0/bandit_npg_vs_pg_seed0.png)

**22 seeds** (0, 1–6, 7, 8–21; 60k env steps each) — `results/bandit_multi_seed/`:

![multi-seed learning curves](results/bandit_multi_seed/mean_learning_curve_sem.png)

| | final π(opt) mean±SEM | P(final≥0.9) | median steps to 0.9 |
|---|---|---|---|
| TRPO | 0.996±0.003 | 22/22 | 6000 |
| PC-REINFORCE | 0.985±0.013 | 21/22 | 38000 |
| PC actor-critic | 0.975±0.014 | 20/22 | 43000 |
| Cleanba PPO | 0.025±0.003 | 0/22 | — |
| REINFORCE | 0.018±0.002 | 0/22 | — |

TRPO and both PC variants escape the adversarial init on almost every seed; first-order backprop methods stay on the plateau.

## TODO

- scale PCPG to Procgen (multi-step TD)
- replace hand-rolled distributions with distrax
