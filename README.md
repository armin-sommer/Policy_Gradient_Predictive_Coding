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
scripts/                 # run_train.py, run_eval.py, run_bandit_comparison.py, run_bandit_multi_init.py
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

# bandit comparison (results/bandit_{init}_seed{seed}/ or bandit_seed{seed} for favor_suboptimal)
python scripts/run_bandit_comparison.py --seed 0
python scripts/run_bandit_comparison.py --init uniform --seed 1
python scripts/run_bandit_comparison.py --logit-bias 1 0 --seed 1   # mild_optimal
python scripts/run_bandit_multi_init.py                            # 5 inits × seeds 1–10
python scripts/summarize_bandit_inits.py                           # -> results/bandit_multi_init/

# eval a checkpoint
python scripts/run_eval.py --config configs/default.yaml \
    --checkpoint outputs/checkpoints/<name>.params --num-episodes 50
```

## Results

2-armed bandit (arm means 1.0 / 0.9), softmax policy started adversarially at pi(optimal) ~ 2%. Vanilla PG has a vanishing gradient there (pi(1-pi)·gap); natural PG cancels it via the Fisher. Where does PC land?

![bandit comparison](results/bandit_seed0/bandit_npg_vs_pg_seed0.png)

| | final pi(opt) | avg pi(opt) |
|---|---|---|
| TRPO (natural PG) | 1.000 | 0.972 |
| PC actor-critic (TD value head) | 1.000 | 0.554 |
| PC-REINFORCE (MC returns) | 1.000 | 0.487 |
| Cleanba PPO | 0.020 | 0.023 |
| REINFORCE (SGD) | 0.010 | 0.015 |

Both PC variants escape the plateau and converge; the first-order backprop methods stay stuck. Same picture on seed 7 (`results/bandit_seed7/`). The PC policies are trained entirely with `jpc.make_pc_step` on advantage-weighted output targets — no backprop; the actor-critic variant bootstraps from a PC-trained value head instead of Monte Carlo returns.

**10 seeds (1–10, 60k env steps each)** — aggregate in `results/bandit_multi_seed/`:

![multi-seed learning curves](results/bandit_multi_seed/mean_learning_curve_sem.png)

| | final π(opt) mean ± SEM | success (≥0.9) | steps to ≥0.9 (median) |
|---|---|---|---|
| TRPO | 0.993 ± 0.007 | 10/10 | 6000 |
| PC-REINFORCE | 0.968 ± 0.029 | 9/10 | 38000 |
| PC actor-critic | 0.969 ± 0.024 | 9/10 | 44000 |
| Cleanba PPO | 0.026 ± 0.003 | 0/10 | — |
| REINFORCE | 0.018 ± 0.003 | 0/10 | — |

Re-run: `python scripts/run_bandit_multi_seed.py` then `python scripts/summarize_bandit_seeds.py`.

**5 policy inits × 10 seeds** — `results/bandit_multi_init/` (presets in `scripts/bandit_inits.py`):

![cross-init final pi](results/bandit_multi_init/cross_init_final_pi.png)

| init (π₀) | TRPO final ± SEM | TRPO ≥0.9 | PC-R final ± SEM | PC-R ≥0.9 | REINFORCE final ± SEM |
|---|---|---|---|---|---|
| favor_optimal (98%) | 0.995 ± 0.005 | 10/10 | 1.000 ± 0.000 | 10/10 | 0.982 ± 0.003 |
| mild_optimal (73%) | 0.992 ± 0.008 | 10/10 | 0.999 ± 0.001 | 10/10 | 0.849 ± 0.009 |
| uniform (50%) | 0.994 ± 0.006 | 10/10 | 0.996 ± 0.002 | 10/10 | 0.756 ± 0.012 |
| mild_suboptimal (27%) | 1.000 ± 0.000 | 10/10 | 0.994 ± 0.004 | 10/10 | 0.559 ± 0.017 |
| favor_suboptimal (2%) | 0.993 ± 0.007 | 10/10 | 0.968 ± 0.029 | 9/10 | 0.018 ± 0.003 |

TRPO and both PC variants reach π(opt) ≥ 0.9 from every init we tried; first-order REINFORCE and PPO only succeed when the policy already favors the optimal arm. The harder the suboptimal start, the slower PC learns (median ~19k steps at uniform vs ~38k at favor_suboptimal), but TRPO stays fast until π₀ drops below ~3%.

## TODO

- scale PCPG to Procgen (multi-step TD)
- replace hand-rolled distributions with distrax
