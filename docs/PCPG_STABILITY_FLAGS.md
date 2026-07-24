# PCPG stability flags

Knobs added to probe / fix the PCPG instability on continuous control. **All
default to off** — with everything unset you get the original behavior. They live
on the `Config` of `pc_actor_critic.py` and `pc_reinforce.py`, and are settable
three ways:

- **YAML** (via `run_train.py`): the `KEY_MAP` path shown below.
- **`gen_benchmark_config.py`**: the `--flag` shown below (also stamps a name tag).
- **CLI override**: `--overrides train.target_clip=2.0`.

## Where each acts in the PC update

```
advantage A ─▶ target = μ + ts·A·(z−μ)/σ²  ─▶ PC inference ─▶ weight grad ─▶ optimizer step
                     │            │                                  │            │
              natural_target  target_clip                      max_grad_norm  (Adam ignores)
              log_std_min (σ floor, feeds the 1/σ² term)
              state_indep_std (how σ itself is parameterized)
```

## The flags

| Flag | Values (default) | YAML path | gen flag | name tag | What it does |
|---|---|---|---|---|---|
| `natural_target` | bool (`False`) | `train.natural_target` | `--natural-target` | `_nat` | Fisher-precondition the target: mean offset `ts·A·(z−μ)` instead of `ts·A·(z−μ)/σ²` (drops the `1/σ²` amplifier), log_std offset halved. The "PC = natural gradient" target. |
| `target_clip` | float / `None` | `train.target_clip` | `--target-clip X` | `_tclipX` | Cap the mean-target offset per coord (output-space trust region). Optimizer-agnostic. |
| `target_clip_rel` | bool (`False`) | `train.target_clip_rel` | `--target-clip-rel` | `_tcliprelX` | Makes the cap relative: `|Δμ| ≤ target_clip·σ` instead of absolute. |
| `log_std_min` | float / `None` (`−2`) | `agent.log_std_min` | `--log-std-min X` | `_sminX` | Raise the σ floor (e.g. `−1` → σ_min 0.37, capping `1/σ²` at ~7 vs ~55). |
| `max_grad_norm` | float / `None` | `train.max_grad_norm` | `--max-grad-norm X` | `_clipX` | Global-norm clip on the PC **policy** gradient (pre-optimizer). Note: **Adam renormalizes it away** — mainly affects SGD. |
| `state_indep_std` | bool (`False`) | `agent.state_indep_std` | `--state-indep-std` | `_stdglobal` | Single global `log_std` vector (PPO-style) instead of a per-state std head. |

They **compose**: e.g. `natural_target` + `target_clip` applies the clip on top of
the natural offset.

## Diagnostics logged every update (`diag/…`)

| Metric | Meaning |
|---|---|
| `policy_kl_max/mean` | exact per-update `D_KL(π_old‖π_new)` — the trust-region quantity |
| `policy_grad_norm_max/mean` | PC policy-gradient global norm, **pre-clip** (exploding-gradient probe) |
| `mu_target_mag_max/mean` | raw mean-target offset magnitude, **pre-clip**, reflects the natural/Euclidean family in use |
| `policy_drift_max/mean` | `|μ_post−μ_pre|/σ` per update |
| `pretanh_sat_frac` | fraction of `|z|>2` (tanh saturation) |
| `log_std_mean/min`, `frac_std_at_min/max` | where σ sits vs its clamps |
| `value_explained_var` | critic fit (actor-critic only) |

## Examples

```bash
# natural-gradient target, ts=1.0, adam, depth 20
python scripts/gen_benchmark_config.py --algo pc_actor_critic --tier bench \
  --opt adam --act tanh --ts 1.0 --max-t1 20 --natural-target
# -> ..._adam_tanh_ts10_bench_mt20_nat.yaml

# target clip 2 + raised σ floor, sgd (baseline lr = 0.03!)
python scripts/gen_benchmark_config.py --algo pc_actor_critic --tier bench \
  --opt sgd --act tanh --ts 1.0 --max-t1 80 --lr 0.03 --target-clip 2 --log-std-min -1
# -> ..._sgd_..._lr003_mt80_tclip2_sminm1.yaml
```

## Naming gotcha (learning rate)

The lr tag is the decimals after `0.`, so **`lr003` = 0.03** and **`lr0003` = 0.003**
(one extra zero = 10× smaller). The SGD baseline that learns uses **0.03** →
always pass `--lr 0.03` for SGD.
