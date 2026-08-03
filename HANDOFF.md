# Handoff — run the PCPG GPU experiments on Modal

You are picking this up in a **Remote Control** session on the user's own machine.
The previous session ran in a Claude Code cloud sandbox and **could not reach Modal**:
Modal's client is gRPC-only, `grpclib` has no HTTP-proxy support (verified: zero hits
for `HTTPS_PROXY`/`CONNECT` in the installed package), and all cloud-sandbox egress
goes through an HTTP CONNECT proxy. It failed at channel creation in ~0.05s, before
any auth. **This is not a token problem and no network-access level fixes it** — which
is why the work moved to a machine with direct egress.

Everything below is built, committed, and pushed. What's left is running it.

- **Branch: `fixing-pcpg`** @ `927d49a` (all the work described here)
- Secondary branch `claude/pcpg-fix-modal-experiments-t6tt73` @ `ada9f2c` holds an
  earlier copy of the Modal runner; **ignore it**, `fixing-pcpg` supersedes it.

---

## 0. Prerequisites

```bash
git checkout fixing-pcpg && git pull origin fixing-pcpg
pip install modal
modal token set --token-id <ID> --token-secret <SECRET> --profile=<workspace>
modal profile activate <workspace>
```

The user has the token; **do not** commit it or paste it into any file. Confirm with
`modal profile list` — a populated table means you are connected, which the cloud
sandbox never achieved.

---

## 1. Run the gate first

```bash
modal run modal_app.py::verify
```

This builds the image and checks `jax.devices()` is a GPU plus `verify_mujoco.py`
passes. **The image build has never been executed anywhere** — it is written from the
validated RunPod recipe but is unverified, so expect this step to be where problems
surface. Do not launch the grid until it prints OK.

Likely failures and fixes:

| Symptom | Cause | Fix |
|---|---|---|
| pip resolution conflict during build | `jpc==1.0.0` / `equinox==0.13.8` / `diffrax==0.7.2` vs `jax==0.4.38` | loosen only the failing pin in `requirements-gpu.txt`; **never** bump `jax` (the pmap code and jpc both need ≤0.5.2) |
| `verify` reports a CPU device | CUDA 12.4 pin didn't take | confirm `requirements-cuda124.txt` ran *after* `requirements-gpu.txt` with `--force-reinstall --no-deps` |
| `ptxas too old` / segfault | Blackwell GPU | `PCPG_GPU=A100-40GB` (never 5090 / PRO 6000 / B200) |
| runs queue instead of starting | workspace GPU quota | lower `PCPG_MAX_CONTAINERS` (default 8) |

`MODAL.md` has the full reference.

---

## 2. The two experiments, in priority order

### 2a. PCPG settling ablation ← **do this first, it's the open scientific question**

```bash
modal run modal_app.py --algos pc_actor_critic --arms both \
    --seeds 1,2,3 --envs halfcheetah --tier bench
```

6 runs (1 env × 1 algo × 2 arms × 3 seeds), one container each; prints a paired
settled-vs-unsettled contrast at the end. Add `--algos pc_actor_critic,pc_reinforce`
for 12, or more seeds if the contrast is borderline. Logs:
`/results/pcpg_modal/<env>_<algo>_<tier>_<arm>_seed<n>.log` in the `pcpg-results`
volume.

**Why:** every committed PCPG result was produced with inference ~1% settled. The fix
(`train.inference_rate_correction`) makes it 99.5% settled. On the bandit, settling
significantly **slows** learning (+3661 ± 407 steps to π≥0.9, t(4)=+9.00). Whether
MuJoCo behaves the same way is unknown and is the thing to find out.

**The prediction to test:** in the MuJoCo/Gaussian geometry, `cos(d_BP, d_EQ) ≈ 0.99`
— inference barely rotates the update — so settling should change little *there*, even
though it changed the bandit a lot (where `cos ≈ 0.78`). If MuJoCo *does* move
substantially, the geometry story in `docs/PCPG_GEOMETRY_FINDINGS.md` §4b needs
revisiting. Report either outcome plainly; a null here is a real result.

### 2b. Backprop baseline seeds 4–5

```bash
modal run modal_app.py            # default grid: 4 envs x {ppo,trpo} x seeds 4,5
```

16 runs. `docs/BENCHMARK_PROTOCOL.md` §7 requires 5 seeds per (task, algo); seeds 1–3
are complete in `results/mujoco/`, 4–5 are missing for all 8 cells. Resumable — a log
containing `TRAINING END` is skipped, so re-launching only fills gaps.

---

## 3. After the runs

```bash
modal run modal_app.py::fetch     # pulls volume logs into results/mujoco/
python scripts/summarize_mujoco.py
```

Then commit the logs and a summary to `fixing-pcpg` and push.

**Reporting discipline** (`BENCHMARK_PROTOCOL.md` §5): no best-over-evals in reported
numbers — the `best=` column the runner prints is a progress read only. Use final +
aggregate with uncertainty. For the ablation, pair by seed and use a real t-test; note
that `n=3` is thin, and that a 2·SEM rule is too liberal at these n (`t_crit`=4.303 at
n=3). `scripts/run_settled_ablation.py` already does this correctly for the bandit —
follow its statistics, and reuse its steps-to-threshold metric, because **final score
alone had no power** on the bandit (both arms saturated) while speed-to-threshold was
unambiguous.

---

## 4. State of the science (so you don't re-derive it)

Full detail in `docs/PCPG_GEOMETRY_FINDINGS.md`. Headlines:

- **The PCPG update does not have natural-gradient geometry.** Inference degrades
  alignment with `d_NG` at every `t₁` in all four regimes; the only NG content comes
  from the `natural_target` flag's `F_out⁻¹`.
- **Innocenti's trust region is inert in this architecture.** `S = I + Σ_l B_l B_lᵀ`
  has eigenvalues in [1.0, 2.0], cond 1.5–1.8, so `cos(d_BP, d_EQ) ≈ 0.99` at every
  depth and in linear nets too. It rescales step *length*, not direction.
- **Caveat on that:** `d_EQ` **cannot** test the paper's actual saddle-escape claim —
  `S → I` as weights approach the origin saddle, so the instrument goes inert exactly
  where the theory predicts the largest effect. Don't cite §4b against the paper.
- **`log_std` targets are ~81% truncated at bench `ts10`** even with σ mid-window
  (F7) — the clamp binds through the *target*, not through where σ sits.
- **No trust region is enforced anywhere** in `src/pc_algorithms/` — `policy_kl` is
  diagnostic-only. `kl_max` reached 170 vs TRPO's `target_kl` of 0.01, so
  experiment-log §3.4's σ-parameterisation conclusion is confounded (documented there).

**Performance context:** PC wins on the bandit (0.968 ± 0.029, 9/10, where PPO and
REINFORCE get ~0.02, 0/10) and fails on MuJoCo (best *stable* ~781; best single 3205
with 2/3 collapse; PPO reaches 9,673 at matched 5M budget). The bandit is 2 arms and
1 step, so treat it as a toy positive.

---

## 5. Files you'll touch

| Path | What |
|---|---|
| `modal_app.py` | the runner; `--tier`, `--arms`, `PCPG_GPU`, `PCPG_MAX_CONTAINERS` |
| `MODAL.md` | setup, knobs, troubleshooting |
| `requirements-gpu.txt` / `requirements-cuda124.txt` | pinned deps; CUDA 12.4 must install second |
| `src/pc_algorithms/inference.py` | the settling fix (`make_pc_step_at_rate`) |
| `scripts/run_settled_ablation.py` | bandit ablation + the statistics to copy |
| `scripts/probe_inference_settling.py` | residual-vs-τ curve, if you need to re-derive settling |
| `docs/PCPG_GEOMETRY_FINDINGS.md` | the coauthor-facing writeup — **update it with the MuJoCo result** |

---

## 6. If the user asks what else is worth doing

Ranked, from the writeup's recommendations:

1. **Enforce a trust region** (KL cap or `target_clip`). Nothing bounds the step today,
   so no σ or geometry conclusion has a valid arm. This is the top blocker independent
   of everything above.
2. **Decouple `target_scale` from the `log_std` clamp** so that channel encodes what
   the theory assumes.
3. **Build Probe 2** (real-checkpoint dump hook) — the one route by which the
   "`S` is inert" finding could soften, since it currently rests on initialisation
   geometry only.
4. **A genuine saddle-escape experiment** (init at/near a saddle, PC vs BP under
   *plain GD* — the paper's property is stated for plain gradient descent, while every
   bench cell is Adam). This tests the paper on its own terms; nothing in the repo does.

Do not read the `docs/PCPG_GEOMETRY_FINDINGS.md` conclusions as settled beyond their
stated scope: single seed, `n=256`, initialisation weights, one architecture family.
The previous session over-generalised §4b once already and the bandit ablation caught
it.
