# RunPod runbook — the settled-vs-unsettled inference ablation

Copy-paste for one pod. The question: **every committed PCPG result was produced with
inference ~1% settled.** Does actually settling it change MuJoCo learning? On the
bandit it did (settling was significantly *slower*). See
`docs/PCPG_GEOMETRY_FINDINGS.md` §§4d–4e.

Prerequisite from `RUNPOD.md`: **non-Blackwell GPU** (A100 / H100 / L40S / RTX 4090 —
never RTX 5090, RTX PRO 6000, B200, B300; `jaxlib 0.4.38` cannot compile for compute
capability 12.0) on a **CUDA 12.4** template (PyTorch 2.4 / `cu124`), not 12.8.

---

## 1. Connect and set up (~10 min)

```bash
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

```bash
cd /workspace
apt-get update && apt-get install -y libgl1 libglib2.0-0 tmux

git clone https://github.com/armin-sommer/Policy_Gradient_Predictive_Coding.git PCPG
cd PCPG
git checkout fixing-pcpg

bash scripts/setup_runpod_pcpg.sh          # jax 0.4.38 + brax/mjx + jpc, then verifies
export PYTHONPATH=/workspace/PCPG/src:$PYTHONPATH
```

Gate — do not continue unless both pass:

```bash
python -c "import jax; print(jax.devices())"      # MUST be [CudaDevice(id=0)]
python -c "import jpc, brax; print('jpc + brax ok')"
```

If `jax.devices()` shows CPU, the CUDA pin didn't take — see `RUNPOD.md` §7.

---

## 2. Calibrate first (~10 min) — this replaces a guess with a measurement

```bash
python scripts/run_settling_ablation_mujoco.py --calibrate
```

Runs one short pair (100k steps, both arms) and prints the **measured** settled /
unsettled wall-clock ratio plus an extrapolated GPU-hour budget for the full grid.

Why this step exists: the ~9.5× cost figure in the docs was measured on **CPU at bench
batch size**, from solver step counts (5 unsettled → 39 at `max_t1=10` → 62 at 20). The
step counts transfer to GPU but the per-step cost may not, and the multiplier applies to
the PC step rather than the whole loop. One short run settles it. **Read the ratio before
launching §3** — if it comes back far above ~10×, drop to `--seeds 1 2` or
`--steps 500000` rather than burning hours.

---

## 3. The ablation, detached (tmux)

```bash
tmux new -s ablation
export PYTHONPATH=/workspace/PCPG/src:$PYTHONPATH

python scripts/run_settling_ablation_mujoco.py --seeds 1 2 3
# detach: Ctrl+B then D      reattach: tmux attach -t ablation
```

Defaults: `halfcheetah`, `pc_actor_critic`, `bench` tier (1M steps), `max_t1=10`,
both arms × 3 seeds = **6 runs**, resumable (a log containing `TRAINING END` is
skipped, so re-launching only fills gaps).

`max_t1=10` is deliberate: it already converges (residual 0.002) and costs ~1.6× less
than 20 (39 vs 62 solver steps). Raising it buys nothing.

Add the second algorithm once the first grid is in:

```bash
python scripts/run_settling_ablation_mujoco.py --seeds 1 2 3 --algos pc_reinforce
```

Monitor without reattaching:

```bash
grep "eval/mean_score" results/settling_ablation_mujoco/*_settled_seed1.log | tail
```

---

## 4. Pull the results back

```bash
# on the POD
cd /workspace/PCPG
tar czf ablation.tar.gz results/settling_ablation_mujoco/
runpodctl send ablation.tar.gz            # prints a code

# on your MAC
runpodctl receive <code>                  # brew install runpod/runpodctl/runpodctl
```

Or commit from the pod, if you set up git credentials there:

```bash
git add results/settling_ablation_mujoco && git commit -m "MuJoCo settling ablation" \
  && git push origin fixing-pcpg
```

**Terminate the pod when done — it bills until you do.**

---

## 5. What the output means

The script prints, per algorithm, the paired settled − unsettled difference in final
eval return with a real **paired t-test** and 95% CI (`t_crit` is 3.182 at n=3, so
n=3 is thin — treat a borderline result as inconclusive rather than negative).

- **No detectable effect** → consistent with the geometry probe: in the MuJoCo/Gaussian
  geometry `cos(d_BP, d_EQ) ≈ 0.99`, so inference barely rotates the update and settling
  should not matter *there*. It also means the committed PCPG results do not need
  re-running despite having been produced unsettled — which is the practically
  important outcome.
- **A real difference** → `docs/PCPG_GEOMETRY_FINDINGS.md` §4b does not hold on this
  geometry either, and every prior PCPG conclusion is suspect, since all of them ran at
  ~1% settled. That is the expensive outcome, so it is worth knowing early.

Either way it is a result. The bandit gave the second answer (+3661 ± 407 steps to
threshold, t(4)=+9.00), which is why this is worth the GPU time.

Report both the final-return table and the paired test. Best-eval is printed as a
sensitivity check only — `docs/BENCHMARK_PROTOCOL.md` §5 forbids best-over-evals in
headline numbers.

---

## 6. If there is spare GPU time

The backprop baseline grid still needs **seeds 4–5** for all 8 (env, algo) cells —
`BENCHMARK_PROTOCOL.md` §7 requires 5 seeds and only 1–3 are complete:

```bash
SEEDS="4 5" bash scripts/run_bench.sh      # resumable, skips completed logs
```

That is independent of the ablation and uses the frozen SOTA configs, so it can run
after it on the same pod.
