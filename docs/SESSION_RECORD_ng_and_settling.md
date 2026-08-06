# Session record — natural-gradient probes, the settling fix, and the GPU handoff

Complete record of one working session on `fixing-pcpg`. The findings themselves live
in `docs/PCPG_GEOMETRY_FINDINGS.md` (narrative) and
`docs/PCPG_NG_experiments.pdf` (report). This file records what those two do **not**:
the correction trail, the infrastructure blockers, and an index of everything added, so
none of it has to be re-derived or re-litigated.

Session commits: `8fe65f1` … `738ab0c` on `fixing-pcpg`, plus `ada9f2c` on
`claude/pcpg-fix-modal-experiments-t6tt73` (superseded — ignore that branch).

---

## 1. What was asked, and what happened

The session began as "run the MuJoCo experiments on Modal". Modal turned out to be
unreachable, so the work moved to what could be done without a GPU: measuring the
natural-gradient claim, finding and fixing a bug that invalidated the way it had been
measured, and preparing the GPU runs for someone who can execute them.

**No GPU experiment was run this session.** Everything measured here is CPU: the
geometry probes and one bandit ablation.

---

## 2. Findings, in one line each

| # | Finding | Where |
|---|---|---|
| F6 | With σ state-independent there is almost no NG geometry to get right — `cos(d_SGD, d_NG)` is 0.941 fixed vs 0.341 at the σ floor. It is the right *control*, the wrong *test case*. | `PCPG_NATURAL_GRADIENT_PROBES.md` |
| F7 | The `log_std` clamp binds through the **target**, not through where σ sits: 80.9% of targets truncated at bench `ts10` with σ ≡ 1 mid-window. F5's stated mechanism was misattributed. | same |
| — | Innocenti's trust-region rescaling is **inert** here: `cond(S)` 1.5–1.8, so `cos(d_BP, d_EQ) ≈ 0.99` at every depth and in linear nets. It rescales step length, not direction. | `PCPG_GEOMETRY_FINDINGS.md` §4b |
| — | **`d_EQ` cannot test the paper's saddle-escape claim** — `S → I` as weights approach the origin saddle, so the instrument goes inert exactly where the theory predicts the largest effect. | §4b, flagged |
| — | Production inference ran at **τ = t1/N ≈ 0.01, ~1% settled**; convergence needs τ ≈ 10. Three compounding causes, all in jpc. Fixed behind `train.inference_rate_correction`. | §4d |
| — | Settling **slows** bandit learning: +3661 ± 407 steps to π≥0.9, t(4)=+9.00. So "inference is inert" is geometry-specific and does not transfer. | §4e |
| — | Experiment-log §3.4's σ-parameterisation conclusion is **confounded**: no trust region is enforced anywhere (`policy_kl` is diagnostic-only, `kl_max` reached 170 vs TRPO's 0.01), and the `log_std` channel was ~81% truncated. | `PCPG_EXPERIMENT_LOG.md` §3.4 |

---

## 3. Corrections and retractions — the full trail

Recorded because each was caught by measurement rather than reasoning, and the pattern
matters more than the individual errors.

**3.1 Over-generalised "inference is inert".** §4b measured one geometry (Gaussian,
17→64→12) and I predicted the settling ablation would therefore change nothing. The
bandit ablation falsified that. Cause: `cos(d_BP, d_EQ) ≈ 0.99` in the Gaussian
geometry but ≈ 0.78 in the bandit's (discrete softmax, 1→32→2). The finding was real;
its scope was not. Fixed in `610c909`.

**3.2 Cited `d_EQ` against the trust-region result.** `d_EQ` encodes a local
direction-rescaling reading of the theorem. The paper's headline claim is saddle
escape, which is a landscape/dynamics property, and `d_EQ` provably goes inert at a
saddle (`cond(S)` → 1.00 as ε → 0.01). F3/F4 stand as "inference supplies no
NG/output-metric content" and must not be cited against the paper. Fixed in `c7df107`.

**3.3 Three successive wrong cost claims for settled inference.** Worth reading as one
story:

| claim | why it was wrong |
|---|---|
| "~80× slower, inherent" | missing `@eqx.filter_jit` on `make_pc_step_at_rate`, re-tracing the whole solve every call (`jpc.make_pc_step` is itself jitted). A real bug, really fixed. |
| "effectively free" | two bad measurements at once — a bandit figure from a tiny net where inference is not the bottleneck, plus a **non-jitted** bench-scale call whose wall-clock was tracing-dominated |
| **"~9.5× per PC step"** (current) | jitted, at the real bench N=2048: 0.045 → 0.428 s/update, corroborated by solver step counts 5 → 62 (39 at `max_t1=10`) |

Evidence that should have caught this earlier and did not: the committed
`mt10/20/40/80` runs all took the *same* wall-clock (403/407/406/412 s), which is
itself a symptom of never settling. Fixed in `91d774e` then `047d4bc`.

**3.4 Statistics too liberal at small n.** The ablation runner's original 2·SEM rule
flagged a `pc_actor_critic` final-π difference as significant when `t(4) = −2.30` is
below the 2.776 critical value. Replaced with a real paired t-test plus CI. Separately,
final-π had **no power** on the bandit (both arms saturated at π ≈ 0.98) — the
steps-to-threshold metric is what showed the effect. Fixed in `610c909`.

**3.5 Claimed the repo had no interp probes or Gaussian PC path.** True of `main`,
false of `fixing-pcpg`, which is where all of it lives. I was reading the wrong branch.

---

## 4. Infrastructure blockers — do not re-litigate these

All three were diagnosed to root cause. None is a credentials problem.

**Modal — unreachable, and no setting fixes it.** `create_channel_with_fallbacks` fails
in ~0.05 s, before any auth exchange, so the token is never tested. Cause: Modal's
client is gRPC-only and `grpclib` has **zero** proxy support (no `HTTPS_PROXY`, no
`CONNECT` handling anywhere in the installed package), while all cloud-sandbox egress
goes through an HTTP CONNECT proxy. Adding `api.modal.com` to an allowlist cannot help,
because the client never speaks to the proxy that enforces the allowlist — `Full`
network access would fail identically. `curl https://api.modal.com` returning 200 is
the trap: plain HTTPS goes through the proxy, gRPC does not.

**RunPod SSH — unreachable.** Three independent blockers: no `ssh` client installed in
the sandbox; `~/.ssh/id_ed25519` does not exist here (it is on the user's machine, and
cloud sessions do not carry local keys); and TCP to the pod's high port times out while
`:443` to the same host is open — the non-443-port restriction the proxy documents.

**The Innocenti paper — unreadable.** arxiv.org, ar5iv, openreview.net,
api.semanticscholar.org and huggingface.co all fail at the proxy with `CONNECT tunnel
failed, response 403`. Only `WebSearch` reaches out. So every statement about
arXiv:2305.18188 in these documents rests on its **title and abstract** plus this
repo's own citations, never on the theorem statement. That limit is stated in the
documents themselves.

**Consequence:** the GPU work is handed off, not done. `HANDOFF.md` covers the Modal
route (for a Remote Control session on the user's machine);
`RUNPOD_SETTLING_ABLATION.md` covers the RunPod route.

---

## 5. What is built and unrun

| experiment | runner | status |
|---|---|---|
| MuJoCo settling ablation | `scripts/run_settling_ablation_mujoco.py` | **unrun** — needs GPU. The open question. |
| Cost calibration on real hardware | same, `--calibrate` | **unrun** — do this first; it replaces the CPU-extrapolated 9.5× |
| Backprop baseline seeds 4–5 | `scripts/run_bench.sh`, `modal_app.py` | **unrun** — 8 cells, `BENCHMARK_PROTOCOL.md` §7 wants 5 seeds, only 1–3 exist |
| Enforced trust region (KL cap / `target_clip`) | — | **not built.** Top blocker: no σ or geometry conclusion has a valid arm until it exists |
| Fixed-point inference solve | — | **not built.** Worth doing now that settling is known to cost ~9.5× |
| Probe 2 (real-checkpoint geometry) | — | specified, not built. The one route by which §4b could soften |
| Saddle-escape test (PC vs BP under plain GD) | — | **not built.** The only thing that would test the paper on its own terms |

---

## 6. Artifact index

**Source**
- `src/pc_algorithms/inference.py` — the settling fix (`make_pc_step_at_rate`,
  `settle_activities`, `inference_residual`); drop-in for `jpc.make_pc_step`
- `pc_actor_critic.py` / `pc_reinforce.py` — `inference_rate_correction` flag,
  default `False` in code so committed results still reproduce
- `run_train.py` — `KEY_MAP` entry; `gen_benchmark_config.py` — `--unsettled` plus a
  name tag derived from the *effective* value (a CLI-only tag would have mislabelled runs)
- all 10 `configs/mujoco_*_pc_*.yaml` — `inference_rate_correction: true`

**Probes and runners** (all CPU except the last)
- `probe_natural_gradient.py` — the NG probe; `fixed` regime added
- `probe_trust_region_metric.py` — `S` spectrum, `cos(d_BP, d_EQ)`, depth sweep
- `probe_inference_settling.py` — residual-vs-τ curve and the fixed-point check
- `probe_natgrad_scaling.py` — output-Fisher check and the σ-units correction
  (on the `claude/…` branch)
- `run_settled_ablation.py` — bandit ablation (ran); `run_settling_ablation_mujoco.py`
  — MuJoCo ablation (unrun, GPU)

**Documents**
- `PCPG_GEOMETRY_FINDINGS.md` — the coauthor-facing narrative
- `PCPG_NG_experiments.pdf` (5 pp) and `PCPG_NG_method_1page.pdf` (1 p), both
  regenerable via `scripts/make_ng_report_pdf.py` / `make_ng_method_1page.py`
- `HANDOFF.md`, `RUNPOD_SETTLING_ABLATION.md`, `MODAL.md`
- this file

**Results**
- `results/ng_probe_rerun/` — probe logs (4 regimes, fixed-regime run)
- `results/settled_ablation/` — bandit ablation, 20 runs plus `results.json`

---

## 7. If you read only one thing

The single most consequential fact: **every PCPG result in the experiment log was
produced with inference ~1% settled.** On the bandit, settling changes learning
materially. Whether it changes the MuJoCo conclusions is unknown and is one GPU-day
away. Until that runs, treat the MuJoCo numbers as measurements of a method that was
not doing what the method description says.
