"""Stamp out a benchmark sweep config into configs/benchmark/, overriding only
optimizer / activation / learning_rate / target_scale on a validated tier config.

    python scripts/gen_benchmark_config.py --algo pc_reinforce --tier sota --opt adam --act tanh --ts 0.5
      -> configs/benchmark/halfcheetah_pc_reinforce_adam_tanh_ts05_sota.yaml

Envs / rollout_length / eval_every / width are inherited from
configs/mujoco_halfcheetah_{algo}_{tier}.yaml, so per-algo tier settings stay
correct (pc_reinforce uses long rollouts + fewer envs; pc_actor_critic short
rollouts + more envs). Only the four sweep knobs change.
"""

import argparse
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["pc_actor_critic", "pc_reinforce"])
    ap.add_argument("--tier", default="sota", choices=["bench", "sota"])
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--act", default="tanh", choices=["relu", "tanh"])
    ap.add_argument("--ts", type=float, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-t1", type=int, default=None, help="PC inference steps (default: tier's)")
    ap.add_argument("--state-indep-std", action="store_true",
                    help="global (state-independent) log_std, like SOTA PPO/TRPO")
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="global-norm clip on the PC policy gradient (default: off)")
    ap.add_argument("--target-clip", type=float, default=None,
                    help="cap on the mean-target offset (output-space trust region)")
    ap.add_argument("--target-clip-rel", action="store_true",
                    help="make --target-clip relative: |loc_target-mu| <= clip*sigma")
    ap.add_argument("--log-std-min", type=float, default=None,
                    help="raise the log_std floor (e.g. -1 -> sigma_min 0.37)")
    ap.add_argument("--natural-target", action="store_true",
                    help="natural-gradient target (drop the 1/sigma^2 amplifier)")
    a = ap.parse_args()

    base = ROOT / "configs" / f"mujoco_halfcheetah_{a.algo}_{a.tier}.yaml"
    c = yaml.safe_load(base.read_text())
    c["train"]["optimizer"] = a.opt
    c["train"]["learning_rate"] = a.lr
    c["train"]["target_scale"] = a.ts
    c["agent"]["act_fn"] = a.act
    if a.max_t1 is not None:
        c["train"]["max_t1"] = a.max_t1
    if a.state_indep_std:
        c["agent"]["state_indep_std"] = True
    if a.max_grad_norm is not None:
        c["train"]["max_grad_norm"] = a.max_grad_norm
    if a.target_clip is not None:
        c["train"]["target_clip"] = a.target_clip
        c["train"]["target_clip_rel"] = a.target_clip_rel
    if a.log_std_min is not None:
        c["agent"]["log_std_min"] = a.log_std_min
    if a.natural_target:
        c["train"]["natural_target"] = True

    ts = "ts" + str(a.ts).replace(".", "").ljust(2, "0")[:2]        # 0.5 -> ts05
    lr = "" if abs(a.lr - 3e-4) < 1e-12 else "_lr" + f"{a.lr:g}".replace(".", "")
    mt = "" if a.max_t1 is None else f"_mt{a.max_t1}"
    sistd = "_stdglobal" if a.state_indep_std else ""
    clip = "" if a.max_grad_norm is None else "_clip" + f"{a.max_grad_norm:g}".replace(".", "")
    tclip = "" if a.target_clip is None else ("_tclip" + ("rel" if a.target_clip_rel else "")
                                              + f"{a.target_clip:g}".replace(".", ""))
    smin = "" if a.log_std_min is None else "_smin" + f"{a.log_std_min:g}".replace(".", "").replace("-", "m")
    nat = "_nat" if a.natural_target else ""
    name = f"halfcheetah_{a.algo}_{a.opt}_{a.act}_{ts}_{a.tier}{lr}{mt}{sistd}{clip}{tclip}{smin}{nat}"
    c["agent"]["experiment_name"] = name.replace("halfcheetah_", "")

    out = ROOT / "configs" / "benchmark" / f"{name}.yaml"
    out.write_text(yaml.safe_dump(c, sort_keys=False))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
