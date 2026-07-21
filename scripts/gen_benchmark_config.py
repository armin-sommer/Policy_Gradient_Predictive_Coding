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
    a = ap.parse_args()

    base = ROOT / "configs" / f"mujoco_halfcheetah_{a.algo}_{a.tier}.yaml"
    c = yaml.safe_load(base.read_text())
    c["train"]["optimizer"] = a.opt
    c["train"]["learning_rate"] = a.lr
    c["train"]["target_scale"] = a.ts
    c["agent"]["act_fn"] = a.act
    if a.max_t1 is not None:
        c["train"]["max_t1"] = a.max_t1

    ts = "ts" + str(a.ts).replace(".", "").ljust(2, "0")[:2]        # 0.5 -> ts05
    lr = "" if abs(a.lr - 3e-4) < 1e-12 else "_lr" + f"{a.lr:g}".replace(".", "")
    mt = "" if a.max_t1 is None else f"_mt{a.max_t1}"
    name = f"halfcheetah_{a.algo}_{a.opt}_{a.act}_{ts}_{a.tier}{lr}{mt}"
    c["agent"]["experiment_name"] = name.replace("halfcheetah_", "")

    out = ROOT / "configs" / "benchmark" / f"{name}.yaml"
    out.write_text(yaml.safe_dump(c, sort_keys=False))
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
