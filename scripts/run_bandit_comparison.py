"""Policy gradient methods on the 2-armed bandit.

Runs REINFORCE (vanilla policy gradient, SGD), TRPO (natural policy gradient
+ line search), Cleanba PPO (clipped surrogate, Adam), and PC-REINFORCE
(predictive-coding-trained policy via jpc) on the 2-armed bandit with an
adversarial initialization (pi(optimal arm) ~ 2%), tracks pi(optimal arm) per
update via stochastic evaluation, and writes a comparison plot.

Why TRPO always wins here: with a softmax policy the vanilla PG gradient on
the logit gap is pi*(1-pi)*gap, which vanishes at the adversarial init, while
NPG preconditions with the inverse Fisher (= 1/(pi*(1-pi))) and makes constant
progress in logit space per update.

Usage:
    python scripts/run_bandit_comparison.py --seed 0
    python scripts/run_bandit_comparison.py --algos reinforce trpo --seed 0
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ARM_MEANS = (1.0, 0.9)  # arm 0 optimal, gap = 0.1
ADVERSARIAL_LOGIT_BIAS = [0.0, 4.0]  # pi(arm 0) = sigmoid(-4) ~ 1.8%

ALGO_MODULES = {
    "reinforce": "backprop_algorithms.reinforce",
    "trpo": "backprop_algorithms.trpo",
    "cleanba_ppo": "backprop_algorithms.cleanba_ppo",
    "pc_reinforce": "pc_algorithms.pc_reinforce",
}

LABELS = {
    "reinforce": "REINFORCE (vanilla PG, SGD)",
    "trpo": "TRPO (natural PG)",
    "cleanba_ppo": "Cleanba PPO (Adam)",
    "pc_reinforce": "PC-REINFORCE (jpc)",
}


class MetricsCapture(logging.Handler):
    """Collects (env_steps, eval/mean_score) pairs from the algorithms' logs."""

    def __init__(self):
        super().__init__()
        self.eval_points = []
        self._last_step = 0

    def emit(self, record):
        msg = record.msg
        if not isinstance(msg, dict):
            return
        if 'training/total_steps' in msg:
            self._last_step = int(msg['training/total_steps'])
        if 'eval/mean_score' in msg:
            self.eval_points.append((self._last_step, float(msg['eval/mean_score'])))


def configure_bandit(Config, seed, total_timesteps):
    Config.env_name = 'bandit'
    Config.arm_means = ARM_MEANS
    Config.deterministic_rewards = True
    Config.use_cnn = False
    Config.policy_hidden_layer_sizes = ()
    Config.value_hidden_layer_sizes = ()
    Config.policy_init_logit_bias = ADVERSARIAL_LOGIT_BIAS
    Config.total_timesteps = total_timesteps
    Config.anneal_lr = False
    Config.entropy_cost = 0.0
    Config.eval_env = True
    Config.eval_every = 1
    Config.num_eval_episodes = 200
    Config.deterministic_eval = False  # stochastic eval => mean score is linear in pi(opt)
    Config.save_model = False
    Config.seed = seed
    Config.gamma = 0.99


def run_algo(algo, seed, total_timesteps, log_dir=None):
    module = importlib.import_module(ALGO_MODULES[algo])
    Config = module.Config
    configure_bandit(Config, seed, total_timesteps)

    if algo == "reinforce":
        Config.num_envs = 1
        Config.optimizer = 'sgd'  # textbook vanilla PG
        Config.learning_rate = 0.5
    elif algo == "trpo":
        Config.num_envs = 8
        Config.unroll_length = 250
        Config.batch_size = 1
        Config.num_minibatches = 8
        Config.update_epochs = 10
        Config.target_kl = 0.01
        Config.learning_rate = 1e-2  # value-function optimizer only
    elif algo == "cleanba_ppo":
        Config.num_envs = 8
        Config.unroll_length = 250
        Config.num_minibatches = 4
        Config.update_epochs = 4
        Config.clip_eps = 0.1
        Config.learning_rate = 2.5e-4
    elif algo == "pc_reinforce":
        Config.num_envs = 8
        Config.unroll_length = 250
        Config.learning_rate = 1e-2
        Config.target_scale = 1.0

    capture = MetricsCapture()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    file_handler = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{algo}.log"), mode="w")
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        root.addHandler(file_handler)
    try:
        module.main(None)
    finally:
        root.removeHandler(capture)
        if file_handler is not None:
            root.removeHandler(file_handler)
            file_handler.close()
    return capture.eval_points


def to_pi_optimal(eval_points):
    """Mean stochastic-eval score -> pi(optimal arm)."""
    gap = ARM_MEANS[0] - ARM_MEANS[1]
    steps = np.array([p[0] for p in eval_points])
    pi = np.clip((np.array([p[1] for p in eval_points]) - ARM_MEANS[1]) / gap, 0.0, 1.0)
    return steps, pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=60_000)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Defaults to results/bandit_seed{seed}/ (tracked in git).")
    parser.add_argument("--algos", nargs="*", default=list(ALGO_MODULES),
                        choices=list(ALGO_MODULES))
    args = parser.parse_args()

    out_dir = args.out_dir or str(REPO_ROOT / "results" / f"bandit_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for algo in args.algos:
        print(f"\n=== running {algo} on the 2-armed bandit ===\n")
        results[algo] = run_algo(algo, args.seed, args.total_steps, log_dir=out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, points in results.items():
        steps, pi = to_pi_optimal(points)
        ax.plot(steps, pi, marker='o', markersize=3, label=LABELS[algo])
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel("env steps")
    ax.set_ylabel(r"$\pi$(optimal arm)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"2-armed bandit, adversarial init $\\pi_0 \\approx 0.018$ (seed {args.seed})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"bandit_npg_vs_pg_seed{args.seed}.png")
    fig.savefig(out_path, dpi=150)

    csv_path = os.path.join(out_dir, "pi_optimal.csv")
    with open(csv_path, "w") as f:
        f.write("algo,env_steps,pi_optimal\n")
        for algo, points in results.items():
            steps, pi = to_pi_optimal(points)
            for s, p in zip(steps, pi):
                f.write(f"{algo},{int(s)},{p:.6f}\n")

    print(f"\nplot saved to {out_path}")
    print(f"per-algo logs and {os.path.basename(csv_path)} in {out_dir}\n")
    summary_lines = []
    for algo, points in results.items():
        steps, pi = to_pi_optimal(points)
        final_pi = pi[-1] if len(pi) else float('nan')
        avg_pi = np.mean(pi) if len(pi) else float('nan')  # evals are evenly spaced
        summary_lines.append(
            f"{LABELS[algo]:<32} final pi(opt) = {final_pi:.3f}   avg pi(opt) = {avg_pi:.3f}")
    print("\n".join(summary_lines))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"seed={args.seed} total_steps={args.total_steps}\n")
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
