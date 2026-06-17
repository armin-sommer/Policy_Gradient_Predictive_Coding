"""Verify the Brax MuJoCo env wrapper (MujocoVecEnv / MujocoEvalEnv).

  env  (needs brax)  : reset/step shapes, continuous actions, done, eval episodes

Runs on a machine with brax + a JAX backend (the A100 box). Auto-skips if brax
is not installed. Usage:
    python scripts/verify_mujoco.py
    python scripts/verify_mujoco.py --env hopper
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def check_env(env_name: str, num_envs: int = 8, episode_length: int = 50) -> bool:
    """MujocoVecEnv: obs/action/reward/done sanity + eval episode accounting."""
    try:
        import brax  # noqa: F401
    except ImportError:
        print("  [env] SKIP - brax not installed (run this on the GPU box)")
        return True

    from env import make_vec_env
    from utils.utils import EnvConfig

    cfg = EnvConfig(env_name=env_name, num_envs=num_envs, episode_length=episode_length)
    env = make_vec_env(cfg)
    env.seed(0)
    state = env.reset()

    obs_dim = env.observation_size
    act_dim = env.action_space.n
    obs_ok = state.obs.shape == (num_envs, obs_dim) and state.obs.dtype == np.float32
    cont_ok = getattr(env.action_space, "continuous", False)
    print(f"  [env] reset obs    : {'ok' if obs_ok else 'BAD'}  {state.obs.shape} {state.obs.dtype} (obs_dim={obs_dim})")
    print(f"  [env] action space : {'ok' if cont_ok else 'BAD'}  continuous, act_dim={act_dim}")

    rng = np.random.default_rng(0)
    done_seen, reward_finite, done_binary = False, True, True
    # episode_length is short, so truncation-driven done should appear quickly.
    for _ in range(episode_length + 20):
        actions = rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32)
        nstate = env.step(actions)
        if not np.isfinite(nstate.reward).all():
            reward_finite = False
        if ((nstate.done != 0) & (nstate.done != 1)).any():
            done_binary = False
        if nstate.done.any():
            done_seen = True
        state = nstate
        if done_seen:
            break
    print(f"  [env] done flips   : {'ok' if done_seen else f'NO EPISODE ENDED in {episode_length + 20} steps'}")
    print(f"  [env] rewards finite: {'ok' if reward_finite else 'NON-FINITE'}")
    print(f"  [env] done in {{0,1}}: {'ok' if done_binary else 'BAD'}")

    eval_env = make_vec_env(cfg, evaluate=True)
    eval_env.seed(1)
    eval_env.reset()
    for _ in range(2 * episode_length + 20):
        eval_env.step(rng.uniform(-1.0, 1.0, size=(num_envs, act_dim)).astype(np.float32))
    returns, lengths = eval_env.evaluate()
    eval_ok = len(returns) > 0 and np.isfinite(returns).all() and all(l > 0 for l in lengths)
    print(f"  [env] eval episodes: {'ok' if eval_ok else 'BAD'}  "
          f"n={len(returns)} mean_return={np.mean(returns) if returns else float('nan'):.3f}")

    return obs_ok and cont_ok and done_seen and reward_finite and done_binary and eval_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah")
    args = parser.parse_args()

    print(f"\n=== env ({args.env}) ===")
    try:
        ok = check_env(args.env)
    except Exception as e:  # surface the failure
        print(f"  [env] ERROR: {type(e).__name__}: {e}")
        ok = False

    print("\n=== summary ===")
    print(f"  env : {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
