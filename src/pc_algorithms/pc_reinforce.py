"""PC-REINFORCE: policy gradient with a predictive-coding-trained policy.

The policy is a predictive coding network (PCN) built and trained with
jpc (https://github.com/thebuckleylab/jpc). Instead of backpropagating the
REINFORCE loss, each update constructs *advantage-weighted output targets*

    y = logits + target_scale * A(a) * (onehot(a) - pi)

so that the PCN's output-layer error epsilon = y - logits equals the policy
gradient of the REINFORCE objective w.r.t. the logits. jpc then (1) relaxes
the network activities to equilibrium via its inference dynamics and
(2) updates the weights with local PC rules at that equilibrium
(jpc.make_pc_step) -- no end-to-end backprop through the policy.

Follows the repo's Config + main() convention so scripts/run_train.py can
dispatch to it.
"""

import logging
import os
import random
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import jpc

from env import make_vec_env
from utils.utils import EnvConfig


class Config:
    # experiment
    experiment_name = 'pc_reinforce'
    seed = 1
    write_logs_to_file = False
    save_model = False

    # environment (bandit; any discrete env with flat obs works)
    env_name = 'bandit'
    num_envs = 8
    num_train_levels = 200
    distribution_mode = 'easy'
    arm_means = (1.0, 0.9)
    deterministic_rewards = True

    # eval
    eval_env = True
    num_eval_episodes = 200
    eval_every = 1

    # algorithm hyperparameters
    total_timesteps = 60_000
    unroll_length = 250            # env steps per update (x num_envs samples)
    learning_rate = 1e-2           # optax optimiser for the PC parameter update
    target_scale = 1.0             # scale of the advantage-weighted output target
    pc_steps_per_update = 1        # how many jpc.make_pc_step calls per batch
    max_t1 = 20                    # jpc inference (activity relaxation) horizon

    # PCN architecture (jpc.make_mlp)
    width = 32
    depth = 2
    act_fn = 'relu'
    policy_init_logit_bias = None  # e.g. [0.0, 4.0] for the bandit plateau init


def _set_final_layer(model, logit_bias):
    """Zero the PCN's final Linear kernel and set its bias (adversarial init)."""
    logit_bias = jnp.asarray(logit_bias, dtype=jnp.float32)
    final_linear = model[-1].layers[1]
    if final_linear.bias is None:
        raise ValueError("policy_init_logit_bias requires use_bias=True in the PCN")
    if final_linear.bias.shape != logit_bias.shape:
        raise ValueError(
            f"logit_bias shape {logit_bias.shape} != logits shape {final_linear.bias.shape}")
    model = eqx.tree_at(lambda m: m[-1].layers[1].weight, model,
                        jnp.zeros_like(final_linear.weight))
    model = eqx.tree_at(lambda m: m[-1].layers[1].bias, model, logit_bias)
    return model


def main(_):
    run_name = f"Exp_{Config.experiment_name}__{Config.env_name}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        log_path = f'./training_logs/pc_reinforce/{run_name}'
        os.makedirs(log_path, exist_ok=True)
        logging.getLogger().addHandler(
            logging.FileHandler(os.path.join(log_path, 'logs')))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    logging.info("|param: value|")
    for k, v in vars(Config).items():
        if not k.startswith('__'):
            logging.info(f"|{k}:  {v}|")

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    key = jr.PRNGKey(Config.seed)
    key, key_model, key_envs, eval_key = jr.split(key, 4)

    env_cfg = EnvConfig(
        env_name=Config.env_name,
        num_envs=Config.num_envs,
        num_train_levels=Config.num_train_levels,
        distribution_mode=Config.distribution_mode,
        arm_means=tuple(Config.arm_means),
        deterministic_rewards=Config.deterministic_rewards,
    )
    envs = make_vec_env(env_cfg)
    envs.seed(int(key_envs[0]))
    env_state = envs.reset()

    action_size = envs.action_space.n
    obs_dim = int(np.prod(env_state.obs.shape[1:]))

    # PCN policy: list of equinox layers trained with predictive coding
    model = jpc.make_mlp(
        key_model,
        input_dim=obs_dim,
        width=Config.width,
        depth=Config.depth,
        output_dim=action_size,
        act_fn=Config.act_fn,
        use_bias=True,
    )
    if Config.policy_init_logit_bias is not None:
        model = _set_final_layer(model, Config.policy_init_logit_bias)

    optim = optax.adam(Config.learning_rate)
    opt_state = optim.init((eqx.filter(model, eqx.is_array), None))

    @eqx.filter_jit
    def policy_logits(model, obs):
        activities = jpc.init_activities_with_ffwd(model=model, input=obs)
        return activities[-1]

    def _flat_obs(obs):
        return envs.normalize_obs(
            np.asarray(obs).reshape(obs.shape[0], -1).astype(np.float32))

    if Config.eval_env:
        eval_cfg = EnvConfig(
            env_name=Config.env_name,
            num_envs=Config.num_eval_episodes,
            num_train_levels=Config.num_train_levels,
            distribution_mode=Config.distribution_mode,
            arm_means=tuple(Config.arm_means),
            deterministic_rewards=Config.deterministic_rewards,
        )
        eval_env = make_vec_env(eval_cfg, evaluate=True)
        eval_env.seed(int(eval_key[0]))

    env_step_per_training_step = Config.num_envs * Config.unroll_length
    num_training_steps = int(np.ceil(Config.total_timesteps / env_step_per_training_step))

    global_step = 0
    start_time = time.time()

    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()
        obs_buf, act_buf, logit_buf, rew_buf = [], [], [], []

        for _ in range(Config.unroll_length):
            obs = _flat_obs(env_state.obs)
            logits = policy_logits(model, jnp.asarray(obs))
            key, key_act = jr.split(key)
            actions = jr.categorical(key_act, logits)
            nstate = envs.step(np.asarray(actions))
            obs_buf.append(obs)
            act_buf.append(np.asarray(actions))
            logit_buf.append(np.asarray(logits))
            rew_buf.append(nstate.reward)
            env_state = nstate

        observations = jnp.concatenate([jnp.asarray(o) for o in obs_buf])      # (B, obs)
        actions = jnp.concatenate([jnp.asarray(a) for a in act_buf])           # (B,)
        logits = jnp.concatenate([jnp.asarray(l) for l in logit_buf])          # (B, A)
        rewards = jnp.concatenate([jnp.asarray(r) for r in rew_buf])           # (B,)

        # REINFORCE with mean-reward baseline (1-step episodes => return = reward)
        advantages = rewards - rewards.mean()

        # Advantage-weighted PC output targets: epsilon = y - logits is exactly
        # the REINFORCE gradient w.r.t. the logits.
        pi = jax.nn.softmax(logits)
        onehot = jax.nn.one_hot(actions, action_size)
        targets = logits + Config.target_scale * advantages[:, None] * (onehot - pi)

        for _ in range(Config.pc_steps_per_update):
            result = jpc.make_pc_step(
                model=model,
                optim=optim,
                opt_state=opt_state,
                output=targets,
                input=observations,
                max_t1=Config.max_t1,
            )
            model, opt_state = result["model"], result["opt_state"]

        global_step += env_step_per_training_step
        metrics = {
            'training/total_steps': global_step,
            'training/updates': training_step,
            'training/walltime': np.round(time.time() - start_time, 3),
            'training/update_time': np.round(time.time() - update_time_start, 3),
            'training/pc_loss': float(result['loss']),
            'training/mean_reward': float(rewards.mean()),
            'training/mean_advantage_abs': float(jnp.abs(advantages).mean()),
        }
        logging.info(metrics)

        if Config.eval_env and training_step % Config.eval_every == 0:
            eval_state = eval_env.reset()
            obs = _flat_obs(eval_state.obs)
            eval_logits = policy_logits(model, jnp.asarray(obs))
            key, key_eval = jr.split(key)
            eval_actions = jr.categorical(key_eval, eval_logits)
            eval_env.step(np.asarray(eval_actions))
            eval_returns, eval_ep_lengths = eval_env.evaluate()
            eval_metrics = {
                'eval/num_episodes': len(eval_returns),
                'eval/mean_score': np.round(np.mean(eval_returns), 4),
                'eval/std_score': np.round(np.std(eval_returns), 4),
                'eval/mean_episode_length': np.mean(eval_ep_lengths),
            }
            logging.info(eval_metrics)

    logging.info('TRAINING END: training duration: %s', time.time() - start_time)

    if Config.save_model:
        checkpoint_dir = getattr(Config, "checkpoint_dir", "weights")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"{run_name}.eqx")
        eqx.tree_serialise_leaves(model_path, model)
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    main(None)
