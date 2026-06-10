"""PC actor-critic: predictive coding policy gradient with a value head.

Unlike PC-REINFORCE (Monte Carlo returns with a batch-mean baseline), this
trains *two* predictive coding networks with jpc and bootstraps from the
critic instead of using Monte Carlo returns:

  - critic (value head): PCN regression on one-step TD targets
        y_V = r + gamma * (1 - done) * V(s')
    trained natively with jpc (PC inference + local weight updates on an
    MSE output layer).
  - actor (policy): PCN trained with advantage-weighted output targets
        y_pi = logits + target_scale * A * (onehot(a) - pi),
    where A = y_V - V(s) is the TD error, so the PC output error equals the
    advantage actor-critic gradient w.r.t. the logits.

No backprop is used for either network. Follows the repo's Config + main()
convention so scripts/run_train.py can dispatch to it.
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
    experiment_name = 'pc_actor_critic'
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
    gamma = 0.99
    learning_rate = 1e-2           # actor (policy) PC parameter optimiser
    value_learning_rate = 1e-2     # critic (value) PC parameter optimiser
    target_scale = 1.0             # scale of the advantage-weighted policy target
    pc_steps_per_update = 1        # jpc.make_pc_step calls per net per batch
    max_t1 = 20                    # jpc inference (activity relaxation) horizon

    # PCN architectures (jpc.make_mlp)
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
        log_path = f'./training_logs/pc_actor_critic/{run_name}'
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
    key, key_policy, key_value, key_envs, eval_key = jr.split(key, 5)

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

    # actor: PCN over logits
    policy_model = jpc.make_mlp(
        key_policy,
        input_dim=obs_dim,
        width=Config.width,
        depth=Config.depth,
        output_dim=action_size,
        act_fn=Config.act_fn,
        use_bias=True,
    )
    if Config.policy_init_logit_bias is not None:
        policy_model = _set_final_layer(policy_model, Config.policy_init_logit_bias)

    # critic: PCN regression head V(s)
    value_model = jpc.make_mlp(
        key_value,
        input_dim=obs_dim,
        width=Config.width,
        depth=Config.depth,
        output_dim=1,
        act_fn=Config.act_fn,
        use_bias=True,
    )

    policy_optim = optax.adam(Config.learning_rate)
    policy_opt_state = policy_optim.init((eqx.filter(policy_model, eqx.is_array), None))
    value_optim = optax.adam(Config.value_learning_rate)
    value_opt_state = value_optim.init((eqx.filter(value_model, eqx.is_array), None))

    @eqx.filter_jit
    def pcn_forward(model, obs):
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
        obs_buf, act_buf, logit_buf, rew_buf, done_buf, next_obs_buf = [], [], [], [], [], []

        for _ in range(Config.unroll_length):
            obs = _flat_obs(env_state.obs)
            logits = pcn_forward(policy_model, jnp.asarray(obs))
            key, key_act = jr.split(key)
            actions = jr.categorical(key_act, logits)
            nstate = envs.step(np.asarray(actions))
            obs_buf.append(obs)
            act_buf.append(np.asarray(actions))
            logit_buf.append(np.asarray(logits))
            rew_buf.append(nstate.reward)
            done_buf.append(nstate.done)
            next_obs_buf.append(_flat_obs(nstate.obs))
            env_state = nstate

        observations = jnp.concatenate([jnp.asarray(o) for o in obs_buf])        # (B, obs)
        next_observations = jnp.concatenate([jnp.asarray(o) for o in next_obs_buf])
        actions = jnp.concatenate([jnp.asarray(a) for a in act_buf])             # (B,)
        logits = jnp.concatenate([jnp.asarray(l) for l in logit_buf])            # (B, A)
        rewards = jnp.concatenate([jnp.asarray(r) for r in rew_buf])             # (B,)
        dones = jnp.concatenate([jnp.asarray(d) for d in done_buf])              # (B,)

        # critic: one-step TD targets (bootstrapped, no Monte Carlo returns)
        values = pcn_forward(value_model, observations).squeeze(-1)              # V(s)
        next_values = pcn_forward(value_model, next_observations).squeeze(-1)    # V(s')
        td_targets = rewards + Config.gamma * (1.0 - dones) * next_values
        advantages = td_targets - values                                          # TD error

        # actor: advantage-weighted PC output targets (epsilon = A2C logit gradient)
        pi = jax.nn.softmax(logits)
        onehot = jax.nn.one_hot(actions, action_size)
        policy_targets = logits + Config.target_scale * advantages[:, None] * (onehot - pi)

        for _ in range(Config.pc_steps_per_update):
            value_result = jpc.make_pc_step(
                model=value_model,
                optim=value_optim,
                opt_state=value_opt_state,
                output=td_targets[:, None],
                input=observations,
                max_t1=Config.max_t1,
            )
            value_model, value_opt_state = value_result["model"], value_result["opt_state"]

            policy_result = jpc.make_pc_step(
                model=policy_model,
                optim=policy_optim,
                opt_state=policy_opt_state,
                output=policy_targets,
                input=observations,
                max_t1=Config.max_t1,
            )
            policy_model, policy_opt_state = policy_result["model"], policy_result["opt_state"]

        global_step += env_step_per_training_step
        metrics = {
            'training/total_steps': global_step,
            'training/updates': training_step,
            'training/walltime': np.round(time.time() - start_time, 3),
            'training/update_time': np.round(time.time() - update_time_start, 3),
            'training/policy_pc_loss': float(policy_result['loss']),
            'training/value_pc_loss': float(value_result['loss']),
            'training/mean_value': float(values.mean()),
            'training/mean_reward': float(rewards.mean()),
            'training/mean_advantage_abs': float(jnp.abs(advantages).mean()),
        }
        logging.info(metrics)

        if Config.eval_env and training_step % Config.eval_every == 0:
            eval_state = eval_env.reset()
            obs = _flat_obs(eval_state.obs)
            eval_logits = pcn_forward(policy_model, jnp.asarray(obs))
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
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, f"{run_name}_policy.eqx"), policy_model)
        eqx.tree_serialise_leaves(
            os.path.join(checkpoint_dir, f"{run_name}_value.eqx"), value_model)
        print(f"models saved to {checkpoint_dir}/{run_name}_*.eqx")

    envs.close()


if __name__ == "__main__":
    main(None)
