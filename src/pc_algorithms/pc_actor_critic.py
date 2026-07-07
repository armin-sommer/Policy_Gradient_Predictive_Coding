"""PC actor-critic — jpc policy + value head, GAE(lambda) advantages."""

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
from pc_algorithms.gaussian_policy import (
    discrete_pc_targets,
    gaussian_pc_targets,
    sample_gaussian_action,
)
from pc_algorithms.pc_eval import evaluate_discrete_policy, evaluate_gaussian_policy
from pc_algorithms.returns import compute_gae


class Config:
    # experiment
    experiment_name = 'pc_actor_critic'
    seed = 1
    write_logs_to_file = False
    save_model = False

    env_name = 'bandit'
    num_envs = 8
    num_train_levels = 200
    distribution_mode = 'easy'
    arm_means = (1.0, 0.9)
    deterministic_rewards = True
    episode_length = 1000

    # eval
    eval_env = True
    num_eval_episodes = 200
    eval_every = 1

    # algorithm hyperparameters
    total_timesteps = 60_000
    unroll_length = 250
    gamma = 0.99
    # gae_lambda=0 reproduces the original TD(0) advantages; >0 bootstraps
    # multi-step credit through the rollout boundary via the value net.
    gae_lambda = 0.95
    learning_rate = 1e-2
    value_learning_rate = 1e-2
    target_scale = 1.0
    pc_steps_per_update = 1
    # PPO-style data reuse: epochs x minibatches per rollout, policy targets
    # recomputed each minibatch; value regresses fixed lambda-returns.
    # Defaults (1, 1) reproduce the original single full-batch PC step.
    update_epochs = 1
    num_minibatches = 1
    normalize_advantages = False
    max_t1 = 20
    normalize_rewards = False
    exp_std = True

    width = 32
    depth = 2
    act_fn = 'relu'
    policy_init_logit_bias = None


def _set_final_layer(model, logit_bias):
    """Zero final layer kernel and set bias."""
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
        episode_length=Config.episode_length,
    )
    envs = make_vec_env(env_cfg)
    envs.seed(int(key_envs[0]))
    env_state = envs.reset()

    continuous = getattr(envs.action_space, "continuous", False)
    action_size = envs.action_space.n
    policy_output_dim = 2 * action_size if continuous else action_size
    obs_dim = int(np.prod(env_state.obs.shape[1:]))

    policy_model = jpc.make_mlp(
        key_policy,
        input_dim=obs_dim,
        width=Config.width,
        depth=Config.depth,
        output_dim=policy_output_dim,
        act_fn=Config.act_fn,
        use_bias=True,
    )
    if Config.policy_init_logit_bias is not None:
        if continuous:
            raise ValueError("policy_init_logit_bias is only supported for discrete policies")
        policy_model = _set_final_layer(policy_model, Config.policy_init_logit_bias)

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

    def _flat_obs(obs, update=True):
        raw = np.asarray(obs).reshape(obs.shape[0], -1).astype(np.float32)
        try:
            return envs.normalize_obs(raw, update=update)
        except TypeError:
            return envs.normalize_obs(raw)

    if Config.eval_env:
        eval_cfg = EnvConfig(
            env_name=Config.env_name,
            num_envs=Config.num_eval_episodes,
            num_train_levels=Config.num_train_levels,
            distribution_mode=Config.distribution_mode,
            arm_means=tuple(Config.arm_means),
            deterministic_rewards=Config.deterministic_rewards,
            episode_length=Config.episode_length,
        )
        eval_env = make_vec_env(eval_cfg, evaluate=True)
        eval_env.seed(int(eval_key[0]))

    env_step_per_training_step = Config.num_envs * Config.unroll_length
    num_training_steps = int(np.ceil(Config.total_timesteps / env_step_per_training_step))

    global_step = 0
    start_time = time.time()

    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()
        obs_buf, act_buf, pre_tanh_buf = [], [], []
        rew_buf, done_buf, next_obs_buf = [], [], []

        for _ in range(Config.unroll_length):
            obs = _flat_obs(env_state.obs)
            params = pcn_forward(policy_model, jnp.asarray(obs))
            key, key_act = jr.split(key)
            if continuous:
                actions, pre_tanh = sample_gaussian_action(
                    key_act, params, action_size, exp_std=Config.exp_std)
                act_np = np.asarray(actions)
                pre_tanh_buf.append(np.asarray(pre_tanh))
            else:
                actions = jr.categorical(key_act, params)
                act_np = np.asarray(actions)
            nstate = envs.step(act_np)
            reward = nstate.reward
            if Config.normalize_rewards and hasattr(envs, "normalize_reward"):
                reward = envs.normalize_reward(reward, nstate.done, Config.gamma)
            obs_buf.append(obs)
            act_buf.append(act_np)
            rew_buf.append(reward)
            done_buf.append(nstate.done)
            next_obs_buf.append(_flat_obs(nstate.obs, update=False))
            env_state = nstate

        # (T, N, ...) buffers; flatten time-major so rows align with GAE output
        t_steps, n_envs = Config.unroll_length, Config.num_envs
        obs_arr = np.stack(obs_buf)
        next_obs_arr = np.stack(next_obs_buf)
        rewards = np.stack(rew_buf)
        dones = np.stack(done_buf)

        observations = obs_arr.reshape(-1, obs_arr.shape[-1])
        next_observations = next_obs_arr.reshape(-1, next_obs_arr.shape[-1])

        # old value estimates for GAE (computed once, before any update)
        values = np.asarray(
            pcn_forward(value_model, jnp.asarray(observations))).squeeze(-1)
        next_values = np.asarray(
            pcn_forward(value_model, jnp.asarray(next_observations))).squeeze(-1)
        advantages, value_targets = compute_gae(
            rewards, dones,
            values.reshape(t_steps, n_envs),
            next_values.reshape(t_steps, n_envs),
            Config.gamma, Config.gae_lambda)
        if Config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if continuous:
            pre_tanh_flat = np.stack(pre_tanh_buf).reshape(-1, action_size)
            actions_flat = None
        else:
            pre_tanh_flat = None
            actions_flat = np.stack(act_buf).reshape(-1)

        batch_size = observations.shape[0]
        mb_count = max(1, int(Config.num_minibatches))
        usable = (batch_size // mb_count) * mb_count

        for _ in range(Config.update_epochs):
            key, key_perm = jr.split(key)
            perm = np.asarray(jr.permutation(key_perm, batch_size))[:usable]
            for mb_idx in perm.reshape(mb_count, -1):
                mb_obs = jnp.asarray(observations[mb_idx])
                mb_adv = jnp.asarray(advantages[mb_idx])

                # value regresses the fixed lambda-returns
                for _ in range(Config.pc_steps_per_update):
                    value_result = jpc.make_pc_step(
                        model=value_model,
                        optim=value_optim,
                        opt_state=value_opt_state,
                        output=jnp.asarray(value_targets[mb_idx])[:, None],
                        input=mb_obs,
                        max_t1=Config.max_t1,
                    )
                    value_model, value_opt_state = (
                        value_result["model"], value_result["opt_state"])

                # policy targets recomputed from the current policy
                params_mb = pcn_forward(policy_model, mb_obs)
                if continuous:
                    policy_targets = gaussian_pc_targets(
                        params_mb, jnp.asarray(pre_tanh_flat[mb_idx]), mb_adv,
                        action_size, Config.target_scale, exp_std=Config.exp_std)
                else:
                    policy_targets = discrete_pc_targets(
                        params_mb, jnp.asarray(actions_flat[mb_idx]).astype(jnp.int32),
                        mb_adv, action_size, Config.target_scale)
                for _ in range(Config.pc_steps_per_update):
                    policy_result = jpc.make_pc_step(
                        model=policy_model,
                        optim=policy_optim,
                        opt_state=policy_opt_state,
                        output=policy_targets,
                        input=mb_obs,
                        max_t1=Config.max_t1,
                    )
                    policy_model, policy_opt_state = (
                        policy_result["model"], policy_result["opt_state"])

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
            'training/mean_advantage_abs': float(np.abs(advantages).mean()),
        }
        logging.info(metrics)

        if Config.eval_env and training_step % Config.eval_every == 0:
            eval_time_start = time.time()
            if continuous:
                eval_returns, eval_ep_lengths, eval_key = evaluate_gaussian_policy(
                    eval_env, pcn_forward, policy_model, action_size, _flat_obs,
                    eval_key, Config.episode_length, exp_std=Config.exp_std)
            else:
                eval_returns, eval_ep_lengths, eval_key = evaluate_discrete_policy(
                    eval_env, pcn_forward, policy_model, _flat_obs,
                    eval_key, Config.episode_length)
            eval_metrics = {
                'eval/num_episodes': len(eval_returns),
                'eval/mean_score': np.round(np.mean(eval_returns), 4),
                'eval/std_score': np.round(np.std(eval_returns), 4),
                'eval/mean_episode_length': np.mean(eval_ep_lengths),
                'eval/eval_time': np.round(time.time() - eval_time_start, 3),
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
