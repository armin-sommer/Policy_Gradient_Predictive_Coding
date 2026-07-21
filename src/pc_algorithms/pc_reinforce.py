"""PC-REINFORCE — REINFORCE with a jpc-trained policy (no backprop)."""

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
    LOG_STD_MAX,
    LOG_STD_MIN,
    discrete_pc_targets,
    gaussian_pc_targets,
    sample_gaussian_action,
    split_gaussian_params,
)
from pc_algorithms.pc_eval import evaluate_discrete_policy, evaluate_gaussian_policy
from pc_algorithms.returns import compute_mc_returns


class Config:
    # experiment
    experiment_name = 'pc_reinforce'
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

    eval_env = True
    num_eval_episodes = 200
    eval_every = 1

    # algorithm hyperparameters
    total_timesteps = 60_000
    unroll_length = 250
    gamma = 0.99
    learning_rate = 1e-2
    target_scale = 1.0
    pc_steps_per_update = 1
    # PPO-style data reuse: epochs x minibatches per rollout, targets
    # recomputed from the current policy each minibatch. Defaults (1, 1)
    # reproduce the original single full-batch PC step (bandit setup).
    update_epochs = 1
    num_minibatches = 1
    normalize_advantages = False
    max_t1 = 20
    normalize_rewards = False
    exp_std = True
    # 'adam' or 'sgd'. Innocenti et al. (2305.18188) derive PC's trust-region
    # property for plain GD on the equilibrated energy; Adam re-preconditions it.
    optimizer = 'adam'

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
        episode_length=Config.episode_length,
    )
    envs = make_vec_env(env_cfg)
    envs.seed(int(key_envs[0]))
    env_state = envs.reset()

    continuous = getattr(envs.action_space, "continuous", False)
    action_size = envs.action_space.n
    policy_output_dim = 2 * action_size if continuous else action_size
    obs_dim = int(np.prod(env_state.obs.shape[1:]))

    model = jpc.make_mlp(
        key_model,
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
        model = _set_final_layer(model, Config.policy_init_logit_bias)

    if Config.optimizer == 'sgd':
        optim = optax.sgd(Config.learning_rate)
    else:
        optim = optax.adam(Config.learning_rate)
    opt_state = optim.init((eqx.filter(model, eqx.is_array), None))

    @eqx.filter_jit
    def policy_forward(model, obs):
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
        rew_buf, done_buf = [], []

        for _ in range(Config.unroll_length):
            obs = _flat_obs(env_state.obs)
            params = policy_forward(model, jnp.asarray(obs))
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
            env_state = nstate

        # (T, N, ...) buffers; flatten time-major so rows align with returns
        obs_arr = np.stack(obs_buf)
        rewards = np.stack(rew_buf)
        dones = np.stack(done_buf)
        returns = compute_mc_returns(rewards, dones, Config.gamma)
        advantages = returns - returns.mean()
        if Config.normalize_advantages:
            advantages = advantages / (advantages.std() + 1e-8)

        observations = obs_arr.reshape(-1, obs_arr.shape[-1])
        if continuous:
            pre_tanh_flat = np.stack(pre_tanh_buf).reshape(-1, action_size)
            actions_flat = None
        else:
            pre_tanh_flat = None
            actions_flat = np.stack(act_buf).reshape(-1)

        batch_size = observations.shape[0]
        mb_count = max(1, int(Config.num_minibatches))
        usable = (batch_size // mb_count) * mb_count

        # fixed probe slice for collapse diagnostics (pre/post-update policy)
        n_probe = min(2048, batch_size)
        probe_obs = jnp.asarray(observations[:n_probe])
        params_pre = policy_forward(model, probe_obs) if continuous else None

        for _ in range(Config.update_epochs):
            key, key_perm = jr.split(key)
            perm = np.asarray(jr.permutation(key_perm, batch_size))[:usable]
            for mb_idx in perm.reshape(mb_count, -1):
                mb_obs = jnp.asarray(observations[mb_idx])
                mb_adv = jnp.asarray(advantages[mb_idx])
                # recompute targets from the current policy (PPO-style reuse)
                params_mb = policy_forward(model, mb_obs)
                if continuous:
                    targets = gaussian_pc_targets(
                        params_mb, jnp.asarray(pre_tanh_flat[mb_idx]), mb_adv,
                        action_size, Config.target_scale, exp_std=Config.exp_std)
                else:
                    targets = discrete_pc_targets(
                        params_mb, jnp.asarray(actions_flat[mb_idx]).astype(jnp.int32),
                        mb_adv, action_size, Config.target_scale)
                for _ in range(Config.pc_steps_per_update):
                    result = jpc.make_pc_step(
                        model=model,
                        optim=optim,
                        opt_state=opt_state,
                        output=targets,
                        input=mb_obs,
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
            'training/mean_advantage_abs': float(np.abs(advantages).mean()),
        }
        if continuous:
            params_post = policy_forward(model, probe_obs)
            loc_pre, scale_pre, _ = split_gaussian_params(
                params_pre, action_size, exp_std=Config.exp_std)
            loc_post, scale_post, log_std_post = split_gaussian_params(
                params_post, action_size, exp_std=Config.exp_std)
            drift = jnp.abs(loc_post - loc_pre) / scale_pre
            # per-update policy KL D_KL(pi_old || pi_new): tanh is a bijection, so
            # this equals the squashed-action policy KL. Bounded KL across training
            # is the trust-region property the "PC update = TRPO" claim predicts.
            policy_kl = (jnp.log(scale_post / scale_pre)
                         + (scale_pre ** 2 + (loc_pre - loc_post) ** 2)
                         / (2.0 * scale_post ** 2) - 0.5).sum(-1)
            probe_targets = gaussian_pc_targets(
                params_pre, jnp.asarray(pre_tanh_flat[:n_probe]),
                jnp.asarray(advantages[:n_probe]), action_size,
                Config.target_scale, exp_std=Config.exp_std)
            mu_target_mag = jnp.abs(
                probe_targets[:, :action_size] - loc_pre)
            metrics.update({
                'diag/log_std_mean': float(log_std_post.mean()),
                'diag/log_std_min': float(log_std_post.min()),
                'diag/frac_std_at_min': float((log_std_post <= LOG_STD_MIN + 1e-3).mean()),
                'diag/frac_std_at_max': float((log_std_post >= LOG_STD_MAX - 1e-3).mean()),
                'diag/mu_abs_mean': float(jnp.abs(loc_post).mean()),
                'diag/policy_drift_mean': float(drift.mean()),
                'diag/policy_drift_max': float(drift.max()),
                'diag/policy_kl_mean': float(policy_kl.mean()),
                'diag/policy_kl_max': float(policy_kl.max()),
                'diag/mu_target_mag_mean': float(mu_target_mag.mean()),
                'diag/mu_target_mag_max': float(mu_target_mag.max()),
                'diag/pretanh_sat_frac': float((np.abs(pre_tanh_flat) > 2.0).mean()),
            })
        logging.info(metrics)

        if Config.eval_env and training_step % Config.eval_every == 0:
            eval_time_start = time.time()
            if continuous:
                eval_returns, eval_ep_lengths, eval_key = evaluate_gaussian_policy(
                    eval_env, policy_forward, model, action_size, _flat_obs,
                    eval_key, Config.episode_length, exp_std=Config.exp_std)
            else:
                eval_returns, eval_ep_lengths, eval_key = evaluate_discrete_policy(
                    eval_env, policy_forward, model, _flat_obs,
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
        model_path = os.path.join(checkpoint_dir, f"{run_name}.eqx")
        eqx.tree_serialise_leaves(model_path, model)
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    main(None)
