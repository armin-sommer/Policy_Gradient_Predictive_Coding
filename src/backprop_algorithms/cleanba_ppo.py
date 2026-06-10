"""Cleanba-style PPO baseline in JAX/Flax.

Single-file PPO following the loss/update structure of Cleanba
(https://github.com/vwxyzjn/cleanba) / CleanRL:
  - GAE computed once per iteration (not recomputed per minibatch),
  - rollout flattened over (steps * envs) and shuffled into minibatches,
  - per-minibatch advantage normalization,
  - clipped value loss,
  - Adam(eps=1e-5) with linear LR decay to 0.

Reuses this repo's networks, distributions, and env wrappers so it plugs
into scripts/run_train.py like the other algorithms.
"""

import logging
from functools import partial
import os
import pickle
import random
import time
from typing import NamedTuple, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from env import make_vec_env
from utils.utils import EnvConfig
from networks.networks import ActivationFn
from backprop_algorithms.common import (
    NetworkParams,
    TrainingState,
    Transition,
    apply_policy_init_logit_bias,
    compute_gae,
    make_inference_fn,
    make_networks,
)


class Config:
    # experiment
    experiment_name = 'cleanba_ppo'
    seed = 1
    platform = 'cpu'
    write_logs_to_file = False
    save_model = False

    # environment (Procgen or bandit)
    env_name = 'coinrun'
    num_envs = 8
    num_train_levels = 200
    distribution_mode = 'easy'
    arm_means = (1.0, 0.9)          # bandit only
    deterministic_rewards = True    # bandit only

    # eval
    eval_env = True
    num_eval_episodes = 10
    eval_every = 5
    deterministic_eval = True

    # algorithm hyperparameters (Cleanba PPO defaults)
    total_timesteps = int(1e6) * 8
    learning_rate = 2.5e-4
    adam_eps = 1e-5
    unroll_length = 128             # a.k.a. num_steps
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4
    normalize_advantages = True
    clip_eps = 0.1
    clip_vloss = True
    entropy_cost = 0.01
    vf_cost = 0.5
    max_grad_norm = 0.5

    # policy params
    use_cnn = True
    policy_hidden_layer_sizes: Sequence[int] = ()
    value_hidden_layer_sizes: Sequence[int] = ()
    activation: ActivationFn = nn.relu
    policy_init_logit_bias = None  # e.g. [0.0, 4.0] for the bandit plateau init


class Batch(NamedTuple):
    observation: jnp.ndarray
    raw_action: jnp.ndarray
    log_prob: jnp.ndarray
    advantage: jnp.ndarray
    value_target: jnp.ndarray
    value_old: jnp.ndarray


def main(_):
    run_name = f"Exp_{Config.experiment_name}__{Config.env_name}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        log_path = f'./training_logs/cleanba_ppo/{run_name}'
        os.makedirs(log_path, exist_ok=True)
        logging.getLogger().addHandler(
            logging.FileHandler(os.path.join(log_path, 'logs')))

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    env_step_per_training_step = Config.num_envs * Config.unroll_length
    num_training_steps = int(np.ceil(Config.total_timesteps / env_step_per_training_step))
    batch_size = Config.num_envs * Config.unroll_length
    assert batch_size % Config.num_minibatches == 0, \
        "num_envs * unroll_length must be divisible by num_minibatches"
    minibatch_size = batch_size // Config.num_minibatches

    logging.info("|param: value|")
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            logging.info(f"|{key}:  {value}|")

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    key = jax.random.PRNGKey(Config.seed)
    key, key_envs, eval_key, key_policy, key_value = jax.random.split(key, 5)

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
    if Config.use_cnn:
        observation_shape = tuple(env_state.obs.shape[1:])
    else:
        observation_shape = int(np.prod(env_state.obs.shape[1:]))

    network = make_networks(
        observation_size=observation_shape,
        action_size=action_size,
        policy_hidden_layer_sizes=Config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=Config.value_hidden_layer_sizes,
        activation=Config.activation,
        discrete_policy=True,
        use_cnn=Config.use_cnn,
    )
    make_policy = make_inference_fn(network)
    dist = network.parametric_action_distribution
    policy_apply = network.policy_network.apply
    value_apply = network.value_network.apply

    if Config.anneal_lr:
        learning_rate = optax.linear_schedule(
            Config.learning_rate, 0.0,
            transition_steps=num_training_steps * Config.update_epochs * Config.num_minibatches,
        )
    else:
        learning_rate = Config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.max_grad_norm),
        optax.adam(learning_rate, eps=Config.adam_eps),
    )

    def loss_fn(params: NetworkParams, mb: Batch, rng: jnp.ndarray):
        logits = policy_apply(params.policy, mb.observation)
        new_log_prob = dist.log_prob(logits, mb.raw_action)
        log_ratio = new_log_prob - mb.log_prob
        ratio = jnp.exp(log_ratio)
        approx_kl = ((ratio - 1) - log_ratio).mean()

        advantages = mb.advantage
        if Config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - Config.clip_eps, 1 + Config.clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        new_value = value_apply(params.value, mb.observation)
        if Config.clip_vloss:
            v_loss_unclipped = jnp.square(new_value - mb.value_target)
            v_clipped = mb.value_old + jnp.clip(
                new_value - mb.value_old, -Config.clip_eps, Config.clip_eps)
            v_loss_clipped = jnp.square(v_clipped - mb.value_target)
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * jnp.square(new_value - mb.value_target).mean()

        entropy = dist.entropy(logits, rng).mean()
        total_loss = pg_loss - Config.entropy_cost * entropy + Config.vf_cost * v_loss
        metrics = {
            'total_loss': total_loss,
            'policy_loss': pg_loss,
            'value_loss': v_loss,
            'entropy': entropy,
            'approx_kl': jax.lax.stop_gradient(approx_kl),
        }
        return total_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def compute_batch(params: NetworkParams, data: Transition) -> Batch:
        """GAE over the (T, N, ...) rollout, computed once per iteration."""
        baseline = value_apply(params.value, data.observation)
        bootstrap_value = value_apply(params.value, data.next_observation[-1])
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)
        vs, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=data.reward,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=Config.gae_lambda,
            discount=Config.gamma,
        )
        batch = Batch(
            observation=data.observation,
            raw_action=data.extras['policy_extras']['raw_action'],
            log_prob=data.extras['policy_extras']['log_prob'],
            advantage=advantages,
            value_target=vs,
            value_old=baseline,
        )
        return jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), batch)

    @jax.jit
    def update_epoch(training_state: TrainingState, batch: Batch, key: jnp.ndarray):
        def minibatch_step(carry, mb: Batch):
            optimizer_state, params, key = carry
            key, key_loss = jax.random.split(key)
            (_, metrics), grads = grad_fn(params, mb, key_loss)
            params_update, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(params, params_update)
            return (optimizer_state, params, key), metrics

        key, key_perm = jax.random.split(key)
        perm = jax.random.permutation(key_perm, batch.observation.shape[0])
        shuffled = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x[perm], (Config.num_minibatches, minibatch_size) + x.shape[1:]),
            batch)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            minibatch_step,
            (training_state.optimizer_state, training_state.params, key),
            shuffled,
            length=Config.num_minibatches)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            env_steps=training_state.env_steps), metrics

    # initialize params & training state
    init_policy_params = network.policy_network.init(key_policy)
    if Config.policy_init_logit_bias is not None:
        init_policy_params = apply_policy_init_logit_bias(
            init_policy_params, Config.policy_init_logit_bias)
    init_params = NetworkParams(
        policy=init_policy_params,
        value=network.value_network.init(key_value))
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        env_steps=jnp.zeros(()))

    # create eval env
    if Config.eval_env:
        eval_cfg = EnvConfig(
            env_name=Config.env_name,
            num_envs=1,
            num_train_levels=Config.num_train_levels,
            distribution_mode=Config.distribution_mode,
            arm_means=tuple(Config.arm_means),
            deterministic_rewards=Config.deterministic_rewards,
        )
        eval_env = make_vec_env(eval_cfg, evaluate=True)
        eval_env.seed(int(eval_key[0]))
        eval_state = eval_env.reset()

    global_step = 0
    start_time = time.time()
    scores = []

    def _flatten_obs(obs):
        if Config.use_cnn:
            return np.asarray(obs, dtype=np.uint8)
        return envs.normalize_obs(obs.reshape(obs.shape[0], -1).astype(np.float32))

    def run_eval(policy_params, deterministic, eval_state, eval_key):
        policy = make_policy(policy_params, deterministic=deterministic)
        eval_steps = 0
        while True:
            eval_steps += 1
            current_key, eval_key = jax.random.split(eval_key)
            obs = _flatten_obs(eval_state.obs)
            actions, _ = policy(obs, current_key)
            eval_state = eval_env.step(np.asarray(actions))
            if len(eval_env.returns) >= Config.num_eval_episodes:
                returns, ep_lengths = eval_env.evaluate()
                return returns, ep_lengths, eval_steps, eval_env.reset(), eval_key

    # training loop
    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()
        key, key_rollout, key_update = jax.random.split(key, 3)

        policy = make_policy(training_state.params.policy)

        transitions = []
        for _ in range(Config.unroll_length):
            key_rollout, current_key = jax.random.split(key_rollout)
            obs = _flatten_obs(env_state.obs)
            actions, policy_extras = policy(obs, current_key)
            actions = np.asarray(actions)
            nstate = envs.step(actions)
            state_extras = {'truncation': jnp.array(
                [info['truncation'] for info in nstate.info])}
            transitions.append(Transition(
                observation=obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=_flatten_obs(nstate.obs),
                extras={
                    'policy_extras': policy_extras,
                    'state_extras': state_extras,
                }))
            env_state = nstate
        # (T, N, ...)
        data = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *transitions)

        epoch_rollout_time = time.time() - update_time_start
        update_time_start = time.time()

        batch = compute_batch(training_state.params, data)
        for _ in range(Config.update_epochs):
            key_update, key_epoch = jax.random.split(key_update)
            training_state, metrics = update_epoch(training_state, batch, key_epoch)
        training_state = training_state.replace(
            env_steps=training_state.env_steps + env_step_per_training_step)

        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_update_time = time.time() - update_time_start
        global_step += env_step_per_training_step
        sps = env_step_per_training_step / (epoch_update_time + epoch_rollout_time)

        metrics = {
            'training/total_steps': global_step,
            'training/updates': training_step,
            'training/sps': np.round(sps, 3),
            'training/walltime': np.round(time.time() - start_time, 3),
            'training/rollout_time': np.round(epoch_rollout_time, 3),
            'training/update_time': np.round(epoch_update_time, 3),
            **{f'training/{name}': float(value) for name, value in metrics.items()}
        }
        logging.info(metrics)

        if Config.eval_env and training_step % Config.eval_every == 0:
            eval_start_time = time.time()
            eval_returns, eval_ep_lengths, eval_steps, eval_state, eval_key = run_eval(
                training_state.params.policy, Config.deterministic_eval, eval_state, eval_key)
            eval_metrics = {
                'eval/num_episodes': len(eval_returns),
                'eval/num_steps': eval_steps,
                'eval/mean_score': np.round(np.mean(eval_returns), 3),
                'eval/std_score': np.round(np.std(eval_returns), 3),
                'eval/mean_episode_length': np.mean(eval_ep_lengths),
                'eval/std_episode_length': np.round(np.std(eval_ep_lengths), 3),
                'eval/eval_time': time.time() - eval_start_time,
            }
            logging.info(eval_metrics)
            scores.append((global_step, np.mean(eval_returns),
                           np.mean(eval_ep_lengths), metrics['training/approx_kl']))

    logging.info('TRAINING END: training duration: %s', time.time() - start_time)

    # final eval
    if Config.eval_env:
        eval_returns, eval_ep_lengths, eval_steps, eval_state, eval_key = run_eval(
            training_state.params.policy, True, eval_state, eval_key)
        eval_metrics = {
            'final_eval/num_episodes': len(eval_returns),
            'final_eval/num_steps': eval_steps,
            'final_eval/mean_score': np.mean(eval_returns),
            'final_eval/std_score': np.std(eval_returns),
            'final_eval/mean_episode_length': np.mean(eval_ep_lengths),
            'final_eval/std_episode_length': np.std(eval_ep_lengths),
        }
        logging.info(eval_metrics)
        scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), None))

        run_dir = os.path.join('experiments', run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "scores.pkl"), "wb") as f:
            pickle.dump(scores, f)

    if Config.save_model:
        checkpoint_dir = getattr(Config, "checkpoint_dir", "weights")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"{run_name}.params")
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes({
                    "policy": training_state.params.policy,
                    "value": training_state.params.value,
                }))
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    main(None)
