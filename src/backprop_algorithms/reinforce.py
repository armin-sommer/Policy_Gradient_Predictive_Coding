"""REINFORCE."""

import logging
from functools import partial
import os
import pickle
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from env import ProcgenVecEnv, ProcgenEvalEnv
from utils.utils import EnvConfig
from networks.networks import ActivationFn
from backprop_algorithms.common import (
    Metrics,
    PMAP_AXIS_NAME as _PMAP_AXIS_NAME,
    Transition,
    NetworkParams,
    Networks,
    TrainingState,
    make_inference_fn,
    make_networks,
    strip_weak_type as _strip_weak_type,
    unpmap as _unpmap,
)

class Config:
    # experiment
    experiment_name = 'reinforce_procgen'
    seed = 10
    platform = 'cpu'
    write_logs_to_file = False
    save_model = False

    # environment (Procgen)
    env_name = 'coinrun'
    num_envs = 1  # DO NOT CHANGE for REINFORCE
    num_train_levels = 200
    distribution_mode = 'easy'

    # eval
    eval_env = True
    num_eval_episodes = 10
    eval_every = 20
    deterministic_eval = True
    normalize_observations = True

    # algorithm hyperparameters
    total_timesteps = int(1e6) * 8
    learning_rate = 3e-4
    unroll_length = 2048
    anneal_lr = True
    gamma = 0.99
    batch_size = 1
    num_minibatches = 1
    update_epochs = 1
    entropy_cost = 0.00
    max_grad_norm = 0.5
    reward_scaling = 1.

    # policy params
    use_cnn = True
    policy_hidden_layer_sizes: Sequence[int] = ()
    value_hidden_layer_sizes: Sequence[int] = ()
    activation: ActivationFn = nn.relu


def compute_reinforce_loss(
    params: NetworkParams,
    data: Transition,
    rng: jnp.ndarray,
    network: Networks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.99,
    reward_scaling: float = 1.,
) -> Tuple[jnp.ndarray, Metrics]:
    """Computes REINFORCE loss."""

    parametric_action_distribution = network.parametric_action_distribution
    policy_apply = network.policy_network.apply
    value_apply = network.value_network.apply

    # unpack data
    observations = data.observation
    actions = data.action
    rewards = data.reward * reward_scaling
    discounts = data.discount

    policy_logits = policy_apply(params.policy, observations)
    baseline = value_apply(params.value, observations)

    # compute log_prob
    if hasattr(data, 'extras') and isinstance(data.extras, dict) and 'policy_extras' in data.extras:
        raw_actions = data.extras['policy_extras'].get('raw_action', actions)
    else:
        raw_actions = actions
    log_prob = parametric_action_distribution.log_prob(policy_logits, raw_actions)

    # compute discounted returns
    def compute_returns(rewards, discounts, gamma):
        def scan_fn(carry, t):
            G = rewards[t] + gamma * discounts[t] * carry
            return G, G
        T = rewards.shape[0]
        _, returns = jax.lax.scan(scan_fn, jnp.zeros_like(rewards[0]),
                                   jnp.arange(T - 1, -1, -1))
        returns = returns[::-1]
        return returns

    returns = compute_returns(rewards, discounts, discounting)
    advantages = returns - baseline

    # policy loss: REINFORCE with baseline
    policy_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(advantages))

    # value loss
    value_loss = jnp.mean(jnp.square(returns - baseline))

    # entropy bonus
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))

    total_loss = policy_loss + 0.5 * value_loss - entropy_cost * entropy

    approx_kl = jnp.zeros(())
    if hasattr(data, 'extras') and isinstance(data.extras, dict) and 'policy_extras' in data.extras:
        old_log_prob = data.extras['policy_extras'].get('log_prob', None)
        if old_log_prob is not None:
            approx_kl = jnp.mean((old_log_prob - log_prob))

    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'approx_kl': jax.lax.stop_gradient(approx_kl),
    }

    return total_loss, metrics


def main(_):
    run_name = f"Exp_{Config.experiment_name}__{Config.env_name}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        log_path = f'./training_logs/reinforce/{run_name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_handler = logging.FileHandler(os.path.join(log_path, 'logs'))
        logging.getLogger().addHandler(file_handler)

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # jax set up devices
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count
    assert Config.num_envs % device_count == 0

    assert Config.batch_size * Config.num_minibatches % Config.num_envs == 0
    env_step_per_training_step = (
        Config.batch_size * Config.unroll_length * Config.num_minibatches)

    # log hyperparameters
    logging.info("|param: value|")
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            logging.info(f"|{key}:  {value}|")

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    key = jax.random.PRNGKey(Config.seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_envs, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value = jax.random.split(global_key, 2)
    del global_key

    # create Procgen env
    env_cfg = EnvConfig(
        env_name=Config.env_name,
        num_envs=Config.num_envs,
        num_train_levels=Config.num_train_levels,
        distribution_mode=Config.distribution_mode,
    )
    envs = ProcgenVecEnv(env_cfg)
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

    # create optimizer
    if Config.anneal_lr:
        learning_rate = optax.linear_schedule(
            Config.learning_rate,
            Config.learning_rate * 0.01,
            transition_steps=Config.total_timesteps,
        )
    else:
        learning_rate = Config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.max_grad_norm),
        optax.adam(learning_rate),
    )

    # create loss function
    loss_fn = partial(
        compute_reinforce_loss,
        network=network,
        entropy_cost=Config.entropy_cost,
        discounting=Config.gamma,
        reward_scaling=Config.reward_scaling,
    )


    def loss_and_pgrad(loss_fn: Callable[..., float],
                        pmap_axis_name: Optional[str],
                        has_aux: bool = False):
        g = jax.value_and_grad(loss_fn, has_aux=has_aux)

        def h(*args, **kwargs):
            value, grad = g(*args, **kwargs)
            return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

        return g if pmap_axis_name is None else h


    def gradient_update_fn(loss_fn: Callable[..., float],
                            optimizer: optax.GradientTransformation,
                            pmap_axis_name: Optional[str],
                            has_aux: bool = False):
        loss_and_pgrad_fn = loss_and_pgrad(
            loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

        def f(*args, optimizer_state):
            value, grads = loss_and_pgrad_fn(*args)
            params_update, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(args[0], params_update)
            return value, params, optimizer_state

        return f

    gradient_update_fn = gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

    def minibatch_step(carry, data: Transition,):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            data,
            key_loss,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics


    def sgd_step(carry, unused_t, data: Transition):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (Config.num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            minibatch_step,
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=Config.num_minibatches)
        return (optimizer_state, params, key), metrics


    def learn(
        data: Transition,
        training_state: TrainingState,
        key_sgd: jnp.ndarray,
    ):
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(
                sgd_step, data=data),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=Config.update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            env_steps=training_state.env_steps + env_step_per_training_step)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return new_training_state, metrics

    learn = jax.pmap(learn, axis_name=_PMAP_AXIS_NAME)


    # initialize params & training state
    init_params = NetworkParams(
        policy=network.policy_network.init(key_policy),
        value=network.value_network.init(key_value))
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        env_steps=0)
    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use])


    # create eval env
    if Config.eval_env:
        eval_cfg = EnvConfig(
            env_name=Config.env_name,
            num_envs=1,
            num_train_levels=Config.num_train_levels,
            distribution_mode=Config.distribution_mode,
        )
        eval_env = ProcgenEvalEnv(eval_cfg)
        eval_env.seed(int(eval_key[0]))
        eval_state = eval_env.reset()


    # initialize metrics
    global_step = 0
    start_time = time.time()
    training_walltime = 0
    scores = []

    def _flatten_obs(obs):
        if Config.use_cnn:
            return np.asarray(obs, dtype=np.uint8)
        return obs.reshape(obs.shape[0], -1).astype(jnp.float32) / 255.0

    # training loop
    training_step = 0
    while global_step < Config.total_timesteps:
        update_time_start = time.time()
        training_step += 1

        new_key, local_key = jax.random.split(local_key)
        training_state = _strip_weak_type(training_state)
        key_sgd, key_generate_unroll = jax.random.split(new_key, 2)

        policy = make_policy(_unpmap(training_state.params.policy))

        data = []
        transitions = []
        episode_steps = 0
        while episode_steps < 2000:
            env_state = envs.reset()
            episode_over = False
            while not episode_over:
                episode_steps += 1
                current_key, key_generate_unroll = jax.random.split(key_generate_unroll)
                obs = _flatten_obs(env_state.obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                nstate = envs.step(actions)
                state_extras = {'truncation': jnp.array([info['truncation'] for info in nstate.info])}
                transition = Transition(
                    observation=_flatten_obs(env_state.obs),
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    next_observation=_flatten_obs(nstate.obs),
                    extras={
                        'policy_extras': policy_extras,
                        'state_extras': state_extras
                })
                transitions.append(transition)
                env_state = nstate

                episode_over = any(jnp.logical_or(state_extras['truncation'], nstate.done))
        data.append(jax.tree_util.tree_map(lambda *x: np.stack(x), *transitions))
        data = jax.tree_util.tree_map(lambda *x: np.stack(x), *data)

        epoch_rollout_time = time.time() - update_time_start
        update_time_start = time.time()

        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)

        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (local_devices_to_use, -1,) + x.shape[1:]),
                                    data)

        keys_sgd = jax.random.split(key_sgd, local_devices_to_use)
        new_training_state, metrics = learn(data=data, training_state=training_state, key_sgd=keys_sgd)

        # logging
        training_state, metrics = _strip_weak_type((new_training_state, metrics))
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_update_time = time.time() - update_time_start
        training_walltime = time.time() - start_time

        sps = episode_steps / (epoch_update_time + epoch_rollout_time)
        global_step += episode_steps

        metrics = {
            'training/total_steps': global_step,
            'training/updates': training_step,
            'training/sps': np.round(sps, 3),
            'training/walltime': np.round(training_walltime, 3),
            'training/rollout_time': np.round(epoch_rollout_time, 3),
            'training/update_time': np.round(epoch_update_time, 3),
            **{f'training/{name}': float(value) for name, value in metrics.items()}
        }

        logging.info(metrics)

        # run eval
        if process_id == 0 and Config.eval_env and training_step % Config.eval_every == 0:
            eval_start_time = time.time()
            eval_steps = 0
            policy_params = _unpmap(training_state.params.policy)
            policy = make_policy(policy_params, deterministic=Config.deterministic_eval)
            while True:
                eval_steps += 1

                current_key, eval_key = jax.random.split(eval_key)
                obs = _flatten_obs(eval_state.obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                eval_state = eval_env.step(actions)
                if len(eval_env.returns) >= Config.num_eval_episodes:
                    eval_returns, eval_ep_lengths = eval_env.evaluate()
                    break
            eval_state = eval_env.reset()
            eval_time = time.time() - eval_start_time
            eval_metrics = {
                'eval/num_episodes': len(eval_returns),
                'eval/num_steps': eval_steps,
                'eval/mean_score': np.round(np.mean(eval_returns), 3),
                'eval/std_score': np.round(np.std(eval_returns), 3),
                'eval/mean_episode_length': np.mean(eval_ep_lengths),
                'eval/std_episode_length': np.round(np.std(eval_ep_lengths), 3),
                'eval/eval_time': eval_time,
            }
            logging.info(eval_metrics)
            scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), metrics['training/approx_kl']))

    logging.info('TRAINING END: training duration: %s', time.time() - start_time)

    # final eval
    if process_id == 0 and Config.eval_env:
        eval_steps = 0
        policy_params = _unpmap(training_state.params.policy)
        policy = make_policy(policy_params, deterministic=True)
        while True:
            eval_steps += 1

            current_key, eval_key = jax.random.split(eval_key)
            obs = _flatten_obs(eval_state.obs)
            actions, policy_extras = policy(obs, current_key)
            actions = np.asarray(actions)
            eval_state = eval_env.step(actions)
            if len(eval_env.returns) >= Config.num_eval_episodes:
                eval_returns, eval_ep_lengths = eval_env.evaluate()
                break
        eval_state = eval_env.reset()
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

        # save scores
        run_dir = os.path.join('experiments', run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        with open(os.path.join(run_dir, "scores.pkl"), "wb") as f:
            pickle.dump(scores, f)

    if Config.save_model:
        checkpoint_dir = getattr(Config, "checkpoint_dir", "weights")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"{run_name}.params")
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes({
                    "policy": _unpmap(training_state.params.policy),
                    "value": _unpmap(training_state.params.value),
                }))
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    main(None)
