"""Trust Region Policy Optimization (TRPO) in JAX/Flax.

Adapted from https://github.com/Matt00n/PolicyGradientsJax (MIT License).
"""

import logging
from functools import partial
import os
import pickle
import random
import time
from typing import Mapping, Sequence, Tuple

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
    Metrics,
    PMAP_AXIS_NAME as _PMAP_AXIS_NAME,
    Transition,
    NetworkParams,
    Networks,
    TrainingState,
    apply_policy_init_logit_bias,
    compute_gae,
    make_inference_fn,
    make_networks,
    strip_weak_type as _strip_weak_type,
    unpmap as _unpmap,
)


class Config:
    # experiment
    experiment_name = 'trpo_procgen'
    seed = 30
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
    eval_every = 2
    deterministic_eval = True
    normalize_observations = True

    # algorithm hyperparameters
    total_timesteps = int(1e6) * 8
    learning_rate = 3e-4
    unroll_length = 2048
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    batch_size = 1
    num_minibatches = 8
    update_epochs = 10
    normalize_advantages = True
    target_kl = 0.01
    cg_damping: float = 0.1
    cg_max_iterations: int = 10
    line_search_max_iter: int = 10
    line_search_shrinking_factor: float = 0.8
    vf_cost = 1.
    max_grad_norm = 0.5
    reward_scaling = 1.

    # policy params
    use_cnn = True
    policy_hidden_layer_sizes: Sequence[int] = ()
    value_hidden_layer_sizes: Sequence[int] = ()
    activation: ActivationFn = nn.relu
    policy_init_logit_bias = None  # e.g. [0.0, 4.0] for the bandit plateau init


def compute_policy_objective(params, data, hidden, advantages, network):
    parametric_action_distribution = network.parametric_action_distribution
    policy_apply = network.policy_network.apply

    policy_logits = policy_apply(params.policy, hidden)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action'])
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    log_ratio = target_action_log_probs - behaviour_action_log_probs
    rho_s = jnp.exp(log_ratio)

    policy_objective = jnp.mean(rho_s * advantages)
    kl_div = jnp.mean(parametric_action_distribution.kl_divergence(
        jax.lax.stop_gradient(policy_logits), policy_logits))
    return policy_objective, kl_div


def compute_policy_objective_and_kl(params, data, hidden, advantages, network,
                                     policy_objective_grad_fn):
    (policy_objective, kl_div), policy_objective_grad = policy_objective_grad_fn(
        params, data, hidden, advantages, network)
    return kl_div, (policy_objective, policy_objective_grad)


def jacobian_vector_product(params, vector, data, hidden, advantages, network,
                            policy_objective_and_kl_grad_fn):
    (_, (_, _)), grad_kl = policy_objective_and_kl_grad_fn(
        params, data, hidden, advantages, network)
    product_tree = jax.tree_util.tree_map(
        lambda x, y: jnp.sum(x * y), grad_kl, jax.lax.stop_gradient(vector))
    return sum(jax.tree_util.tree_leaves(product_tree))


def hessian_vector_product(vector, params, data, hidden, advantages, network,
                           hessian_fn, cg_damping=0.1):
    hessian = hessian_fn(params, vector, data, hidden, advantages, network)
    return jax.tree_util.tree_map(lambda x, y: x + cg_damping * y, hessian, vector)


def trpo_policy_update(
    params, data, network,
    policy_objective_and_kl_grad_fn,
    hessian_vector_product,
    target_kl=0.01,
    line_search_max_iter=10,
    line_search_shrinking_factor=0.8,
    cg_max_iterations=10,
    discounting=0.9,
    reward_scaling=1.0,
    gae_lambda=0.95,
    normalize_advantage=True,
    pmap_axis_name=None,
):
    parametric_action_distribution = network.parametric_action_distribution
    policy_apply = network.policy_network.apply
    value_apply = network.value_network.apply

    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    hidden = data.observation
    hidden_boot = data.next_observation[-1]

    policy_logits = policy_apply(params.policy, hidden)
    baseline = value_apply(params.value, hidden)
    bootstrap_value = value_apply(params.value, hidden_boot)

    rewards = data.reward * reward_scaling
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    _, advantages = compute_gae(
        truncation=truncation, termination=termination,
        rewards=rewards, values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda, discount=discounting)
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    (kl_div, (policy_objective, policy_objective_grad)), grad_kl = policy_objective_and_kl_grad_fn(
        params, data, hidden, advantages, network)

    policy_objective_grad = jax.lax.pmean(policy_objective_grad, axis_name=pmap_axis_name)
    grad_kl = jax.lax.pmean(grad_kl, axis_name=pmap_axis_name)

    hessian_vector_product_fn = partial(hessian_vector_product,
                                        params=params, data=data,
                                        hidden=hidden, advantages=advantages,
                                        network=network)

    search_direction, _ = jax.scipy.sparse.linalg.cg(
        hessian_vector_product_fn, policy_objective_grad,
        tol=1e-10, maxiter=cg_max_iterations)

    line_search_max_step_size = 2 * target_kl
    denom_tree = jax.tree_util.tree_map(
        lambda x, y: jnp.sum(x * y),
        search_direction, hessian_vector_product_fn(search_direction))
    line_search_max_step_size /= sum(jax.tree_util.tree_leaves(denom_tree))
    line_search_max_step_size = jnp.sqrt(line_search_max_step_size)
    line_search_backtrack_coeff = 1.0

    def line_search_step(carry):
        (iteration, new_policy_objective, kl_div, line_search_backtrack_coeff, unused_params) = carry
        iteration += 1

        new_params = jax.tree_util.tree_map(
            lambda x, y: x + line_search_backtrack_coeff * line_search_max_step_size * y,
            params, search_direction)

        logits = policy_apply(new_params.policy, data.observation)
        target_action_log_probs = parametric_action_distribution.log_prob(
                logits, data.extras['policy_extras']['raw_action'])

        log_ratio = target_action_log_probs - behaviour_action_log_probs
        rho_s = jnp.exp(log_ratio)
        new_policy_objective = (advantages * rho_s).mean()

        kl_div = jnp.mean(parametric_action_distribution.kl_divergence(policy_logits, logits))

        line_search_backtrack_coeff *= line_search_shrinking_factor
        return (iteration, new_policy_objective, kl_div, line_search_backtrack_coeff, new_params)

    (iterations, new_policy_objective, kl_div, _, new_params) = jax.lax.while_loop(
            lambda x: jnp.logical_and(
                jnp.logical_or(x[2] > target_kl, x[1] < policy_objective),
                x[0] < line_search_max_iter),
            line_search_step,
            (0, policy_objective, 100., line_search_backtrack_coeff, params))

    is_line_search_success = jnp.logical_and(kl_div <= target_kl, new_policy_objective >= policy_objective)
    policy_params = jax.lax.cond(is_line_search_success, lambda: new_params, lambda: params)

    metrics = {
        'policy_objective': policy_objective,
        'new_policy_objective': new_policy_objective,
        'kl_div': jax.lax.stop_gradient(kl_div),
        'line_search_success': jnp.array(is_line_search_success, int),
        'iterations': jnp.array(iterations, int),
    }

    return policy_params, metrics


def compute_value_loss(params, data, network, vf_cost=1., discounting=0.9,
                       reward_scaling=1.0, gae_lambda=0.95):
    value_apply = network.value_network.apply

    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    hidden = data.observation
    hidden_boot = data.next_observation[-1]

    baseline = value_apply(params.value, hidden)
    bootstrap_value = value_apply(params.value, hidden_boot)

    rewards = data.reward * reward_scaling
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    vs, _ = compute_gae(
        truncation=truncation, termination=termination,
        rewards=rewards, values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda, discount=discounting)

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_cost

    return v_loss, {'value_loss': v_loss}


def main(_):
    run_name = f"Exp_{Config.experiment_name}__{Config.env_name}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        log_path = f'./training_logs/trpo/{run_name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        file_handler = logging.FileHandler(os.path.join(log_path, 'logs'))
        logging.getLogger().addHandler(file_handler)

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count
    assert Config.num_envs % device_count == 0

    assert Config.batch_size * Config.num_minibatches % Config.num_envs == 0
    env_step_per_training_step = (
        Config.batch_size * Config.unroll_length * Config.num_minibatches)
    num_training_steps = np.ceil(Config.total_timesteps / env_step_per_training_step).astype(int)

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

    # create env (Procgen or bandit)
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

    compute_policy_objective_fn = partial(compute_policy_objective)
    policy_objective_grad_fn = jax.value_and_grad(compute_policy_objective_fn, has_aux=True)

    compute_policy_objective_and_kl_fn = partial(
        compute_policy_objective_and_kl, policy_objective_grad_fn=policy_objective_grad_fn)
    policy_objective_and_kl_grad_fn = jax.value_and_grad(
        compute_policy_objective_and_kl_fn, has_aux=True)

    jacobian_vector_product_fn = partial(
        jacobian_vector_product, policy_objective_and_kl_grad_fn=policy_objective_and_kl_grad_fn)

    hessian_fn = jax.grad(jacobian_vector_product_fn)
    hessian_vector_product_fn = partial(
        hessian_vector_product, hessian_fn=hessian_fn, cg_damping=Config.cg_damping)

    policy_loss_fn = partial(
        trpo_policy_update,
        network=network,
        policy_objective_and_kl_grad_fn=policy_objective_and_kl_grad_fn,
        hessian_vector_product=hessian_vector_product_fn,
        target_kl=Config.target_kl,
        line_search_max_iter=Config.line_search_max_iter,
        line_search_shrinking_factor=Config.line_search_shrinking_factor,
        cg_max_iterations=Config.cg_max_iterations,
        discounting=Config.gamma,
        reward_scaling=Config.reward_scaling,
        gae_lambda=Config.gae_lambda,
        normalize_advantage=Config.normalize_advantages,
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    v_loss_fn = partial(
        compute_value_loss,
        network=network,
        vf_cost=Config.vf_cost,
        discounting=Config.gamma,
        reward_scaling=Config.reward_scaling,
        gae_lambda=Config.gae_lambda,
    )


    def loss_and_pgrad(loss_fn, pmap_axis_name, has_aux=False):
        g = jax.value_and_grad(loss_fn, has_aux=has_aux)
        def h(*args, **kwargs):
            value, grad = g(*args, **kwargs)
            return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)
        return g if pmap_axis_name is None else h

    def gradient_update_fn(loss_fn, optimizer, pmap_axis_name, has_aux=False):
        loss_and_pgrad_fn = loss_and_pgrad(
            loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)
        def f(*args, optimizer_state):
            value, grads = loss_and_pgrad_fn(*args)
            params_update, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(args[0], params_update)
            return value, params, optimizer_state
        return f

    gradient_update_fn = gradient_update_fn(
        v_loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    def v_minibatch_step(carry, data: Transition):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params, data, optimizer_state=optimizer_state)
        return (optimizer_state, params, key), metrics

    def v_sgd_step(carry, unused_t, data: Transition):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (Config.num_minibatches, -1) + x.shape[1:])
            return x
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            v_minibatch_step,
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=Config.num_minibatches)
        return (optimizer_state, params, key), metrics

    def policy_minibatch_step(carry, data: Transition):
        params, key = carry
        key, key_loss = jax.random.split(key)
        params, metrics = policy_loss_fn(params, data)
        return (params, key), metrics

    def policy_sgd_step(carry, unused_t, data: Transition):
        params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (Config.num_minibatches, -1) + x.shape[1:])
            return x
        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (params, _), metrics = jax.lax.scan(
            policy_minibatch_step,
            (params, key_grad),
            shuffled_data,
            length=Config.num_minibatches)
        return (params, key), metrics

    def learn(data: Transition, training_state: TrainingState, key_sgd: jnp.ndarray):
        # jax arrays are immutable; no deepcopy needed (and tracers don't support it)
        value_params = training_state.params.value
        key_policy, key_sgd = jax.random.split(key_sgd)
        (policy_params, _), policy_metrics = policy_sgd_step(
            (training_state.params, key_policy), (), data=data)
        policy_params = policy_params.replace(value=value_params)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(v_sgd_step, data=data),
            (training_state.optimizer_state, policy_params, key_sgd), (),
            length=Config.update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            env_steps=training_state.env_steps + env_step_per_training_step)

        metrics = policy_metrics | metrics
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return new_training_state, metrics

    learn = jax.pmap(learn, axis_name=_PMAP_AXIS_NAME)

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
            arm_means=tuple(Config.arm_means),
            deterministic_rewards=Config.deterministic_rewards,
        )
        eval_env = make_vec_env(eval_cfg, evaluate=True)
        eval_env.seed(int(eval_key[0]))
        eval_state = eval_env.reset()

    global_step = 0
    start_time = time.time()
    training_walltime = 0
    scores = []

    def _flatten_obs(obs):
        if Config.use_cnn:
            return np.asarray(obs, dtype=np.uint8)
        return envs.normalize_obs(obs.reshape(obs.shape[0], -1).astype(np.float32))

    # training loop
    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()

        new_key, local_key = jax.random.split(local_key)
        training_state = _strip_weak_type(training_state)
        key_sgd, key_generate_unroll = jax.random.split(new_key, 2)

        policy = make_policy(_unpmap(training_state.params.policy))

        data = []
        for step in range(Config.batch_size * Config.num_minibatches // Config.num_envs):
            transitions = []
            for unroll_step in range(Config.unroll_length):
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
            data.append(jax.tree_util.tree_map(lambda *x: np.stack(x), *transitions))
        data = jax.tree_util.tree_map(lambda *x: np.stack(x), *data)

        epoch_rollout_time = time.time() - update_time_start
        update_time_start = time.time()

        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        assert data.discount.shape[1:] == (Config.unroll_length,)

        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (local_devices_to_use, -1,) + x.shape[1:]), data)

        keys_sgd = jax.random.split(key_sgd, local_devices_to_use)
        new_training_state, metrics = learn(data=data, training_state=training_state, key_sgd=keys_sgd)

        training_state, metrics = _strip_weak_type((new_training_state, metrics))
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_update_time = time.time() - update_time_start
        training_walltime = time.time() - start_time
        sps = env_step_per_training_step / (epoch_update_time + epoch_rollout_time)
        global_step += env_step_per_training_step

        current_step = int(_unpmap(training_state.env_steps))

        metrics = {
            'training/total_steps': current_step,
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
            scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), metrics['training/kl_div']))

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
