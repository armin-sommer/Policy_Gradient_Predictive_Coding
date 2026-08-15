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
from pc_algorithms import gaussian_policy as gpol
from pc_algorithms.likelihood_energy import make_likelihood_pc_step
from pc_algorithms.inference import make_pc_step_at_rate, inference_residual
from pc_algorithms.precision_energy import (
    effective_policy_outputs,
    natural_output_displacement,
    settle_precision_outputs,
    trust_region_precision,
)
from pc_algorithms.parameter_space_pc import make_parameter_pc_policy_step
from pc_algorithms.gaussian_policy import (
    discrete_pc_targets,
    gaussian_pc_targets,
    sample_gaussian_action,
    split_gaussian_params,
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
    # Run the inference ODE at per-sample rate instead of jpc's batch-normalised
    # rate. jpc's energy carries a 1/N factor, which is a pure rescaling of the
    # inference clock (same fixed point) but makes settling need t1/N ~ 10-40 --
    # unreachable at bench N, since jpc hard-stops at t=4096. With this on,
    # max_t1=20 settles at any batch size. See pc_algorithms/inference.py and
    # scripts/probe_inference_settling.py. Off = the committed runs' behaviour.
    inference_rate_correction = False
    normalize_rewards = False
    exp_std = True
    # State-independent std: match the SOTA PPO/TRPO policy (a single global
    # log_std vector, init 0 -> std=1), instead of the network emitting a
    # per-state log_std. Leaves the mean/PC pathway untouched; only changes how
    # sigma is parameterized + updated (global, batch-averaged score step). This
    # removes the per-state sigma collapse that detonates the 1/sigma^2 target.
    state_indep_std = False
    # 'adam' or 'sgd'. Innocenti et al. (2305.18188) derive PC's trust-region
    # property for plain GD on the equilibrated energy; Adam re-preconditions it.
    optimizer = 'adam'
    # Optional global-norm clip on the PC *policy* gradient (None = off). Applied
    # before the optimizer step; the logged policy_grad_norm_max is the PRE-clip
    # norm, so you see the true spike and how often it exceeds the clip.
    max_grad_norm = None
    # Natural-gradient target: Fisher-precondition the target so the mean offset
    # is ts*A*(z-mu) (no 1/sigma^2) instead of ts*A*(z-mu)/sigma^2. This is the
    # target the "PC update = natural gradient" claim implies; removes the
    # amplifier at the source. False = Euclidean (current). See gaussian_pc_targets.
    natural_target = False
    # Output-space trust region: cap the mean-target offset (None = off).
    # Optimizer-agnostic; composes on top of natural_target.
    # target_clip_rel=True makes the cap relative (|loc_target-mu| <= clip*sigma).
    target_clip = None
    target_clip_rel = False
    # Likelihood-energy route (mutually exclusive with the target route above).
    # False = build a target and let PC hit it with squared error (natural_target
    # supplies the geometry). True = no target; PC's OUTPUT ENERGY becomes the
    # advantage-weighted Gaussian NLL, so sigma enters the objective itself.
    # Only this route tests "PC controls the geometry implicitly". Continuous only.
    likelihood_energy = False
    # 'signed' = A * NLL (faithful, unbounded below); 'exp' = exp(A/tau) * NLL
    # (positive weights, bounded below; the RWR/MPO form).
    likelihood_adv_mode = 'signed'
    likelihood_tau = 1.0
    # Innocenti-style free policy-output inference. ``precision_npg`` uses a
    # fixed additive-error precision; ``precision_tr`` chooses the precision so
    # the settled output lies on precision_max_radius. The settled output is then
    # consolidated by the ordinary hidden-activity PC update.
    policy_geometry = 'target'  # target | precision_* | parameter_*
    precision_beta = 1.0
    precision_max_radius = 0.01
    # Parameter-space PC: infer the shared displacement that minimises
    # 0.5*d.T(F+damping*I)d-g.T*d. NPG uses a fixed scale; TR rescales to the
    # local Fisher-radius boundary. Exact empirical KL is diagnostic only.
    parameter_pc_damping = 0.1
    parameter_pc_step_size = 0.05  # Outer NPG scale eta.
    parameter_pc_max_radius = 0.01
    parameter_pc_max_kl = 0.01
    parameter_pc_cg_iters = 50
    parameter_pc_cg_tol = 1e-8
    parameter_pc_joint_steps = 64
    parameter_pc_joint_parameter_rate = 0.01
    parameter_pc_joint_activity_rate = 0.01
    parameter_pc_joint_error_rate = 0.01
    parameter_pc_joint_penalty = 1.0
    parameter_pc_joint_precision_rate = 0.01
    parameter_pc_joint_precision_min = 0.1
    parameter_pc_joint_precision_max = 10.0
    parameter_pc_exact_kl_steps = 1000
    parameter_pc_exact_kl_primal_rate = 0.01
    parameter_pc_exact_kl_dual_rate = 0.02
    parameter_pc_exact_kl_penalty = 1.0
    parameter_pc_exact_kl_tol = 1e-4
    # Raise the log_std floor to tame the 1/sigma^2 amplifier (None = default -2;
    # e.g. -1 -> sigma_min 0.37, so 1/sigma^2 caps at ~7 instead of ~55).
    log_std_min = None
    log_std_max = None

    width = 32
    depth = 2
    act_fn = 'relu'
    sota_init = False
    policy_init_logit_bias = None


def _global_norm(tree):
    """Global L2 norm from jpc's per-layer grad-norm pytree: sqrt(sum ||g_l||^2)."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))


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


def _orthogonal_init_model(model, key, output_gain):
    """Match the orthogonal initialization used by the PPO/TRPO baselines."""
    keys = jr.split(key, len(model))
    for index, (block, layer_key) in enumerate(zip(model, keys)):
        linear = block.layers[1]
        gain = output_gain if index == len(model) - 1 else jnp.sqrt(2.0)
        weight = jax.nn.initializers.orthogonal(gain)(
            layer_key, linear.weight.shape, linear.weight.dtype)
        model = eqx.tree_at(
            lambda m, i=index: m[i].layers[1].weight, model, weight)
        if linear.bias is not None:
            model = eqx.tree_at(
                lambda m, i=index: m[i].layers[1].bias,
                model,
                jnp.zeros_like(linear.bias),
            )
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

    if Config.log_std_min is not None:
        gpol.LOG_STD_MIN = float(Config.log_std_min)  # raise the sigma floor
    if Config.log_std_max is not None:
        gpol.LOG_STD_MAX = float(Config.log_std_max)

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
    precision_modes = ('precision_npg', 'precision_tr')
    parameter_modes = (
        'parameter_npg', 'parameter_tr', 'parameter_local_tr',
        'parameter_local_joint_tr', 'parameter_local_precision_tr',
        'parameter_exact_tr')
    state_indep = continuous and Config.state_indep_std
    use_precision = continuous and Config.policy_geometry in precision_modes
    use_parameter = continuous and Config.policy_geometry in parameter_modes
    mean_only_policy = state_indep and use_parameter
    policy_output_dim = (
        action_size if mean_only_policy else
        (2 * action_size if continuous else action_size))
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
    if Config.sota_init:
        if not continuous:
            raise ValueError("sota_init is only supported for continuous policies")
        policy_model = _orthogonal_init_model(
            policy_model, jr.fold_in(key_policy, 1), output_gain=0.01)
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
    if Config.sota_init:
        value_model = _orthogonal_init_model(
            value_model, jr.fold_in(key_value, 1), output_gain=1.0)

    make_optim = optax.sgd if Config.optimizer == 'sgd' else optax.adam
    policy_optim = make_optim(Config.learning_rate)
    if Config.max_grad_norm is not None:
        policy_optim = optax.chain(
            optax.clip_by_global_norm(float(Config.max_grad_norm)), policy_optim)
    policy_opt_state = policy_optim.init((eqx.filter(policy_model, eqx.is_array), None))
    value_optim = make_optim(Config.value_learning_rate)
    value_opt_state = value_optim.init((eqx.filter(value_model, eqx.is_array), None))

    @eqx.filter_jit
    def pcn_forward(model, obs):
        activities = jpc.init_activities_with_ffwd(model=model, input=obs)
        return activities[-1]

    # State-independent std: a single global log_std vector (init 0 -> std=1 with
    # exp_std), shared across states like the SOTA PPO/TRPO policy. The network
    # still emits 2*action_size, but its std head is frozen (PC-targeted to its
    # own output) and ignored; behavior uses this global vector instead.
    use_likelihood = continuous and Config.likelihood_energy
    if Config.policy_geometry not in ('target', *precision_modes, *parameter_modes):
        raise ValueError(
            "policy_geometry must be target, precision_npg, precision_tr, "
            "parameter_npg, parameter_tr, parameter_local_tr, "
            "parameter_local_joint_tr, parameter_local_precision_tr, or "
            "parameter_exact_tr")
    if use_likelihood and state_indep:
        raise ValueError("likelihood_energy does not support state_indep_std")
    if use_likelihood and Config.natural_target:
        raise ValueError("set natural_target=False when likelihood_energy=True: "
                         "the energy supplies the geometry, not the target")
    if use_precision and state_indep:
        raise ValueError("precision-energy modes require state-dependent policy outputs")
    if use_precision and use_likelihood:
        raise ValueError("precision-energy and likelihood-energy modes are mutually exclusive")
    if use_precision and Config.natural_target:
        raise ValueError("set natural_target=False for precision-energy modes: "
                         "output inference supplies the natural displacement")
    if use_parameter and use_likelihood:
        raise ValueError("parameter-space PC and likelihood-energy modes are mutually exclusive")
    if use_parameter and Config.natural_target:
        raise ValueError("set natural_target=False for parameter-space PC: "
                         "the Fisher solve supplies the natural geometry")
    policy_log_std = jnp.zeros((action_size,), dtype=jnp.float32)

    def _with_global_std(net_out):
        """Replace the std half of a [.., 2*action_size] output with global std."""
        mean = net_out[..., :action_size]
        return jnp.concatenate(
            [mean, jnp.broadcast_to(policy_log_std, mean.shape)], axis=-1)

    def _eval_policy_forward(model, obs):
        outputs = pcn_forward(model, obs)
        return _with_global_std(outputs) if state_indep else outputs

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
    # Every PC policy-gradient norm seen so far (one per weight update). Used for
    # the cumulative tail percentiles + clip bind-rate below: a *numerical guard*
    # should sit in the tail and rarely bind, so p99/p99.9 is how you pick
    # max_grad_norm from data instead of guessing, and bind_rate is how you tell a
    # guard (rare) from a de-facto step limiter (frequent).
    run_gnorms = []

    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()
        obs_buf, act_buf, pre_tanh_buf = [], [], []
        rew_buf, done_buf, next_obs_buf = [], [], []

        for _ in range(Config.unroll_length):
            obs = _flat_obs(env_state.obs)
            params = pcn_forward(policy_model, jnp.asarray(obs))
            if state_indep:
                params = _with_global_std(params)
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

        # fixed probe slice for collapse diagnostics (pre/post-update policy)
        n_probe = min(2048, batch_size)
        probe_obs = jnp.asarray(observations[:n_probe])
        params_pre = pcn_forward(policy_model, probe_obs) if continuous else None
        if state_indep:
            params_pre = _with_global_std(params_pre)  # pre-update global std

        # accumulate the PC policy-gradient norm across every weight update this
        # step (exploding-gradient diagnostic; max is the spike to watch/clip).
        pgn_max = jnp.array(0.0)
        pgn_sum = jnp.array(0.0)
        pgn_cnt = 0
        mb_gnorms = []  # this step's norms, stacked once to avoid per-mb syncs
        precision_betas = []
        precision_radii = []
        precision_residuals = []

        for _ in range(Config.update_epochs):
            key, key_perm = jr.split(key)
            perm = np.asarray(jr.permutation(key_perm, batch_size))[:usable]
            for mb_idx in perm.reshape(mb_count, -1):
                mb_obs = jnp.asarray(observations[mb_idx])
                mb_adv = jnp.asarray(advantages[mb_idx])

                # value regresses the fixed lambda-returns
                for _ in range(Config.pc_steps_per_update):
                    _value_step = (make_pc_step_at_rate
                                   if Config.inference_rate_correction
                                   else jpc.make_pc_step)
                    value_result = _value_step(
                        model=value_model,
                        optim=value_optim,
                        opt_state=value_opt_state,
                        output=jnp.asarray(value_targets[mb_idx])[:, None],
                        input=mb_obs,
                        max_t1=Config.max_t1,
                    )
                    value_model, value_opt_state = (
                        value_result["model"], value_result["opt_state"])

                # Parameter-PC has one shared free displacement for the entire
                # rollout. It is inferred below after value minibatch updates.
                if use_parameter:
                    continue

                # policy targets recomputed from the current policy
                params_mb = pcn_forward(policy_model, mb_obs)
                if continuous and state_indep:
                    mb_pre_tanh = jnp.asarray(pre_tanh_flat[mb_idx])
                    params_mb_std = _with_global_std(params_mb)
                    # mean target uses the global sigma; std head is frozen by
                    # targeting its own current output (zero error there).
                    policy_targets = gaussian_pc_targets(
                        params_mb_std, mb_pre_tanh, mb_adv,
                        action_size, Config.target_scale, exp_std=Config.exp_std,
                        target_clip=Config.target_clip,
                        target_clip_rel=Config.target_clip_rel,
                        natural_target=Config.natural_target)
                    policy_targets = policy_targets.at[:, action_size:].set(
                        params_mb[:, action_size:])
                    # global log_std <- batch-averaged Gaussian score on log_std,
                    # the same signal PPO's state-independent log_std receives.
                    loc_mb, scale_mb, _ = split_gaussian_params(
                        params_mb_std, action_size, exp_std=Config.exp_std)
                    z_mb = (mb_pre_tanh - loc_mb) / scale_mb
                    dstd = Config.target_scale * (
                        mb_adv[:, None] * (jnp.square(z_mb) - 1.0)).mean(0)
                    if Config.natural_target:
                        dstd = dstd * 0.5  # F^-1 on log_std channel
                    policy_log_std = jnp.clip(
                        policy_log_std + dstd,
                        gpol.LOG_STD_MIN,
                        gpol.LOG_STD_MAX)
                elif continuous and use_likelihood:
                    policy_targets = None      # no target: the energy carries it
                elif continuous and use_precision:
                    mb_pre_tanh = jnp.asarray(pre_tanh_flat[mb_idx])
                    if Config.policy_geometry == 'precision_tr':
                        beta = trust_region_precision(
                            params_mb, mb_pre_tanh, mb_adv, action_size,
                            Config.precision_max_radius,
                            target_scale=Config.target_scale,
                            exp_std=Config.exp_std)
                    else:
                        beta = jnp.asarray(Config.precision_beta, dtype=params_mb.dtype)
                    precision_result = settle_precision_outputs(
                        params_mb, mb_pre_tanh, mb_adv,
                        action_dim=action_size,
                        beta=beta,
                        max_t1=Config.max_t1,
                        target_scale=Config.target_scale,
                        exp_std=Config.exp_std)
                    policy_targets = precision_result["targets"]
                    precision_betas.append(precision_result["beta"])
                    precision_radii.append(precision_result["radius"])
                    precision_residuals.append(precision_result["residual_rms"])
                elif continuous:
                    policy_targets = gaussian_pc_targets(
                        params_mb, jnp.asarray(pre_tanh_flat[mb_idx]), mb_adv,
                        action_size, Config.target_scale, exp_std=Config.exp_std,
                        target_clip=Config.target_clip,
                        target_clip_rel=Config.target_clip_rel,
                        natural_target=Config.natural_target)
                else:
                    policy_targets = discrete_pc_targets(
                        params_mb, jnp.asarray(actions_flat[mb_idx]).astype(jnp.int32),
                        mb_adv, action_size, Config.target_scale)
                for _ in range(Config.pc_steps_per_update):
                    if use_likelihood:
                        policy_result = make_likelihood_pc_step(
                            policy_model, policy_optim, policy_opt_state,
                            mb_obs, jnp.asarray(pre_tanh_flat[mb_idx]), mb_adv,
                            action_dim=action_size,
                            max_t1=Config.max_t1,
                            exp_std=Config.exp_std,
                            adv_mode=Config.likelihood_adv_mode,
                            tau=Config.likelihood_tau,
                        )
                    elif Config.inference_rate_correction:
                        policy_result = make_pc_step_at_rate(
                            model=policy_model,
                            optim=policy_optim,
                            opt_state=policy_opt_state,
                            output=policy_targets,
                            input=mb_obs,
                            max_t1=Config.max_t1,
                            grad_norms=True,
                        )
                    else:
                        policy_result = jpc.make_pc_step(
                            model=policy_model,
                            optim=policy_optim,
                            opt_state=policy_opt_state,
                            output=policy_targets,
                            input=mb_obs,
                            max_t1=Config.max_t1,
                            grad_norms=True,
                        )
                    policy_model, policy_opt_state = (
                        policy_result["model"], policy_result["opt_state"])
                    gnorm = _global_norm(policy_result["model_grad_norms"])
                    pgn_max = jnp.maximum(pgn_max, gnorm)
                    pgn_sum = pgn_sum + gnorm
                    pgn_cnt += 1
                    mb_gnorms.append(gnorm)

        if use_parameter:
            policy_result = make_parameter_pc_policy_step(
                policy_model,
                jnp.asarray(observations),
                jnp.asarray(pre_tanh_flat),
                jnp.asarray(advantages),
                action_size,
                mode=Config.policy_geometry,
                damping=Config.parameter_pc_damping,
                step_size=Config.parameter_pc_step_size,
                max_radius=Config.parameter_pc_max_radius,
                max_kl=Config.parameter_pc_max_kl,
                cg_iters=Config.parameter_pc_cg_iters,
                cg_tol=Config.parameter_pc_cg_tol,
                joint_steps=Config.parameter_pc_joint_steps,
                joint_parameter_rate=Config.parameter_pc_joint_parameter_rate,
                joint_activity_rate=Config.parameter_pc_joint_activity_rate,
                joint_error_rate=Config.parameter_pc_joint_error_rate,
                joint_penalty=Config.parameter_pc_joint_penalty,
                joint_precision_rate=Config.parameter_pc_joint_precision_rate,
                joint_precision_min=Config.parameter_pc_joint_precision_min,
                joint_precision_max=Config.parameter_pc_joint_precision_max,
                exact_kl_steps=Config.parameter_pc_exact_kl_steps,
                exact_kl_primal_rate=Config.parameter_pc_exact_kl_primal_rate,
                exact_kl_dual_rate=Config.parameter_pc_exact_kl_dual_rate,
                exact_kl_penalty=Config.parameter_pc_exact_kl_penalty,
                exact_kl_tol=Config.parameter_pc_exact_kl_tol,
                policy_log_std=(policy_log_std if state_indep else None),
            )
            policy_model = policy_result["model"]
            if state_indep:
                policy_log_std = jnp.clip(
                    policy_result["policy_log_std"],
                    gpol.LOG_STD_MIN,
                    gpol.LOG_STD_MAX,
                )
            gnorm = policy_result["gradient_norm"]
            pgn_max = gnorm
            pgn_sum = gnorm
            pgn_cnt = 1
            mb_gnorms.append(gnorm)

        global_step += env_step_per_training_step
        if mb_gnorms:
            run_gnorms.extend(np.asarray(jnp.stack(mb_gnorms)).tolist())
        gn_hist = np.asarray(run_gnorms) if run_gnorms else np.zeros(1)
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
            'diag/policy_grad_norm_max': float(pgn_max),
            'diag/policy_grad_norm_mean': float(pgn_sum / max(pgn_cnt, 1)),
            # cumulative tail of the grad-norm distribution (all updates so far):
            # read p99/p999 off a clip-free run to *choose* max_grad_norm.
            'diag/policy_grad_norm_p50': float(np.percentile(gn_hist, 50)),
            'diag/policy_grad_norm_p90': float(np.percentile(gn_hist, 90)),
            'diag/policy_grad_norm_p99': float(np.percentile(gn_hist, 99)),
            'diag/policy_grad_norm_p999': float(np.percentile(gn_hist, 99.9)),
            # fraction of ALL weight updates so far whose pre-clip norm would hit
            # max_grad_norm. Rare (<1%) = numerical guard; frequent (>10%) = the
            # clip is really acting as a step limiter.
            'diag/policy_grad_norm_bind_rate': (
                float((gn_hist >= float(Config.max_grad_norm)).mean())
                if Config.max_grad_norm is not None else 0.0),
            'diag/value_explained_var': float(
                1.0 - np.var(value_targets - values) / (np.var(value_targets) + 1e-8)),
        }
        if precision_betas:
            metrics.update({
                'diag/precision_beta_mean': float(jnp.mean(jnp.stack(precision_betas))),
                'diag/precision_radius_mean': float(jnp.mean(jnp.stack(precision_radii))),
                'diag/precision_radius_max': float(jnp.max(jnp.stack(precision_radii))),
                'diag/precision_residual_rms_max': float(
                    jnp.max(jnp.stack(precision_residuals))),
            })
        if use_parameter:
            metrics.update({
                'diag/parameter_pc_kkt': float(policy_result['relative_kkt']),
                'diag/parameter_pc_fisher_cosine': float(
                    policy_result['fisher_equation_cosine']),
                'diag/parameter_pc_direction_norm': float(
                    policy_result['direction_norm']),
                'diag/parameter_pc_quadratic_radius': float(
                    policy_result['quadratic_radius']),
                'diag/parameter_pc_actual_kl': float(policy_result['actual_kl']),
                'diag/parameter_pc_step_scale': float(policy_result['step_scale']),
                'diag/parameter_pc_precision_beta': float(
                    policy_result['precision_beta']),
                'diag/parameter_pc_objective_before': float(
                    policy_result['objective_before']),
                'diag/parameter_pc_objective_after': float(
                    policy_result['objective_after']),
                'diag/parameter_pc_exact_kl_constraint': float(
                    policy_result['exact_kl_constraint']),
                'diag/parameter_pc_exact_kl_inference_residual': float(
                    policy_result['exact_kl_inference_residual']),
                'diag/parameter_pc_exact_kl_scalar_kkt': float(
                    policy_result['exact_kl_scalar_kkt']),
                'diag/parameter_pc_exact_kl_inference_steps': int(
                    policy_result['exact_kl_inference_steps']),
                'diag/parameter_pc_exact_kl_local_scale': float(
                    policy_result['exact_kl_local_scale']),
                'diag/parameter_pc_joint_constraint_rms': float(
                    policy_result['joint_constraint_rms']),
                'diag/parameter_pc_joint_precision_mean': float(
                    policy_result['joint_precision_mean']),
                'diag/parameter_pc_joint_inference_steps': int(
                    policy_result['joint_inference_steps']),
            })
        if continuous:
            params_post = pcn_forward(policy_model, probe_obs)
            if state_indep:
                params_post = _with_global_std(params_post)  # post-update global std
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
            if use_parameter:
                probe_targets = None
            elif use_precision:
                probe_pre_tanh = jnp.asarray(pre_tanh_flat[:n_probe])
                probe_adv = jnp.asarray(advantages[:n_probe])
                if Config.policy_geometry == 'precision_tr':
                    probe_beta = trust_region_precision(
                        params_pre, probe_pre_tanh, probe_adv, action_size,
                        Config.precision_max_radius,
                        target_scale=Config.target_scale,
                        exp_std=Config.exp_std)
                else:
                    probe_beta = jnp.asarray(
                        Config.precision_beta, dtype=params_pre.dtype)
                probe_targets = (
                    effective_policy_outputs(
                        params_pre, action_size, exp_std=Config.exp_std)
                    + natural_output_displacement(
                        params_pre, probe_pre_tanh, probe_adv, action_size,
                        target_scale=Config.target_scale,
                        exp_std=Config.exp_std) / probe_beta)
            else:
                probe_targets = gaussian_pc_targets(
                    params_pre, jnp.asarray(pre_tanh_flat[:n_probe]),
                    jnp.asarray(advantages[:n_probe]), action_size,
                    Config.target_scale, exp_std=Config.exp_std,
                    natural_target=Config.natural_target)
            metrics.update({
                'diag/log_std_mean': float(log_std_post.mean()),
                'diag/log_std_min': float(log_std_post.min()),
                'diag/frac_std_at_min': float((log_std_post <= gpol.LOG_STD_MIN + 1e-3).mean()),
                'diag/frac_std_at_max': float(
                    (log_std_post >= gpol.LOG_STD_MAX - 1e-3).mean()),
                'diag/mu_abs_mean': float(jnp.abs(loc_post).mean()),
                'diag/policy_drift_mean': float(drift.mean()),
                'diag/policy_drift_max': float(drift.max()),
                'diag/policy_kl_mean': float(policy_kl.mean()),
                'diag/policy_kl_max': float(policy_kl.max()),
                'diag/pretanh_sat_frac': float((np.abs(pre_tanh_flat) > 2.0).mean()),
            })
            if probe_targets is not None:
                mu_target_mag = jnp.abs(
                    probe_targets[:, :action_size] - loc_pre)
                metrics.update({
                    'diag/mu_target_mag_mean': float(mu_target_mag.mean()),
                    'diag/mu_target_mag_max': float(mu_target_mag.max()),
                })
        logging.info(metrics)

        if Config.eval_env and training_step % Config.eval_every == 0:
            eval_time_start = time.time()
            if continuous:
                eval_returns, eval_ep_lengths, eval_key = evaluate_gaussian_policy(
                    eval_env, _eval_policy_forward, policy_model, action_size, _flat_obs,
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
