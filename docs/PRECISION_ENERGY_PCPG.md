# Precision-Energy NPG-PCPG and TR-PCPG

## Construction

For each rollout sample, the policy output activity is

$$
z_i=(\mu_i,\ell_i), \qquad \ell_i=\log\sigma_i,
$$

and is initialized from the effective (log-standard-deviation-clipped) old
policy output $z_i^0$. Let

$$
u_i=A_i\frac{\partial\log\pi(a_i\mid s_i)}{\partial z_i}
$$

be the output policy-gradient drive. The additive precision energy is

$$
F_{\mathrm{out}}(z)
=\frac1N\sum_i\left[
-u_{\mu,i}^{\mathsf T}\Delta\mu_i
-u_{\ell,i}^{\mathsf T}\Delta\ell_i
+\frac\beta2\left\lVert\frac{\Delta\mu_i}{\sigma_i^0}\right\rVert^2
+\beta\lVert\Delta\ell_i\rVert^2
\right].
$$

Settled output inference therefore gives

$$
\Delta z_i^*=\frac1\beta F_{\mathrm{out},i}^{-1}u_i,
\qquad
F_{\mathrm{out},i}=\operatorname{diag}(1/(\sigma_i^0)^2,2).
$$

No Fisher matrix or KL is constructed. The inverse-Fisher action is the fixed
point of ordinary additive precision errors. The settled output is then clamped
as the target for the existing rate-corrected hidden-activity JPC inference and
local weight update.

The two stages are intentionally separate. If the network-output prediction
error were included in the output inference energy, it would add identity
curvature and change the equilibrium to $(I+\beta F_{\mathrm{out}})^{-1}u$.

## Modes

`policy_geometry: precision_npg` uses a fixed scalar `precision_beta`. This
changes step length but not the output-natural direction.

`policy_geometry: precision_tr` chooses `beta` per minibatch so that the settled
activity reaches the additive quadratic radius

$$
R(z)=\frac1N\sum_i\left[
\frac12\left\lVert\frac{\Delta\mu_i}{\sigma_i^0}\right\rVert^2
+\lVert\Delta\ell_i\rVert^2
\right]=\epsilon.
$$

The precision solve accounts for the production $\log\sigma\in[-2,2]$ box. Its
reported inference residual is consequently the projected KKT residual.

This is an output/activity-space NPG and trust region in the sense of the
Innocenti analysis. With a shared MLP, the subsequent local weight update is not
guaranteed to equal parameter-space NPG, and the activity radius is not the same
as the actual post-weight-update policy KL. The trainer logs both quantities.

## Validation

Synthetic GPU checks:

| Check | Result |
| --- | ---: |
| Fixed-precision equilibrium max error vs analytic NPG | `8.37e-6` |
| Fixed-precision projected residual RMS | `3.13e-6` |
| Requested TR radius | `0.015000` |
| Settled TR radius | `0.015000` |
| TR projected residual RMS | `2.24e-6` |

Matched real HalfCheetah checks used 256 environments, rollout length 32,
width 64, one hidden layer, tanh, two epochs, four minibatches, and 131,072
environment steps (16 updates), seed 1.

| Mode | Final activity radius | Final residual | Final actual mean KL | Final evaluation return |
| --- | ---: | ---: | ---: | ---: |
| fixed NPG, $\beta=24$ | `0.0100603` | `2.35e-4` | `5.39e-4` | `-4.4309` |
| TR, $\epsilon=0.01$ | `0.00999999` | `6.22e-4` | `5.57e-4` | `-4.4974` |

These short runs validate inference, constraint enforcement, integration, and
finite training. They are not long enough to establish learning performance.

## Entry Points

- Implementation: `src/pc_algorithms/precision_energy.py`
- Trainer integration: `src/pc_algorithms/pc_actor_critic.py`
- Synthetic check: `scripts/test_precision_energy.py`
- NPG config: `configs/mujoco_halfcheetah_pc_actor_critic_npg.yaml`
- TR config: `configs/mujoco_halfcheetah_pc_actor_critic_tr.yaml`
