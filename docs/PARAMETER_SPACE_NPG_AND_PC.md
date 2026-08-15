# When Predictive Coding Is a Policy Natural Gradient

## Question

There are two distinct statements that are often conflated:

1. PC inference supplies a second-order trust-region correction in *activity*
   space.
2. A PC policy update is a natural gradient in *policy parameter* space.

The first is the local result analysed by Innocenti, Singh, and Buckley,
"Understanding Predictive Coding as an Adaptive Trust-Region Method"
([arXiv:2305.18188](https://arxiv.org/pdf/2305.18188)). The second requires an
additional identity that does not generally hold for a shared neural policy.

## Ordinary PC: an activity-space trust region

For one hidden activity `h`, feedforward prediction `a`, target `y`, and output
`o(h)`, the relevant PC energy is

$$
E(h) = \tfrac12 \lVert h-a\rVert^2 + \tfrac12 \lVert y-o(h)\rVert^2.
$$

Let `r = y-o(a)`, `delta_h = h-a`, and linearise the output as
`o(h) ~= o(a) + B delta_h`, where `B = d o / d h`. Then

$$
E(\delta h) \approx \tfrac12 \lVert\delta h\rVert^2
  + \tfrac12 \lVert r-B\delta h\rVert^2,
\qquad
\delta h^*=(I+B^\mathsf{T}B)^{-1}B^\mathsf{T}r.
$$

Eliminating the activity gives an effective output metric

$$
S^{-1}=(I+BB^\mathsf{T})^{-1}.
$$

This is a local adaptive trust-region/Gauss-Newton effect. It controls hidden
activity displacement, not the KL distance between old and new policies.

## Policy NPG: a parameter-space trust region

For policy output `o_theta(s_i)`, let

$$
J_i=\frac{\partial o_\theta(s_i)}{\partial\theta},
\qquad
F_\theta=\frac1N\sum_i J_i^\mathsf{T}F_{\mathrm{out},i}J_i.
$$

For a diagonal Gaussian with output `(mu, log_sigma)`,

$$
F_{\mathrm{out},i}=\operatorname{diag}(1/\sigma_i^2,2).
$$

This is the quadratic approximation to policy KL:

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\theta+\delta\theta})
\approx \tfrac12\delta\theta^\mathsf{T}F_\theta\delta\theta.
$$

If `u_i = dL/d o_i` and the desired natural output displacement is
`r_i = F_out,i^-1 u_i`, define an energy over one *shared parameter
displacement*:

$$
E_{\mathrm{NPG}}(\delta\theta)
=\frac1{2N}\sum_i\lVert J_i\delta\theta-r_i\rVert^2_{F_{\mathrm{out},i}}
 +\frac\lambda2\lVert\delta\theta\rVert^2.
$$

Its gradient is

$$
\frac{\partial E_{\mathrm{NPG}}}{\partial\delta\theta}
=(F_\theta+\lambda I)\delta\theta-g,
\qquad
g=\frac1N\sum_iJ_i^\mathsf{T}u_i,
$$

and its optimum is exactly the damped natural-gradient descent direction:

$$
\delta\theta^*=(F_\theta+\lambda I)^{-1}g.
$$

At `lambda=0`, `F_theta` is normally singular for an over-parameterised neural
policy, so the correct object is the pseudoinverse solution or a lightly damped
CG solve. This construction is mathematically PC-like inference, but it is not
ordinary neural-activity PC: it requires a globally shared parameter variable
and Fisher-vector products through the whole policy.

## Exact PC=NPG regime

The current target construction is exactly NPG for a one-state/tabular Gaussian
policy whose trainable parameters are the direct outputs `(mu, log_sigma)`. With
an unclipped natural target,

$$
\mathrm{target}_\mu-\mu=A(z-\mu),
\qquad
\mathrm{target}_{\log\sigma}-\log\sigma
=\frac A2\left[\left(\frac{z-\mu}{\sigma}\right)^2-1\right].
$$

The direct-output squared-error PC update is exactly `F_out^-1` times the
Gaussian policy gradient. `scripts/probe_exact_npg_regime.py` verifies the
identity numerically:

```text
cos(PC, NPG)       = 1.000000000000
max |PC - NPG|     = 3.816e-17
```

This regime has no hidden inference. It establishes what the natural target can
provide: the output-Fisher factor. It does not show that hidden activity
inference provides the parameter-Fisher inverse.

## Real HalfCheetah tanh result

Setup: real HalfCheetah PC actor-critic rollout, 256 environments x 32 steps,
first minibatch `N=2048`, width 64, one hidden layer (`depth=2` in jpc), tanh,
and rate-corrected inference. At the initial checkpoint, `tau=20` reaches both
the activity tolerance and the long-horizon PC update:

```text
||dF/dz|| / ||dF/dz||_ffwd = 0.00532
cos(PC update tau=20, tau=100) = 1.000000
```

On that same frozen minibatch, the explicit parameter-space energy above was
solved with exact Fisher-vector products and CG at `lambda=1`. Its relative KKT
residual was `1.08e-4`, confirming the solve. The converged ordinary-PC update
does not equal that parameter-space energy/NPG direction:

| PC target family | cos(PC update, parameter-NPG energy at lambda=1) |
| --- | ---: |
| configured Euclidean target | 0.1702 |
| natural target | 0.1909 |

The full tanh trajectory at updates 0, 20, and 60 also found that the ordinary
PC update and a best-fit damped policy NPG are not consistently aligned. See
`results/settling_budget/tanh_ng_trajectory.json`.

## Implication

Settled tanh PC can retain the activity-space trust-region mechanism from the
Innocenti analysis. It does not thereby become a TRPO/NPG policy optimizer.
To obtain a true policy NPG, infer a shared `delta_theta` under `E_NPG` above,
which is equivalent in computation to a Fisher-vector-product/CG natural-gradient
or TRPO method.
