"""An exact regime where a PC target update is a Gaussian natural gradient.

This is deliberately a tabular one-state policy: its trainable parameters ARE
the Gaussian output (mu, log_sigma), so no shared hidden network Jacobian
separates output space from parameter space.  With an unclipped natural target,
the squared-error PC update is exactly F_out^-1 times the policy gradient.

It is the strongest setting in which the current target construction can be
called an NPG.  It does not claim that latent inference in a shared MLP supplies
the parameter-space Fisher inverse; `probe_settling_budget.py` tests that claim
and rejects it on MuJoCo.
"""

import numpy as np


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    rng = np.random.default_rng(0)
    n = 4096
    mu = np.array([0.3, -0.5])
    log_std = np.array([-0.2, 0.4])
    sigma = np.exp(log_std)
    pre_tanh = mu + sigma * rng.normal(size=(n, mu.size))
    advantages = rng.normal(size=n)

    z_score = (pre_tanh - mu) / sigma
    # Gradient of loss L = -E[A log pi] in output coordinates (mu, log_std).
    grad_loss = -np.concatenate([
        np.mean(advantages[:, None] * z_score / sigma, axis=0),
        np.mean(advantages[:, None] * (z_score**2 - 1.0), axis=0),
    ])

    # Exact Gaussian Fisher for (mu, log_std), averaged over the batch.
    fisher = np.diag(np.concatenate([1.0 / sigma**2,
                                     2.0 * np.ones_like(sigma)]))
    natural_gradient = -np.linalg.solve(fisher, grad_loss)

    # PC energy E = 1/(2N) sum ||out - target_i||^2.  The natural target is
    # F_out^-1 times the score target. Its negative output gradient is exactly
    # the natural-gradient direction above when there are no hidden parameters
    # and no target clipping.
    target_offset = np.concatenate([
        advantages[:, None] * (pre_tanh - mu),
        0.5 * advantages[:, None] * (z_score**2 - 1.0),
    ], axis=-1)
    pc_update = np.mean(target_offset, axis=0)

    print("one-state Gaussian, direct-output PC regime")
    print(f"cos(PC, NPG)       = {cosine(pc_update, natural_gradient):.12f}")
    print(f"max |PC - NPG|     = {np.max(np.abs(pc_update - natural_gradient)):.3e}")
    assert np.allclose(pc_update, natural_gradient, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    main()
