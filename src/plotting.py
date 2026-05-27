"""Four-panel comparison figure: learning curve, KL, theta trajectory, cosine to ref."""
import matplotlib.pyplot as plt
import numpy as np


_DEFAULT_COLORS = {"REINFORCE": "C0", "PPO": "C1", "TRPO": "C2",
                   "PCPG": "C3", "NPG": "k"}


def four_panel(results: dict, *, ref: str = "NPG", out_path: str,
               colors: dict | None = None, title_suffix: str = ""):
    """Render the four-panel figure and save to out_path.

    results: {name: Log}
    ref:     algorithm name whose update direction is the cosine-sim reference.
    """
    colors = colors or _DEFAULT_COLORS
    names = list(results.keys())
    num_steps = np.asarray(results[names[0]].mean_ep_return).shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) learning curves: mean episodic return per iteration, across seeds
    ax = axes[0, 0]
    for name, log in results.items():
        m = np.asarray(log.mean_ep_return).mean(0)
        s = np.asarray(log.mean_ep_return).std(0)
        c = colors.get(name, None)
        ax.plot(m, label=name, color=c)
        ax.fill_between(np.arange(num_steps), m - s, m + s, color=c, alpha=0.15)
    ax.set_title(f"(a) learning curve  mean episodic return {title_suffix}".strip())
    ax.set_xlabel("iteration"); ax.set_ylabel("mean episodic return"); ax.legend()

    # (b) realized KL per step
    ax = axes[0, 1]
    for name, log in results.items():
        ax.plot(np.asarray(log.kl).mean(0), label=name,
                color=colors.get(name, None))
    ax.set_yscale("log")
    ax.set_title("(b) realized KL per step")
    ax.set_xlabel("step"); ax.set_ylabel("KL(pi_t || pi_{t-1})"); ax.legend()

    # (c) PCA-projected parameter trajectory (seed 0)
    ax = axes[1, 0]
    stacked = np.concatenate([np.asarray(results[n].theta[0]) for n in names],
                             axis=0)
    mean = stacked.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(stacked - mean, full_matrices=False)
    basis = Vt[:2]
    for name, log in results.items():
        proj = (np.asarray(log.theta[0]) - mean) @ basis.T
        c = colors.get(name, None)
        ax.plot(proj[:, 0], proj[:, 1], label=name, color=c)
        ax.scatter(proj[0, 0], proj[0, 1], color=c, marker="o")
        ax.scatter(proj[-1, 0], proj[-1, 1], color=c, marker="x")
    ax.set_title("(c) theta trajectory (PCA, seed 0)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()

    # (d) cosine similarity to reference update direction
    ax = axes[1, 1]
    ref_d = np.asarray(results[ref].dtheta)
    late_summary = {}
    late = slice(num_steps // 2, num_steps)
    for name, log in results.items():
        if name == ref:
            continue
        d = np.asarray(log.dtheta)
        cos = (d * ref_d).sum(-1)
        late_summary[name] = float(cos[:, late].mean())
        m = cos.mean(0); s = cos.std(0)
        c = colors.get(name, None)
        ax.plot(m, label=name, color=c)
        ax.fill_between(np.arange(num_steps), m - s, m + s, color=c, alpha=0.15)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"(d) cosine sim of update dir vs {ref}")
    ax.set_xlabel("step"); ax.set_ylabel(f"cos(., {ref})"); ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return late_summary
