"""Policy init presets for the 2-armed bandit."""

import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INIT = "favor_suboptimal"
LEGACY_INIT_DIRS = {"favor_suboptimal": "bandit_seed"}

# pi(arm0) = sigmoid(b0 - b1) with zeroed final kernel; arm 0 is optimal.
INIT_PRESETS = {
    "favor_suboptimal": [0.0, 4.0],
    "uniform": [0.0, 0.0],
    "mild_suboptimal": [0.0, 1.0],
    "mild_optimal": [1.0, 0.0],
    "favor_optimal": [4.0, 0.0],
}


def pi_optimal(logit_bias):
    b0, b1 = logit_bias
    return 1.0 / (1.0 + math.exp(b1 - b0))


def load_inits():
    return {
        name: {"logit_bias": list(bias), "pi_opt": pi_optimal(bias)}
        for name, bias in INIT_PRESETS.items()
    }


def result_dir(init_name, seed, results_root=None):
    results_root = Path(results_root) if results_root else REPO_ROOT / "results"
    legacy_prefix = LEGACY_INIT_DIRS.get(init_name)
    if legacy_prefix:
        legacy = results_root / f"{legacy_prefix}{seed}"
        if legacy.exists():
            return legacy
    return results_root / f"bandit_{init_name}_seed{seed}"
