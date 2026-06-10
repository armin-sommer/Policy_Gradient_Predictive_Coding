"""Run bandit comparison across multiple RNG seeds."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = list(range(1, 11))


def seed_complete(seed: int, results_root: Path) -> bool:
    csv_path = results_root / f"bandit_seed{seed}" / "pi_optimal.csv"
    if not csv_path.exists():
        return False
    lines = csv_path.read_text().strip().splitlines()
    return len(lines) > 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--total-steps", type=int, default=60_000)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    script = REPO_ROOT / "scripts" / "run_bandit_comparison.py"
    python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)

    results_root = REPO_ROOT / "results"
    t0 = time.time()
    ran, skipped = [], []

    for seed in args.seeds:
        if args.skip_existing and seed_complete(seed, results_root):
            print(f"seed {seed}: skip (pi_optimal.csv exists)")
            skipped.append(seed)
            continue
        print(f"\n######## seed {seed} ########\n")
        subprocess.run(
            [
                str(python),
                str(script),
                "--seed", str(seed),
                "--total-steps", str(args.total_steps),
                "--no-plot",
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        ran.append(seed)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Ran {len(ran)} seeds, skipped {len(skipped)}.")
    if ran:
        print("ran:", ran)
    if skipped:
        print("skipped:", skipped)


if __name__ == "__main__":
    main()
