"""Run bandit comparison across policy inits and RNG seeds."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from bandit_inits import DEFAULT_INIT, load_inits, result_dir

DEFAULT_SEEDS = list(range(1, 11))


def run_complete(init_name: str, seed: int, results_root: Path) -> bool:
    csv_path = result_dir(init_name, seed, results_root) / "pi_optimal.csv"
    if not csv_path.exists():
        return False
    lines = csv_path.read_text().strip().splitlines()
    return len(lines) > 1


def main():
    parser = argparse.ArgumentParser()
    inits = load_inits()
    parser.add_argument("--inits", nargs="*", default=list(inits.keys()),
                        choices=list(inits.keys()))
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

    for init_name in args.inits:
        for seed in args.seeds:
            key = f"{init_name}/seed{seed}"
            if args.skip_existing and run_complete(init_name, seed, results_root):
                print(f"{key}: skip (pi_optimal.csv exists)")
                skipped.append(key)
                continue
            print(f"\n######## {init_name} seed {seed} ########\n")
            subprocess.run(
                [
                    str(python),
                    str(script),
                    "--init", init_name,
                    "--seed", str(seed),
                    "--total-steps", str(args.total_steps),
                    "--no-plot",
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            ran.append(key)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Ran {len(ran)}, skipped {len(skipped)}.")
    if ran:
        print("ran:", ran)
    if skipped:
        print("skipped:", skipped)


if __name__ == "__main__":
    main()
