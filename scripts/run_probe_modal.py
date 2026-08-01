"""Run the natural-gradient probe (scripts/probe_natural_gradient.py) on Modal.

One-time setup on your machine:

    pip install modal
    modal token new        # or export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...

Run:

    modal run scripts/run_probe_modal.py                          # full pilot
    modal run scripts/run_probe_modal.py --probe-args "--regimes floor --n 64"
    modal run scripts/run_probe_modal.py --seeds "0 1 2"          # seed fan-out
    modal run scripts/run_probe_modal.py --seeds "0 1 2" \
        --probe-args "--n 2048 --t1 0 20 4096"                    # bench-size N

Output streams to your terminal; every run's full log is also persisted to the
`pcpg-probe-results` Modal volume:

    modal volume ls pcpg-probe-results
    modal volume get pcpg-probe-results <name>.log

The probe is CPU-only by design (tiny nets, matrix-free CG); bump `cpu=` or add
`gpu="T4"` to the function decorator only if you scale --n / --width far past
the bench geometry. The container ships the repo's `src/` and the probe script,
and installs jpc from git exactly as pyproject.toml pins it.
"""

import shlex
import subprocess
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_REMOTE = "/repo"

app = modal.App("pcpg-ng-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # jpc is installed straight from GitHub
    .pip_install("jpc @ git+https://github.com/thebuckleylab/jpc")
    .add_local_dir(str(REPO_ROOT / "src"), remote_path=f"{REPO_REMOTE}/src")
    .add_local_file(str(REPO_ROOT / "scripts" / "probe_natural_gradient.py"),
                    f"{REPO_REMOTE}/scripts/probe_natural_gradient.py")
)

results = modal.Volume.from_name("pcpg-probe-results", create_if_missing=True)


@app.function(image=image, cpu=8.0, memory=16384, timeout=3600,
              volumes={"/results": results})
def run_probe(argv: str = "") -> str:
    cmd = ["python", f"{REPO_REMOTE}/scripts/probe_natural_gradient.py",
           *shlex.split(argv)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "_".join(shlex.split(argv)).replace("-", "").replace("/", "") or "pilot"
    log_path = Path(f"/results/probe_{stamp}_{tag[:60]}.log")
    log_path.write_text(f"$ {' '.join(cmd)}\n\n{text}")
    results.commit()
    print(text)
    if proc.returncode != 0:
        raise RuntimeError(f"probe exited {proc.returncode}; log: {log_path.name}")
    return text


@app.local_entrypoint()
def main(probe_args: str = "", seeds: str = ""):
    """Single run by default; pass --seeds "0 1 2" to fan out in parallel."""
    if seeds:
        argvs = [f"--seed {s} {probe_args}".strip() for s in seeds.split()]
        for argv, _ in zip(argvs, run_probe.map(argvs)):
            print(f"[done] probe {argv}")
    else:
        run_probe.remote(probe_args)
