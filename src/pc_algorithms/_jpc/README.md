# Vendored JPC

This directory contains a vendored copy of **JPC** (the JAX library for Predictive
Coding from the Buckley Lab) used as the PC backend for PCPG in this repository.

- **Upstream**: https://github.com/thebuckleylab/jpc
- **Vendored version**: 1.0.0
- **Citation**: Innocenti, F. et al. *JPC: Flexible Inference for Predictive Coding
  Networks in JAX.* arXiv:2412.03676 (2024).
- **License**: MIT (see `LICENSE`).

## Why vendored

PCPG depends on JPC's activity-relaxation inference and parameter-update primitives
(`update_pc_activities`, `update_pc_params`). Vendoring keeps the PC math present
and inspectable inside this repo so the PCPG implementation is fully reproducible
from this checkout alone, without relying on an upstream pip release.

## How it is wired up

`src/pc_algorithms/__init__.py` prepends this directory (`src/pc_algorithms/_jpc/`)
to `sys.path` at import time, so `import jpc` anywhere inside the project resolves
to the package under `_jpc/jpc/`. No pip install of `jpc` is needed.

## Updating

To bump the vendored version: re-copy from a clean install of the desired tag,
preserve this README, refresh `LICENSE` if upstream changed it, and re-run the
parity check and `scripts/mdp_v1_tier1.py` to confirm numerics.
