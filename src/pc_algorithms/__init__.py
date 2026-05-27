"""Predictive-coding algorithms for this repository.

Importing from this package transparently makes the vendored JPC available:
`import jpc` resolves to src/pc_algorithms/_jpc/jpc/. See _jpc/README.md.
"""
import sys as _sys
from pathlib import Path as _Path

_VENDOR_DIR = _Path(__file__).resolve().parent / "_jpc"
if str(_VENDOR_DIR) not in _sys.path:
    _sys.path.insert(0, str(_VENDOR_DIR))
