"""
tmm_core — Transfer-Matrix Method for planar multilayer optics.

Public API
----------
    tmm_r    -> complex reflection coefficient r
    tmm_R    -> reflectance R = |r|^2
    tmm_full -> dict with r, R, T and per-layer absorption A

See `docs/theory/theory.pdf` for the derivation, `docs/usage.md` for
usage, and `docs/nomenclature.md` for symbol conventions.
"""

from tmm_core.core import tmm_r, tmm_R, tmm_full

__version__ = "1.0.0"
__all__ = ["tmm_r", "tmm_R", "tmm_full", "__version__"]
