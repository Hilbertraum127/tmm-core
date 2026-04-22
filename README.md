# tmm-core

[![Tests](https://github.com/Hilbertraum127/tmm-core/actions/workflows/tests.yml/badge.svg)](https://github.com/Hilbertraum127/tmm-core/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A transfer-matrix method (TMM) implementation for the optical response of
planar multilayer stacks. Convention in which $r_s = r_p$ at normal
incidence, complex refractive indices, arbitrary angle of incidence, both
polarisations, and per-layer absorption with machine-precision energy
conservation.

## Install

```bash
pip install git+https://github.com/Hilbertraum127/tmm-core.git
```

For the worked examples:

```bash
pip install "git+https://github.com/Hilbertraum127/tmm-core.git#egg=tmm-core[examples]"
```

## Quickstart

```python
from tmm_core import tmm_full

out = tmm_full(
    n_list=[1.0, 2.35, 1.38, 2.35, 1.38, 2.35, 1.52],      # air / HLHLH / glass quarter-wave stack
    d_list=[58.5, 99.6, 58.5, 99.6, 58.5],                 # quarter-wave at 550 nm
    wl=550.0,
    theta_0_deg=0.0,
    pol='s',
)
print(f"R = {out['R']:.4f}, T = {out['T']:.4f}, A = {out['A'].sum():.2e}")
```

## Documentation

- [`docs/theory/theory.pdf`](docs/theory/theory.pdf) — step-by-step derivation (compiled from `theory.tex`)
- [`docs/examples/examples.pdf`](docs/examples/examples.pdf) — sixteen worked analytical benchmarks with plots
- [`docs/usage.md`](docs/usage.md) — API, conventions, pitfalls
- [`docs/nomenclature.md`](docs/nomenclature.md) — symbol reference
- [`examples/`](examples/) — sixteen runnable Python scripts

## Scope

Coherent, planar, isotropic, non-magnetic multilayers. Passive media only
(`Im(n) ≥ 0`). Incoherent substrates, magnetic media, and anisotropy are
not supported.

## Citation

If you use this package in academic work, please cite via the
[`CITATION.cff`](CITATION.cff) metadata (GitHub's "Cite this repository"
button renders it).

## License

MIT. See [`LICENSE`](LICENSE).
