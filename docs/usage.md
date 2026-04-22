# Usage

## Installation

### From GitHub (recommended, no PyPI account needed)

```bash
pip install git+https://github.com/Hilbertraum127/tmm-core.git
```

### Editable clone (for development)

```bash
git clone https://github.com/Hilbertraum127/tmm-core.git
cd tmm-core
pip install -e ".[dev,examples]"
```

## Quickstart

```python
from tmm_core import tmm_R

# Air / glass / air stack, normal incidence, 550 nm
R = tmm_R(
    n_list=[1.0, 1.5, 1.0],
    d_list=[100.0],          # nm
    wl=550.0,                # nm
    theta_0_deg=0.0,
    pol='s',
)
print(f"R = {R:.4f}")
```

## Conventions (critical)

- **Stack ordering.** `n_list[0]` is the incident medium, `n_list[-1]` is
  the substrate. `d_list` contains thicknesses of the finite interior
  layers only: `len(d_list) == len(n_list) - 2`.
- **Units.** `wl` and every entry of `d_list` must be in the same length
  unit. Nanometres are the pragmatic default.
- **Refractive index.** Complex, with non-negative imaginary part for
  absorbing (passive) media: `n = n' + i*κ`, `κ ≥ 0`.
- **Angle.** `theta_0_deg` is measured in the incident medium, in degrees.
- **Polarisation.** `'s'` for TE, `'p'` for TM. At normal incidence
  `R_s = R_p`.
- **Time convention.** `exp(-iωt)` (physics convention). See
  `docs/theory/theory.pdf` for why this matters for absorbing media
  (sign of `Im(n)` and direction of wave decay).

## API

### `tmm_r(n_list, d_list, wl, theta_0_deg=0.0, pol='s') -> complex`

Complex reflection coefficient `r = M_total[1,0] / M_total[0,0]`.

**Raises** `ValueError` if `len(n_list) != len(d_list) + 2` or if `pol`
is not `'s'` or `'p'`.

### `tmm_R(n_list, d_list, wl, theta_0_deg=0.0, pol='s') -> float`

Reflectance `R = |r|²`. Same arguments as `tmm_r`.

### `tmm_full(n_list, d_list, wl, theta_0_deg=0.0, pol='s') -> dict`

Returns a dict with keys `'r'`, `'R'`, `'T'`, `'A'` (numpy array of
per-layer absorbed fractions). Requires a lossless incident medium
(`Im(n_list[0]) == 0`). For passive stacks, `R + T + A.sum() == 1` to
machine precision.

## Common pitfalls

- **Mismatched units.** `wl = 550` with `d_list = [1e-7]` silently produces
  nonsense. Use the same unit everywhere.
- **Wrong stack order.** If the substrate is on the wrong end, reflection
  phase and sign of off-normal `r_p` will be inverted.
- **Gain media.** `κ < 0` is outside the assumed regime and the enforced
  branch cut will produce unphysical results.
- **Incoherent thick substrates.** This package is fully coherent. Thick
  substrates (> coherence length) should be handled by an incoherent-layer
  wrapper; not provided here.

## See also

- `docs/theory/theory.pdf` — step-by-step derivation of every formula
- `docs/examples/examples.pdf` — sixteen worked analytical benchmarks
- `docs/nomenclature.md` — symbol and convention reference
- `examples/` — runnable Python scripts
