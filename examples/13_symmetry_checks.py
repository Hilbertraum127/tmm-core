"""
14 - Symmetry and energy-conservation checks.

(a) Reciprocity (Lorentz): for any stack (including absorbing), the
    intensity transmittance T is identical under stack reversal. The
    amplitude t is NOT reciprocal (it picks up a factor of the index
    ratio); only T = (adm_out/adm_in) |t|^2 is.
(b) Convention A: r_s(0) = r_p(0) for arbitrary stacks at normal incidence.
(c) Energy conservation: R + T = 1 for a lossless Bragg stack over a
    theta sweep, in both polarizations.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_r, tmm_full
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

rng = np.random.default_rng(0)


# --- (a) reciprocity of T for an asymmetric absorbing stack over wavelength ---
n_list = [1.0, 1.5 + 0.02j, 2.5 + 0.1j, 3.5 + 0.05j, 1.8]
d_list = [120.0, 80.0, 50.0]
wl_arr = np.linspace(400.0, 1000.0, 601)
# Note: tmm_full requires lossless incident medium on BOTH sides (for reversal),
# so we ensure n_list[0] and n_list[-1] are real. They are (1.0 and 1.8).

T_fwd = np.array([tmm_full(n_list, d_list, w, 0.0, 's')['T'] for w in wl_arr])
T_rev = np.array([tmm_full(n_list[::-1], d_list[::-1], w, 0.0, 's')['T'] for w in wl_arr])
rec_res = np.abs(T_fwd - T_rev)
print(f"(a) max |T_fwd - T_rev| (absorbing interior) = {rec_res.max():.2e}")

# --- (b) r_s(0) = r_p(0) for random stacks at normal incidence ---
diffs = []
for _ in range(20):
    L = int(rng.integers(1, 5))
    n_rand = [1.0] + [(1 + rng.uniform(0, 3) + 1j * rng.uniform(0, 0.2)) for _ in range(L)] + [1.5]
    d_rand = [rng.uniform(20.0, 300.0) for _ in range(L)]
    r_s = tmm_r(n_rand, d_rand, 600.0, 0.0, 's')
    r_p = tmm_r(n_rand, d_rand, 600.0, 0.0, 'p')
    diffs.append(abs(r_s - r_p))
diffs = np.array(diffs)
print(f"(b) max |r_s(0) - r_p(0)| over {len(diffs)} random stacks = {diffs.max():.2e}")

# --- (c) R+T=1 for lossless Bragg over theta ---
n0, n_H, n_L, n_sub = 1.0, 2.35, 1.46, 1.5
N_pairs = 5
wl0 = 550.0
d_H, d_L = wl0 / (4 * n_H), wl0 / (4 * n_L)
n_bragg = [n0] + [n_H, n_L] * N_pairs + [n_sub]
d_bragg = [d_H, d_L] * N_pairs
theta = np.linspace(0.0, 85.0, 171)

out_s = np.array([[tmm_full(n_bragg, d_bragg, wl0, t, 's')[k] for k in ('R', 'T')]
                  for t in theta])
out_p = np.array([[tmm_full(n_bragg, d_bragg, wl0, t, 'p')[k] for k in ('R', 'T')]
                  for t in theta])
res_s = np.abs(1.0 - (out_s[:, 0] + out_s[:, 1]))
res_p = np.abs(1.0 - (out_p[:, 0] + out_p[:, 1]))
print(f"(c) max |1 - R - T| lossless Bragg: s = {res_s.max():.2e}, p = {res_p.max():.2e}")

fig, axes = fig_collage(ncols=3, nrows=1, height_cm=6.5)
ax_a, ax_b, ax_c = axes[0]

ax_a.semilogy(wl_arr, rec_res + 1e-300, color=CB10[0], lw=0.7)
ax_a.set_xlabel("Wavelength (nm)")
ax_a.set_ylabel(r"$|T_\mathrm{fwd} - T_\mathrm{rev}|$")
add_panel_label(ax_a, 'a')

ax_b.bar(np.arange(len(diffs)), diffs + 1e-300, color=CB10[1], width=0.8)
ax_b.set_yscale('log')
ax_b.set_xlabel("random stack #")
ax_b.set_ylabel(r"$|r_s(0) - r_p(0)|$")
add_panel_label(ax_b, 'b')

ax_c.semilogy(theta, res_s + 1e-300, color=CB10[0], lw=0.7, label='s')
ax_c.semilogy(theta, res_p + 1e-300, color=CB10[1], lw=0.7, ls='--', label='p')
ax_c.set_xlabel(r"$\theta$ (deg)")
ax_c.set_ylabel(r"$|1 - R - T|$")
ax_c.legend(loc='best')
add_panel_label(ax_c, 'c')

out = Path(__file__).parent / 'figures' / '13_symmetry_checks.pdf'
finalize(fig, out)
