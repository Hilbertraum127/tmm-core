"""
12 - Half-wave invisible layer and quarter-wave AR.

Two analytical identities at a single design wavelength lambda0:

  (a) Half-wave invisible layer: a lossless film of thickness
      d = lambda0 / (2 n_f) returns exactly the bare-substrate
      reflectance at lambda0.

  (b) Quarter-wave antireflection with n_f = sqrt(n0 n_s) and
      d = lambda0 / (4 n_f) gives exactly R = 0 at lambda0. At other
      wavelengths R is a smooth V-shape touching zero at lambda0.

Both panels overlay TMM (colored solid) against the bare-substrate
reflectance (dashed black) and mark lambda0 with a dotted line.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

wl0 = 550.0
wl = np.linspace(400.0, 800.0, 801)

# --- (a) half-wave invisible layer ---
n0_a, n_f_a, n_s_a = 1.0, 2.0, 1.5
d_hw = wl0 / (2 * n_f_a)
R_film_a = np.array([tmm_R([n0_a, n_f_a, n_s_a], [d_hw], w, 0.0, 's') for w in wl])
R_bare_a = np.array([tmm_R([n0_a, n_s_a], [], w, 0.0, 's') for w in wl])
R_bare_a_l0 = tmm_R([n0_a, n_s_a], [], wl0, 0.0, 's')
R_film_a_l0 = tmm_R([n0_a, n_f_a, n_s_a], [d_hw], wl0, 0.0, 's')
print(f"(a) half-wave: R_bare = {R_bare_a_l0:.6e},"
      f" R_film(lambda0) = {R_film_a_l0:.6e},"
      f" diff = {abs(R_film_a_l0 - R_bare_a_l0):.2e}")

# --- (b) quarter-wave AR ---
n0_b, n_s_b = 1.0, 1.5
n_f_b = np.sqrt(n0_b * n_s_b)
d_qw = wl0 / (4 * n_f_b)
R_film_b = np.array([tmm_R([n0_b, n_f_b, n_s_b], [d_qw], w, 0.0, 's') for w in wl])
R_bare_b = np.array([tmm_R([n0_b, n_s_b], [], w, 0.0, 's') for w in wl])
R_film_b_l0 = tmm_R([n0_b, n_f_b, n_s_b], [d_qw], wl0, 0.0, 's')
print(f"(b) quarter-wave AR: n_f = sqrt(n_s) = {n_f_b:.4f},"
      f" R(lambda0) = {R_film_b_l0:.2e}")

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.plot(wl, R_film_a, color=CB10[0], lw=1.0, label=r'TMM, $d=\lambda_0/2n_f$')
ax1.plot(wl, R_bare_a, color='k', ls='--', lw=0.7, label='bare substrate')
ax1.axvline(wl0, color=CB10[9], ls=':', lw=0.5)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Reflectance")
ax1.text(0.97, 0.97,
         f"$n_0={n0_a:g}$, $n_f={n_f_a:g}$, $n_s={n_s_a:g}$\n"
         f"$d_\\mathrm{{HW}} = {d_hw:.1f}$ nm",
         transform=ax1.transAxes, fontsize=6, ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
ax1.legend(loc='upper left', fontsize=6)
add_panel_label(ax1, 'a')

ax2.plot(wl, R_film_b, color=CB10[0], lw=1.0,
         label=r'TMM, $n_f=\sqrt{n_0 n_s}$, $d=\lambda_0/4n_f$')
ax2.plot(wl, R_bare_b, color='k', ls='--', lw=0.7, label='bare substrate')
ax2.axvline(wl0, color=CB10[9], ls=':', lw=0.5)
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel("Reflectance")
ax2.text(0.97, 0.97,
         f"$n_0={n0_b:g}$, $n_f={n_f_b:.4f}$, $n_s={n_s_b:g}$\n"
         f"$d_\\mathrm{{QW}} = {d_qw:.1f}$ nm",
         transform=ax2.transAxes, fontsize=6, ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
ax2.legend(loc='upper left', fontsize=6)
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '05_halfwave_quarterwave.pdf'
finalize(fig, out)
