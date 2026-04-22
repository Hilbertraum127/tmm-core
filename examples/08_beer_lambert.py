"""
11 - Beer-Lambert depth profile in a metal via per-layer absorption.

A gold half-space is sliced into N = 100 virtual layers of 2 nm each
(total 200 nm, deep enough that the reflectance has converged to the
bulk value). tmm_full returns the absorbed fraction A_j in each slice;
per unit depth this sampled absorption density should follow the
Beer-Lambert profile

    dA/dz = alpha * I_in * exp(-alpha z),    alpha = 2 k0 Im(n),

with I_in = 1 - R (the intensity that enters the metal). Integrating
back, the cumulative absorbed fraction up to depth z is

    A(<z) = (1 - R) * (1 - exp(-alpha z)).

Panels:
  a) per-layer absorption A_j vs midpoint depth z_j, TMM vs the
     Beer-Lambert prediction alpha * dz * (1 - R) * exp(-alpha z_j).
  b) cumulative absorption sum_{j <= k} A_j, TMM vs
     (1 - R) * (1 - exp(-alpha z)).
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_full
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

wl = 550.0
n_au = 0.425 + 2.367j        # Au at 550 nm (Johnson-Christy)
k0 = 2 * np.pi / wl
alpha = 2 * k0 * n_au.imag

N_slices = 100
dz = 2.0
d_list = [dz] * N_slices
n_list = [1.0] + [n_au] * N_slices + [n_au]

out = tmm_full(n_list, d_list, wl, 0.0, 's')
R = out['R']
T = out['T']
A_j = out['A']

z_mid = dz * (np.arange(N_slices) + 0.5)
A_j_BL = alpha * dz * (1 - R) * np.exp(-alpha * z_mid)

A_cum_tmm = np.cumsum(A_j)
A_cum_BL = (1 - R) * (1 - np.exp(-alpha * dz * np.arange(1, N_slices + 1)))

print(f"alpha = {alpha:.4e} /nm, 1/alpha = {1/alpha:.2f} nm")
print(f"R = {R:.4f}, T = {T:.2e}, sum(A) = {np.sum(A_j):.6f}")
print(f"max |A_TMM - A_BL| / A_TMM near surface (z<20 nm) = "
      f"{np.max(np.abs(A_j[:10] - A_j_BL[:10]) / A_j[:10]):.2e}")
print(f"max |A_cum_TMM - A_cum_BL| = {np.max(np.abs(A_cum_tmm - A_cum_BL)):.2e}")

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.semilogy(z_mid, A_j, color=CB10[1], lw=1.0, marker='o', markersize=2,
             label='TMM $A_j$')
ax1.semilogy(z_mid, A_j_BL, color='k', ls='--', lw=0.7,
             label=r'$\alpha\,dz\,(1-R)\,e^{-\alpha z}$')
ax1.set_xlabel("Depth $z$ (nm)")
ax1.set_ylabel(r"Per-layer absorption $A_j$")
ax1.legend(loc='best', fontsize=6)
add_panel_label(ax1, 'a')

ax2.plot(z_mid, A_cum_tmm, color=CB10[1], lw=1.0, label='TMM cumulative')
ax2.plot(z_mid, A_cum_BL, color='k', ls='--', lw=0.7,
         label=r'$(1-R)(1-e^{-\alpha z})$')
ax2.axhline(1 - R, color=CB10[9], ls=':', lw=0.5, label=r'$1-R$')
ax2.set_xlabel("Depth $z$ (nm)")
ax2.set_ylabel(r"$\sum_{j <= k} A_j$")
ax2.legend(loc='best', fontsize=6)
add_panel_label(ax2, 'b')

out_png = Path(__file__).parent / 'figures' / '08_beer_lambert.pdf'
finalize(fig, out_png)
