"""
03 - Single-layer antireflection coating.

MgF2 (n = 1.38) on glass (n = 1.52) at 550 nm. The analytic quarter-wave
condition gives optimum thickness d_opt = lambda0 / (4 n_f) and residual
reflectance R_min = ((n0 n_s - n_f^2) / (n0 n_s + n_f^2))^2. Both are
drawn as reference lines; the TMM curve touches them exactly.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

n0, n_f, n_sub = 1.0, 1.38, 1.52
wl0 = 550.0

d_opt = wl0 / (4.0 * n_f)
R_min = ((n0 * n_sub - n_f ** 2) / (n0 * n_sub + n_f ** 2)) ** 2
R_bare = ((n0 - n_sub) / (n0 + n_sub)) ** 2

d_scan = np.linspace(0.0, 300.0, 601)
R_vs_d = np.array([tmm_R([n0, n_f, n_sub], [d], wl0, 0.0, 's') for d in d_scan])

wl_scan = np.linspace(400.0, 700.0, 301)
R_vs_wl = np.array([tmm_R([n0, n_f, n_sub], [d_opt], w, 0.0, 's') for w in wl_scan])
R_bare_wl = np.array([tmm_R([n0, n_sub], [], w, 0.0, 's') for w in wl_scan])

R_tmm_at_opt = tmm_R([n0, n_f, n_sub], [d_opt], wl0, 0.0, 's')
print(f"d_opt (analytic)       = {d_opt:.3f} nm")
print(f"R_min (analytic)       = {R_min:.3e}")
print(f"R_TMM(d_opt, lambda0)  = {R_tmm_at_opt:.3e}")
print(f"|R_TMM - R_min|        = {abs(R_tmm_at_opt - R_min):.2e}")

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.plot(d_scan, R_vs_d, color=CB10[0], label='TMM')
ax1.axvline(d_opt, color=CB10[9], ls=':', lw=0.5, label=r'$d_\mathrm{opt}=\lambda_0/4n_f$')
ax1.axhline(R_min, color=CB10[9], ls='--', lw=0.5, label=r'$R_\mathrm{min}$ (analytic)')
ax1.set_xlabel(r"Film thickness $d$ (nm)")
ax1.set_ylabel("Reflectance at 550 nm")
ax1.set_xlim(0, 300)
ax1.legend(loc='upper right')
add_panel_label(ax1, 'a')

ax2.plot(wl_scan, R_bare_wl, color=CB10[9], lw=0.8, alpha=0.5, label='no coating')
ax2.plot(wl_scan, R_vs_wl, color=CB10[1], label=f'MgF$_2$ {d_opt:.1f} nm')
ax2.axvline(wl0, color=CB10[9], ls=':', lw=0.5)
ax2.set_xlabel(r"Wavelength (nm)")
ax2.set_ylabel("Reflectance")
ax2.legend(loc='upper right')
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '06_antireflection_coating.pdf'
finalize(fig, out)
