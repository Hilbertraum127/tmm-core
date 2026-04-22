"""
06 - Absorbing thin film: energy conservation and Beer-Lambert limit.

a-Si-like film (n = 4 + 0.5i) on glass at 550 nm.

Panel a: R, T, A vs film thickness. The three components oscillate
through Airy fringes while their sum stays at 1 to machine precision
(the residual max |1 - (R+T+A)| is printed; it is below 1e-14).

Panel b: T(d) extended to 2000 nm, compared with the Beer-Lambert
asymptote T_if * exp(-alpha d) with alpha = 2 k0 Im(n_f) and
T_if = (1 - R_front)(1 - R_back). The interference ripple dies out and
the TMM curve follows the exponential envelope for d >> 1/alpha.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_full
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

n0, n_f, n_sub = 1.0, 4.0 + 0.5j, 1.52
wl = 550.0
k0 = 2 * np.pi / wl
alpha = 2 * k0 * n_f.imag
R_front = np.abs((n0 - n_f) / (n0 + n_f)) ** 2
R_back = np.abs((n_f - n_sub) / (n_f + n_sub)) ** 2
T_if = (1 - R_front) * (1 - R_back)

d_scan = np.linspace(0.0, 400.0, 801)
R, T, A = [], [], []
for d in d_scan:
    out = tmm_full([n0, n_f, n_sub], [d], wl, 0.0, 's')
    R.append(out['R']); T.append(out['T']); A.append(out['A'][0])
R = np.array(R); T = np.array(T); A = np.array(A)

sum_res = np.max(np.abs(1.0 - (R + T + A)))
print(f"max |1 - (R+T+A)| = {sum_res:.2e}")
print(f"alpha = {alpha:.4e} /nm, absorption length 1/alpha = {1/alpha:.2f} nm")
print(f"T_interface = {T_if:.4f}")

d_thick = np.linspace(1.0, 2000.0, 1001)
T_thick = np.array([tmm_full([n0, n_f, n_sub], [d], wl, 0.0, 's')['T'] for d in d_thick])
T_BL = T_if * np.exp(-alpha * d_thick)

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.plot(d_scan, R, color=CB10[0], lw=1.0, label='$R$')
ax1.plot(d_scan, T, color=CB10[1], lw=1.0, label='$T$')
ax1.plot(d_scan, A, color=CB10[2], lw=1.0, label='$A$')
ax1.set_xlabel(r"Film thickness $d$ (nm)")
ax1.set_ylabel("Power fraction")
ax1.set_xlim(0, 400)
ax1.set_ylim(0, 1.02)
ax1.text(0.97, 0.97, f"$n_f = {n_f}$\n$|1-(R+T+A)| < {sum_res:.0e}$",
         transform=ax1.transAxes, fontsize=6, ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
ax1.legend(loc='center right')
add_panel_label(ax1, 'a')

ax2.semilogy(d_thick, T_thick, color=CB10[1], lw=1.0, label='TMM $T$')
ax2.semilogy(d_thick, T_BL, color='k', ls='--', lw=0.7,
             label=r'$T_\mathrm{if}\,e^{-\alpha d}$')
ax2.set_xlabel(r"Film thickness $d$ (nm)")
ax2.set_ylabel("Transmittance")
ax2.set_xlim(0, 2000)
ax2.legend(loc='upper right')
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '07_absorbing_layer.pdf'
finalize(fig, out)
