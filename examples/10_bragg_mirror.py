"""
04 - Quarter-wave distributed Bragg reflector.

TiO2 (n_H = 2.35) / SiO2 (n_L = 1.38) pairs on glass, centered at 550 nm.
Two panels:
(a) R(lambda) for N = 2, 5, 10 pairs, with the analytic stopband width
    Delta_lambda / lambda0 = (4/pi) arcsin((n_H - n_L)/(n_H + n_L))
    drawn as a shaded band around lambda0.
(b) R_peak vs N pairs, TMM vs analytic DBR formula
    R_peak = ((n_inc - Y_eff) / (n_inc + Y_eff))^2,
    Y_eff = (n_H/n_L)^(2N) * n_sub.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

n_inc, n_sub = 1.0, 1.52
n_H, n_L = 2.35, 1.38
wl0 = 550.0
wl_scan = np.linspace(400.0, 800.0, 401)


def dbr(N):
    n = [n_inc] + [n_H, n_L] * N + [n_sub]
    d = [wl0 / (4 * n_H), wl0 / (4 * n_L)] * N
    return n, d


band = (4.0 / np.pi) * np.arcsin((n_H - n_L) / (n_H + n_L))
lam_lo = wl0 / (1 + band / 2)
lam_hi = wl0 / (1 - band / 2)
print(f"analytic stopband: {lam_lo:.1f} - {lam_hi:.1f} nm (Delta/lambda0 = {band:.3f})")

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.axvspan(lam_lo, lam_hi, color=CB10[9], alpha=0.12, label='analytic stopband')
for k, N in enumerate([2, 5, 10]):
    n, d = dbr(N)
    R = np.array([tmm_R(n, d, w, 0.0, 's') for w in wl_scan])
    ax1.plot(wl_scan, R, color=CB10[k], lw=0.8, label=f"{N} pairs")
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Reflectance")
ax1.set_xlim(400, 800)
ax1.set_ylim(0, 1.02)
ax1.legend(loc='lower right', fontsize=6)
add_panel_label(ax1, 'a')

N_range = np.arange(1, 16)
R_peak_tmm, R_peak_an = [], []
for N in N_range:
    n, d = dbr(N)
    R_peak_tmm.append(tmm_R(n, d, wl0, 0.0, 's'))
    Y = (n_H / n_L) ** (2 * N) * n_sub
    R_peak_an.append(((n_inc - Y) / (n_inc + Y)) ** 2)
R_peak_tmm = np.array(R_peak_tmm)
R_peak_an = np.array(R_peak_an)
print(f"max |R_peak_TMM - R_peak_analytic| = {np.max(np.abs(R_peak_tmm - R_peak_an)):.2e}")

ax2.plot(N_range, R_peak_tmm, color=CB10[1], marker='o',
         markersize=3, lw=1.0, label='TMM')
ax2.plot(N_range, R_peak_an, color='k', ls='--', lw=0.7, label='analytic')
ax2.set_xlabel("Number of HL pairs $N$")
ax2.set_ylabel(r"$R_\mathrm{peak}$ at 550 nm")
ax2.set_ylim(0, 1.02)
ax2.legend(loc='lower right')
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '10_bragg_mirror.pdf'
finalize(fig, out)
