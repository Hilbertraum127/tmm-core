"""
08 - 2D reflectance map R(theta, wavelength) for a narrow bandpass filter.

Three-cavity Fabry-Perot on glass at design wavelength 550 nm. Maps s-
and p-polarised reflectance; a dashed black curve shows the analytic
angular blueshift of the passband,
    lambda(theta) = lambda0 * sqrt(1 - sin^2(theta) / n_eff^2),
with the effective cavity index n_eff taken as the geometric mean of
the high and low indices.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R
from plotter import apply_style, fig_collage, add_panel_label, finalize

apply_style()

n_inc, n_sub = 1.0, 1.52
n_H, n_L = 2.35, 1.38
wl0 = 550.0

N_HR = 4
n_stack = ([n_inc]
           + [n_H, n_L] * N_HR
           + [n_H]
           + [n_L, n_H] * N_HR
           + [n_sub])
d_stack = ([wl0 / (4 * n_H), wl0 / (4 * n_L)] * N_HR
           + [wl0 / (2 * n_H)]
           + [wl0 / (4 * n_L), wl0 / (4 * n_H)] * N_HR)

wl_scan = np.linspace(500.0, 600.0, 201)
theta_scan = np.linspace(0.0, 40.0, 41)

R_s = np.zeros((len(theta_scan), len(wl_scan)))
R_p = np.zeros_like(R_s)
for i, th in enumerate(theta_scan):
    for j, w in enumerate(wl_scan):
        R_s[i, j] = tmm_R(n_stack, d_stack, w, th, 's')
        R_p[i, j] = tmm_R(n_stack, d_stack, w, th, 'p')

# Analytic passband shift: n_eff ~ sqrt(n_H * n_L) (first-order estimate)
n_eff = np.sqrt(n_H * n_L)
theta_curve = np.linspace(0.0, 40.0, 201)
lam_shift = wl0 * np.sqrt(1 - np.sin(np.radians(theta_curve)) ** 2 / n_eff ** 2)

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
for ax, data, lab, panel in ((axes[0][0], R_s, 's', 'a'),
                             (axes[0][1], R_p, 'p', 'b')):
    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=[wl_scan[0], wl_scan[-1], theta_scan[0], theta_scan[-1]],
                   vmin=0, vmax=1, cmap='viridis')
    ax.plot(lam_shift, theta_curve, color='r', ls='--', lw=0.7,
            label=r'$\lambda_0\sqrt{1-\sin^2\theta/n_\mathrm{eff}^2}$')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Angle of incidence (deg)")
    ax.set_xlim(wl_scan[0], wl_scan[-1])
    ax.set_ylim(theta_scan[0], theta_scan[-1])
    ax.legend(loc='upper right', fontsize=6)
    add_panel_label(ax, panel)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(f"$R_{lab}$", rotation=0, labelpad=10)

out = Path(__file__).parent / 'figures' / '11_angle_wavelength_map.pdf'
finalize(fig, out)
