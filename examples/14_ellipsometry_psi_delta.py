"""
07 - Ellipsometric angles Psi, Delta for SiO2 / Si at 633 nm.

Panel a, b: Psi, Delta vs angle of incidence, comparing
  - bare Si (colored solid = TMM, dashed black = closed-form Fresnel),
  - 25 nm SiO2 / Si (colored solid = TMM).
Panel c: pretend we do not know the oxide thickness; fit d_SiO2 from the
Psi, Delta TMM data via least squares, recover the true 25 nm.
"""

from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from tmm_core import tmm_r
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

n_air = 1.0
n_sio2 = 1.457
n_si = 3.882 + 0.019j
d_true = 25.0
wl = 633.0
theta = np.linspace(0.1, 89.0, 451)


def psi_delta_from_r(r_s, r_p):
    rho = r_p / r_s
    psi = np.degrees(np.arctan(np.abs(rho)))
    delta = np.degrees(np.angle(rho)) % 360.0
    return psi, delta


def psi_delta_stack(n_list, d_list, wl, theta_deg):
    r_s = np.array([tmm_r(n_list, d_list, wl, t, 's') for t in theta_deg])
    r_p = np.array([tmm_r(n_list, d_list, wl, t, 'p') for t in theta_deg])
    return psi_delta_from_r(r_s, r_p)


def fresnel_rs_rp(n0, n1, theta_deg):
    th0 = np.radians(theta_deg)
    sin_t1 = n0 / n1 * np.sin(th0)
    cos_t1 = np.sqrt(1 - sin_t1 ** 2 + 0j)
    r_s = (n0 * np.cos(th0) - n1 * cos_t1) / (n0 * np.cos(th0) + n1 * cos_t1)
    r_p_convB = (n1 * np.cos(th0) - n0 * cos_t1) / (n1 * np.cos(th0) + n0 * cos_t1)
    # Convention A to match tmm_core (r_s = +r_p at normal)
    r_p_convA = -r_p_convB
    return r_s, r_p_convA


# TMM curves
psi_bare, delta_bare = psi_delta_stack([n_air, n_si], [], wl, theta)
psi_ox, delta_ox = psi_delta_stack([n_air, n_sio2, n_si], [d_true], wl, theta)

# Analytic Fresnel for bare Si
r_s_an, r_p_an = fresnel_rs_rp(n_air, n_si, theta)
psi_bare_an, delta_bare_an = psi_delta_from_r(r_s_an, r_p_an)

# Thickness fit: minimise chi^2 on (Psi, Delta) vs theta
theta_fit = np.linspace(50.0, 85.0, 71)
psi_data, delta_data = psi_delta_stack([n_air, n_sio2, n_si], [d_true], wl, theta_fit)


def chi2(d):
    psi_m, delta_m = psi_delta_stack([n_air, n_sio2, n_si], [float(d)], wl, theta_fit)
    return np.sum((psi_m - psi_data) ** 2 + (delta_m - delta_data) ** 2)


res = minimize_scalar(chi2, bracket=(5.0, 25.0, 60.0), method='brent',
                      options={'xtol': 1e-8})
d_fit = float(res.x)
print(f"true d_SiO2 = {d_true:.3f} nm")
print(f"fitted d_SiO2 = {d_fit:.6f} nm")
print(f"residual chi^2 = {res.fun:.2e}")

# Scan chi^2(d) for plotting
d_scan = np.linspace(10.0, 50.0, 201)
chi_scan = np.array([chi2(d) for d in d_scan])

fig, axes = fig_collage(ncols=3, nrows=1, height_cm=6.5)
ax_p, ax_d, ax_fit = axes[0]

ax_p.plot(theta, psi_bare, color=CB10[0], lw=1.0, label='bare Si (TMM)')
ax_p.plot(theta, psi_bare_an, color='k', ls='--', lw=0.7, label='Fresnel analytic')
ax_p.plot(theta, psi_ox, color=CB10[1], lw=1.0, label=f'{d_true:.0f} nm SiO$_2$ / Si')
ax_p.set_xlabel(r"Angle $\theta_0$ (deg)")
ax_p.set_ylabel(r"$\Psi$ (deg)")
ax_p.set_xlim(0, 90)
ax_p.legend(loc='best', fontsize=6)
add_panel_label(ax_p, 'a')

ax_d.plot(theta, delta_bare, color=CB10[0], lw=1.0, label='bare Si (TMM)')
ax_d.plot(theta, delta_bare_an, color='k', ls='--', lw=0.7, label='Fresnel analytic')
ax_d.plot(theta, delta_ox, color=CB10[1], lw=1.0, label=f'{d_true:.0f} nm SiO$_2$ / Si')
ax_d.set_xlabel(r"Angle $\theta_0$ (deg)")
ax_d.set_ylabel(r"$\Delta$ (deg)")
ax_d.set_xlim(0, 90)
ax_d.legend(loc='best', fontsize=6)
add_panel_label(ax_d, 'b')

ax_fit.semilogy(d_scan, chi_scan + 1e-300, color=CB10[1], lw=1.0)
ax_fit.axvline(d_fit, color='k', ls='--', lw=0.7, label=f'fit = {d_fit:.2f} nm')
ax_fit.axvline(d_true, color=CB10[9], ls=':', lw=0.5, label=f'true = {d_true:.0f} nm')
ax_fit.set_xlabel(r"$d_\mathrm{SiO_2}$ (nm)")
ax_fit.set_ylabel(r"$\chi^2(d)$")
ax_fit.legend(loc='best', fontsize=6)
add_panel_label(ax_fit, 'c')

out = Path(__file__).parent / 'figures' / '14_ellipsometry_psi_delta.pdf'
finalize(fig, out)
