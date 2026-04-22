"""
01 - Fresnel single interface.

R_s, R_p vs angle for air/glass (n = 1.0 / 1.5) at 550 nm. The closed-form
Fresnel reflectances are overlaid as dashed black curves and coincide with
the TMM result to machine precision. The Brewster angle
theta_B = arctan(n_glass / n_air) is marked; |r_p(theta_B)| is printed as
a numerical check.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R, tmm_r
from plotter import apply_style, fig_single, CB10, finalize

apply_style()

n_air, n_glass = 1.0, 1.5
wl = 550.0
theta = np.linspace(0.0, 89.9, 901)


def fresnel_R(n0, n1, theta_deg, pol):
    th0 = np.radians(theta_deg)
    sin_t1 = n0 / n1 * np.sin(th0)
    cos_t1 = np.sqrt(1 - sin_t1 ** 2 + 0j)
    if pol == 's':
        r = (n0 * np.cos(th0) - n1 * cos_t1) / (n0 * np.cos(th0) + n1 * cos_t1)
    else:
        r = (n1 * np.cos(th0) - n0 * cos_t1) / (n1 * np.cos(th0) + n0 * cos_t1)
    return np.abs(r) ** 2


R_s = np.array([tmm_R([n_air, n_glass], [], wl, t, 's') for t in theta])
R_p = np.array([tmm_R([n_air, n_glass], [], wl, t, 'p') for t in theta])
R_s_an = fresnel_R(n_air, n_glass, theta, 's')
R_p_an = fresnel_R(n_air, n_glass, theta, 'p')

theta_B = np.degrees(np.arctan(n_glass / n_air))
r_p_at_B = tmm_r([n_air, n_glass], [], wl, theta_B, 'p')
print(f"Brewster angle (analytic) theta_B = {theta_B:.6f} deg")
print(f"|r_p(theta_B)| = {abs(r_p_at_B):.2e}  (expected: ~machine precision)")
print(f"max |R_s_TMM - R_s_an| = {np.max(np.abs(R_s - R_s_an)):.2e}")
print(f"max |R_p_TMM - R_p_an| = {np.max(np.abs(R_p - R_p_an)):.2e}")

fig, ax = fig_single()
ax.plot(theta, R_s, color=CB10[0], lw=1.0, label='$R_s$ TMM')
ax.plot(theta, R_p, color=CB10[1], lw=1.0, label='$R_p$ TMM')
ax.plot(theta, R_s_an, color='k', ls='--', lw=0.7, label='analytic')
ax.plot(theta, R_p_an, color='k', ls='--', lw=0.7)
ax.axvline(theta_B, color=CB10[9], ls=':', lw=0.5)
ax.text(theta_B, 0.5, f"  Brewster {theta_B:.2f}" + r"$^\circ$", fontsize=7)
ax.set_xlabel(r"Angle of incidence $\theta_0$ (deg)")
ax.set_ylabel("Reflectance")
ax.set_xlim(0, 90)
ax.set_ylim(0, 1)
ax.legend(loc='upper left')

out = Path(__file__).parent / 'figures' / '01_fresnel_single_interface.pdf'
finalize(fig, out)
