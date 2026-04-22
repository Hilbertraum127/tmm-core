"""
10 - Total internal reflection and frustrated TIR.

Glass (n = 1.5) -> air (n = 1) at 550 nm. Above theta_c = arcsin(1/1.5)
= 41.81 deg an isolated interface reflects 100% and imparts an
angle-dependent phase. With an air gap between two glass half-spaces the
evanescent wave tunnels; the full three-layer Airy formula for T(d) is
exact for any gap thickness and is a much tighter reference than the
pure exponential asymptote.

Panels:
  a) log |1 - R_s|, log |1 - R_p| vs theta above theta_c (both -> 0
     at machine precision).
  b) Phase of r_s vs theta compared to Lekner's closed form
       tan(phi_s/2) = sqrt(sin^2 theta - n_r^2) / cos(theta),   n_r = n2/n1.
  c) FTIR T(d) at theta = 60 deg: TMM vs exact 3-layer Airy
     (glass / air-gap / glass) evaluated in the evanescent branch
     q_air = +i kappa.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_r, tmm_full
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

n1, n2 = 1.5, 1.0
wl = 550.0
k0 = 2 * np.pi / wl
theta_c = np.degrees(np.arcsin(n2 / n1))
theta = np.linspace(theta_c + 0.01, 89.0, 401)

r_s = np.array([tmm_r([n1, n2], [], wl, t, 's') for t in theta])
r_p = np.array([tmm_r([n1, n2], [], wl, t, 'p') for t in theta])

R_s_dev = np.abs(1.0 - np.abs(r_s) ** 2)
R_p_dev = np.abs(1.0 - np.abs(r_p) ** 2)

nr = n2 / n1
phi_s_an = 2 * np.arctan2(np.sqrt(np.sin(np.radians(theta)) ** 2 - nr ** 2),
                          np.cos(np.radians(theta)))
phi_s_tm = np.unwrap(np.angle(r_s))

# FTIR at 60 deg: glass | air gap | glass, exact 3-layer Airy with the
# evanescent branch Im(q) >= 0.
theta_ftir = 60.0
beta = k0 * n1 * np.sin(np.radians(theta_ftir))
q_g = np.sqrt((k0 * n1) ** 2 - beta ** 2 + 0j)
q_a = np.sqrt((k0 * n2) ** 2 - beta ** 2 + 0j)
if q_a.imag < 0 or (q_a.imag == 0 and q_a.real < 0):
    q_a = -q_a
r12 = (q_g - q_a) / (q_g + q_a)       # glass -> air (s-pol)
r21 = -r12                            # air -> glass
t12 = 2 * q_g / (q_g + q_a)
t21 = 2 * q_a / (q_g + q_a)

d_gap = np.linspace(10.0, 600.0, 401)
phase = np.exp(1j * q_a * d_gap)      # single pass through the gap
# Airy reflection and transmission (amplitude) for symmetric 3-layer stack
r_stack = (r12 + r21 * phase ** 2) / (1 + r12 * r21 * phase ** 2)
t_stack = (t12 * t21 * phase) / (1 + r12 * r21 * phase ** 2)
T_airy = (q_g.real / q_g.real) * np.abs(t_stack) ** 2   # lossless outer media
R_airy = np.abs(r_stack) ** 2

T_tmm = np.array([tmm_full([n1, n2, n1], [d], wl, theta_ftir, 's')['T']
                  for d in d_gap])

print(f"theta_c = {theta_c:.3f} deg")
print(f"max |1 - R| above theta_c: s = {np.max(R_s_dev):.2e}, p = {np.max(R_p_dev):.2e}")
print(f"FTIR max |T_TMM - T_Airy| = {np.max(np.abs(T_tmm - T_airy)):.2e}")

fig, axes = fig_collage(ncols=3, nrows=1, height_cm=6.5)
ax_R, ax_ph, ax_T = axes[0]

ax_R.semilogy(theta, R_s_dev + 1e-300, color=CB10[0], lw=1.0, label='s')
ax_R.semilogy(theta, R_p_dev + 1e-300, color=CB10[1], lw=1.0, label='p')
ax_R.set_xlabel(r"$\theta$ (deg)")
ax_R.set_ylabel(r"$|1 - R|$")
ax_R.legend(loc='best')
add_panel_label(ax_R, 'a')

ax_ph.plot(theta, np.degrees(phi_s_tm), color=CB10[0], lw=1.0, label='TMM')
ax_ph.plot(theta, np.degrees(-phi_s_an), color='k', ls='--', lw=0.7, label='analytic')
ax_ph.set_xlabel(r"$\theta$ (deg)")
ax_ph.set_ylabel(r"$\arg(r_s)$ (deg)")
ax_ph.legend(loc='best')
add_panel_label(ax_ph, 'b')

ax_T.semilogy(d_gap, T_tmm, color=CB10[0], lw=1.0, label='TMM')
ax_T.semilogy(d_gap, T_airy, color='k', ls='--', lw=0.7, label='3-layer Airy')
ax_T.set_xlabel("Gap thickness (nm)")
ax_T.set_ylabel(r"$T$")
ax_T.legend(loc='best')
add_panel_label(ax_T, 'c')

out = Path(__file__).parent / 'figures' / '03_tir_and_ftir.pdf'
finalize(fig, out)
