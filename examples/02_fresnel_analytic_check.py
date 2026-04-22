"""
02 - Fresnel single interface: TMM vs closed-form Fresnel, two materials.

Two single-interface cases at 550 nm, each shown as reflectance R(theta)
and phase arg(r)(theta) for s- and p-polarisation.

  Column 0: air (n0 = 1) / glass (n1 = 1.5). Real dielectric, Brewster
            zero at arctan(1.5) = 56.31 deg in r_p.
  Column 1: air / absorbing Si-like (n1 = 4 + 0.05j). No true Brewster
            zero, just a shallow minimum and a non-trivial phase walk.

TMM uses Convention A (r_s = +r_p at normal). The Wikipedia-style
Fresnel formulas are written in Convention B, so we negate r_p_B to
compare amplitudes and phases consistently.

Panels:
  a) R_s, R_p for air/glass; TMM solid coloured, analytic dashed black.
  b) R_s, R_p for air/Si.
  c) arg(r_s), arg(r_p) for air/glass.
  d) arg(r_s), arg(r_p) for air/Si.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_r
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()


def fresnel_rs_rp(n0, n1, theta_deg):
    """Wikipedia/Convention-B Fresnel amplitude coefficients. Complex n1 ok."""
    th0 = np.radians(theta_deg)
    cos0 = np.cos(th0)
    sin1_sq = (n0 / n1) ** 2 * np.sin(th0) ** 2
    cos1 = np.sqrt(1.0 - sin1_sq + 0j)
    if np.imag(cos1) < 0:
        cos1 = -cos1
    r_s = (n0 * cos0 - n1 * cos1) / (n0 * cos0 + n1 * cos1)
    r_p_B = (n1 * cos0 - n0 * cos1) / (n1 * cos0 + n0 * cos1)
    r_p_A = -r_p_B  # flip sign for r_s = r_p convention used by tmm_core
    return r_s, r_p_A


wl = 550.0
theta = np.linspace(0.0, 89.5, 361)

cases = [
    ("air / glass",      1.0, 1.5 + 0j),
    ("air / Si-like",    1.0, 4.0 + 0.05j),
]

fig, axes = fig_collage(ncols=2, nrows=2, height_cm=12.0)

for col, (label, n0, n1) in enumerate(cases):
    r_s_tmm = np.array([tmm_r([n0, n1], [], wl, t, 's') for t in theta])
    r_p_tmm = np.array([tmm_r([n0, n1], [], wl, t, 'p') for t in theta])
    r_s_an = np.zeros_like(theta, dtype=complex)
    r_p_an = np.zeros_like(theta, dtype=complex)
    for i, t in enumerate(theta):
        rs, rp = fresnel_rs_rp(n0, n1, t)
        r_s_an[i] = rs
        r_p_an[i] = rp

    R_s_tmm = np.abs(r_s_tmm) ** 2
    R_p_tmm = np.abs(r_p_tmm) ** 2
    R_s_an = np.abs(r_s_an) ** 2
    R_p_an = np.abs(r_p_an) ** 2
    # Phase in [0, 360) so that the sign of the imaginary "zero" cannot
    # flip +pi vs -pi between TMM and analytic series.
    def _phase_deg(z):
        return np.degrees(np.angle(z)) % 360.0

    ph_s_tmm = _phase_deg(r_s_tmm)
    ph_p_tmm = _phase_deg(r_p_tmm)
    ph_s_an = _phase_deg(r_s_an)
    ph_p_an = _phase_deg(r_p_an)

    err_R = max(np.max(np.abs(R_s_tmm - R_s_an)),
                np.max(np.abs(R_p_tmm - R_p_an)))

    def ang_diff(a, b):
        return np.abs((a - b + 180.0) % 360.0 - 180.0)

    err_ph = max(np.max(ang_diff(ph_s_tmm, ph_s_an)),
                 np.max(ang_diff(ph_p_tmm, ph_p_an)))
    print(f"{label}: max|R_TMM - R_an| = {err_R:.2e},"
          f" max|phase TMM - an| = {err_ph:.2e} deg")

    ax_R = axes[0][col]
    ax_R.plot(theta, R_s_tmm, color=CB10[0], lw=1.0, label='s (TMM)')
    ax_R.plot(theta, R_p_tmm, color=CB10[1], lw=1.0, label='p (TMM)')
    ax_R.plot(theta, R_s_an, color='k', ls='--', lw=0.7, label='Fresnel')
    ax_R.plot(theta, R_p_an, color='k', ls='--', lw=0.7)
    ax_R.set_xlabel(r"$\theta$ (deg)")
    ax_R.set_ylabel("Reflectance")
    ax_R.text(0.03, 0.97, label, transform=ax_R.transAxes,
              fontsize=6, ha='left', va='top',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
    ax_R.legend(loc='best', fontsize=6)
    add_panel_label(ax_R, 'a' if col == 0 else 'b')

    ax_ph = axes[1][col]
    ax_ph.plot(theta, ph_s_tmm, color=CB10[0], lw=1.0, label='s (TMM)')
    ax_ph.plot(theta, ph_p_tmm, color=CB10[1], lw=1.0, label='p (TMM)')
    ax_ph.plot(theta, ph_s_an, color='k', ls='--', lw=0.7, label='Fresnel')
    ax_ph.plot(theta, ph_p_an, color='k', ls='--', lw=0.7)
    ax_ph.set_xlabel(r"$\theta$ (deg)")
    ax_ph.set_ylabel(r"$\arg(r)$ (deg)")
    ax_ph.legend(loc='best', fontsize=6)
    add_panel_label(ax_ph, 'c' if col == 0 else 'd')

out = Path(__file__).parent / 'figures' / '02_fresnel_analytic_check.pdf'
finalize(fig, out)
