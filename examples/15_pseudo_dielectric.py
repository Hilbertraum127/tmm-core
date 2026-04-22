"""
15 - Pseudo-dielectric function from ellipsometry.

Ellipsometry measures the complex ratio rho = r_p / r_s at angle theta.
For a bulk (substrate-only) sample the inversion

    <eps> = sin^2(theta) * [ 1 + tan^2(theta) * ((1-rho)/(1+rho))^2 ]

returns exactly the substrate permittivity N^2. For a film-covered
substrate, <eps> is an 'apparent' dielectric function that includes the
film's interference and is *not* equal to either the film or the
substrate permittivity; the systematic deviation from the bulk curve
quantifies the film's contribution.

The formula assumes the Wikipedia / Azzam-Bashara 'Convention B' sign
of r_p. tmm_core uses Convention A (r_s = +r_p at normal incidence),
so we negate the measured rho before inverting.

Panels:
  (a) Bare substrate (toy c-Si dispersion): extracted <eps> (solid
      coloured = TMM inversion) matches the input N^2 (dashed black =
      model) to machine precision.
  (b) 2 nm SiO2 / Si: extracted <eps> (solid coloured) shows a visible
      bias relative to the bulk N^2 (dashed black). This is the classic
      reason ellipsometry requires a model, not a direct inversion.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_r
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()


def pseudo_eps(rho_convA, theta_deg):
    theta = np.radians(theta_deg)
    rho_B = -rho_convA
    return np.sin(theta) ** 2 * (1 + np.tan(theta) ** 2 *
                                  ((1 - rho_B) / (1 + rho_B)) ** 2)


def rho_of_stack(n_list, d_list, wl, theta_deg):
    r_p = tmm_r(n_list, d_list, wl, theta_deg, 'p')
    r_s = tmm_r(n_list, d_list, wl, theta_deg, 's')
    return r_p / r_s


# Toy c-Si dispersion with absorption peak near 370 nm
wl = np.linspace(300.0, 800.0, 251)
n_real = 3.5 + 0.5 * np.exp(-((wl - 370) / 40) ** 2)
k_img = 1.5 * np.exp(-((wl - 370) / 60) ** 2)
N_Si = n_real + 1j * k_img
eps_Si = N_Si ** 2

theta_deg = 70.0

rho_bare = np.array([rho_of_stack([1.0, N_Si[i]], [], w, theta_deg)
                     for i, w in enumerate(wl)])
eps_bare = pseudo_eps(rho_bare, theta_deg)
res_bare = np.abs(eps_bare - eps_Si)
print(f"(a) max |eps_extracted - N^2| bare substrate = {res_bare.max():.2e}")

n_SiO2 = 1.457
d_ox = 2.0
rho_ox = np.array([rho_of_stack([1.0, n_SiO2, N_Si[i]], [d_ox], w, theta_deg)
                   for i, w in enumerate(wl)])
eps_ox = pseudo_eps(rho_ox, theta_deg)
print(f"(b) 2 nm SiO2 deviation: max|Re(<eps>-N^2)| = {np.max(np.abs(eps_ox.real - eps_Si.real)):.3f},"
      f" max|Im| = {np.max(np.abs(eps_ox.imag - eps_Si.imag)):.3f}")

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.plot(wl, eps_bare.real, color=CB10[0], lw=1.0, label=r'TMM $\mathrm{Re}\langle\varepsilon\rangle$')
ax1.plot(wl, eps_bare.imag, color=CB10[1], lw=1.0, label=r'TMM $\mathrm{Im}\langle\varepsilon\rangle$')
ax1.plot(wl, eps_Si.real, color='k', ls='--', lw=0.7, label=r'$N^2$ model')
ax1.plot(wl, eps_Si.imag, color='k', ls=':', lw=0.7)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel(r"$\langle\varepsilon\rangle$")
ax1.text(0.03, 0.97, "bare substrate", transform=ax1.transAxes,
         fontsize=6, ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
ax1.legend(loc='upper right', fontsize=6)
add_panel_label(ax1, 'a')

ax2.plot(wl, eps_ox.real, color=CB10[0], lw=1.0, label=r'TMM $\mathrm{Re}\langle\varepsilon\rangle$')
ax2.plot(wl, eps_ox.imag, color=CB10[1], lw=1.0, label=r'TMM $\mathrm{Im}\langle\varepsilon\rangle$')
ax2.plot(wl, eps_Si.real, color='k', ls='--', lw=0.7, label=r'bulk $N^2$')
ax2.plot(wl, eps_Si.imag, color='k', ls=':', lw=0.7)
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel(r"$\langle\varepsilon\rangle$")
ax2.text(0.03, 0.97, f"{d_ox:g} nm SiO$_2$ / Si", transform=ax2.transAxes,
         fontsize=6, ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
ax2.legend(loc='upper right', fontsize=6)
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '15_pseudo_dielectric.pdf'
finalize(fig, out)
