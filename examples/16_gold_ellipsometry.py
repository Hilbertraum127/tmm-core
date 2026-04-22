"""
16 - Ellipsometry on gold: pseudo-dielectric inversion over wavelength.

Bare gold substrate (Johnson-Christy 1972) at theta = 70 deg. TMM returns
r_s and r_p via tmm_r with d_list=[]; the pseudo-dielectric inversion

    <eps> = sin^2(theta) * [ 1 + tan^2(theta) * ((1-rho)/(1+rho))^2 ],
    rho = r_p / r_s  (Convention A -> feed -rho for the textbook formula)

should reproduce the input N^2 = (n + i k)^2 exactly, because Au here is
a bare substrate (no overlayer).

Panel a: n(lambda), k(lambda) from the JC table (dashed black) vs
         n, k recovered by sqrt(<eps>) from the TMM inversion (solid).
Panel b: Re(eps), Im(eps) from the JC table (dashed black) vs
         Re(<eps>), Im(<eps>) from the TMM inversion (solid).
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_r
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

# Johnson-Christy 1972 gold, wl (nm), n, k.
AU = np.array([
    [ 400, 1.658, 1.956],
    [ 450, 1.494, 1.898],
    [ 500, 0.922, 1.900],
    [ 550, 0.425, 2.367],
    [ 600, 0.234, 2.970],
    [ 650, 0.174, 3.455],
    [ 700, 0.161, 3.866],
    [ 800, 0.165, 4.720],
    [ 900, 0.190, 5.490],
    [1000, 0.239, 6.190],
])


def n_of(wl):
    return (np.interp(wl, AU[:, 0], AU[:, 1])
            + 1j * np.interp(wl, AU[:, 0], AU[:, 2]))


def pseudo_eps(rho_convA, theta_deg):
    theta = np.radians(theta_deg)
    rho_B = -rho_convA
    return np.sin(theta) ** 2 * (1 + np.tan(theta) ** 2 *
                                  ((1 - rho_B) / (1 + rho_B)) ** 2)


theta_deg = 70.0
wl = np.linspace(400.0, 1000.0, 601)
N_in = np.array([n_of(w) for w in wl])
eps_in = N_in ** 2

r_s = np.array([tmm_r([1.0, N_in[i]], [], w, theta_deg, 's')
                for i, w in enumerate(wl)])
r_p = np.array([tmm_r([1.0, N_in[i]], [], w, theta_deg, 'p')
                for i, w in enumerate(wl)])
rho = r_p / r_s
eps_extract = pseudo_eps(rho, theta_deg)
N_extract = np.sqrt(eps_extract)
# Enforce passive branch (Im N >= 0) so the visual matches the input convention.
N_extract = np.where(N_extract.imag < 0, -N_extract, N_extract)

err_eps = np.max(np.abs(eps_extract - eps_in))
err_N = np.max(np.abs(N_extract - N_in))
print(f"max |eps_extracted - N_in^2| = {err_eps:.2e}")
print(f"max |N_extracted - N_in|     = {err_N:.2e}")

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

ax1.plot(wl, N_extract.real, color=CB10[0], lw=1.0, label='$n$ (TMM)')
ax1.plot(wl, N_extract.imag, color=CB10[1], lw=1.0, label='$k$ (TMM)')
ax1.plot(wl, N_in.real, color='k', ls='--', lw=0.7, label='JC $n$, $k$')
ax1.plot(wl, N_in.imag, color='k', ls='--', lw=0.7)
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel(r"$n$, $k$ for Au")
ax1.legend(loc='best', fontsize=6)
ax1.text(0.03, 0.97, r"$\theta = 70^\circ$", transform=ax1.transAxes,
         fontsize=6, ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
add_panel_label(ax1, 'a')

ax2.plot(wl, eps_extract.real, color=CB10[0], lw=1.0, label=r'$\mathrm{Re}\langle\varepsilon\rangle$ (TMM)')
ax2.plot(wl, eps_extract.imag, color=CB10[1], lw=1.0, label=r'$\mathrm{Im}\langle\varepsilon\rangle$ (TMM)')
ax2.plot(wl, eps_in.real, color='k', ls='--', lw=0.7, label=r'JC $N^2$')
ax2.plot(wl, eps_in.imag, color='k', ls='--', lw=0.7)
ax2.set_xlabel("Wavelength (nm)")
ax2.set_ylabel(r"$\langle\varepsilon\rangle$ for Au")
ax2.legend(loc='best', fontsize=6)
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '16_gold_ellipsometry.pdf'
finalize(fig, out)
