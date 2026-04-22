"""
05 - Metal reflectance: Ag and Au, with closed-form Fresnel cross-check.

Reflectance of a semi-infinite silver or gold surface over 400-1000 nm
using hard-coded Johnson-Christy (Phys. Rev. B 6, 4370, 1972) refractive
index data. Values are linearly interpolated on the requested grid.

For a bare interface the normal-incidence reflectance follows directly
from the Fresnel equations,
    R = ((n - 1)^2 + k^2) / ((n + 1)^2 + k^2),
which must agree with tmm_R([1, n+ik], [], wl, 0, 's') at machine precision.
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R
from plotter import apply_style, fig_single, CB10, finalize

apply_style()

# Johnson-Christy 1972, sampled in the visible/NIR. Columns: wl(nm), n, k.
AG = np.array([
    [ 400, 0.173, 1.95 ],
    [ 450, 0.173, 2.56 ],
    [ 500, 0.129, 3.06 ],
    [ 550, 0.120, 3.59 ],
    [ 600, 0.131, 4.09 ],
    [ 650, 0.140, 4.52 ],
    [ 700, 0.149, 4.90 ],
    [ 800, 0.171, 5.67 ],
    [ 900, 0.188, 6.35 ],
    [1000, 0.211, 7.05 ],
])
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


def n_of(wl, table):
    n_interp = np.interp(wl, table[:, 0], table[:, 1])
    k_interp = np.interp(wl, table[:, 0], table[:, 2])
    return n_interp + 1j * k_interp


def R_fresnel_normal(n, k):
    return ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2)


wl = np.linspace(400.0, 1000.0, 601)
N_ag = np.array([n_of(w, AG) for w in wl])
N_au = np.array([n_of(w, AU) for w in wl])

R_ag_tmm = np.array([tmm_R([1.0, n_k], [], w, 0.0, 's') for n_k, w in zip(N_ag, wl)])
R_au_tmm = np.array([tmm_R([1.0, n_k], [], w, 0.0, 's') for n_k, w in zip(N_au, wl)])
R_ag_an = R_fresnel_normal(N_ag.real, N_ag.imag)
R_au_an = R_fresnel_normal(N_au.real, N_au.imag)

print(f"max |R_TMM - R_Fresnel| Ag = {np.max(np.abs(R_ag_tmm - R_ag_an)):.2e}")
print(f"max |R_TMM - R_Fresnel| Au = {np.max(np.abs(R_au_tmm - R_au_an)):.2e}")

fig, ax = fig_single()
ax.plot(wl, R_ag_tmm, color=CB10[7], label='Ag (TMM)')
ax.plot(wl, R_au_tmm, color=CB10[5], label='Au (TMM)')
ax.plot(wl, R_ag_an, color=CB10[9], ls='--', lw=0.5, label='Fresnel analytic')
ax.plot(wl, R_au_an, color=CB10[9], ls='--', lw=0.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance (normal incidence)")
ax.set_ylim(0, 1)
ax.legend(loc='lower right')

out = Path(__file__).parent / 'figures' / '09_metal_reflectance.pdf'
finalize(fig, out)
