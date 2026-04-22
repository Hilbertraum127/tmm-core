"""
02 - Thin film vs. analytic Airy (three cases).

Merges the former 02 (lossless) and 09 (absorbing). The analytic Airy
formula uses the exp(-i*omega*t) / Yeh phase convention exp(+2 i beta d);
TMM matches it at machine precision whenever the Airy formula applies
(single film between two half-spaces).
"""

from pathlib import Path
import numpy as np
from tmm_core import tmm_R
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()


def R_airy(wl, d, n0, n1, n2):
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    phase = np.exp(+2j * 2 * np.pi * n1 * d / wl)
    r = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
    return np.abs(r) ** 2


cases = [
    ('a', 1.0, 2.0 + 0.0j,   1.5 + 0.0j,   100.0),
    ('b', 1.0, 3.0 + 0.05j,  1.5 + 0.0j,   500.0),
    ('c', 1.0, 3.0 + 0.05j,  3.11 + 4.89j, 500.0),
]

wl = np.linspace(300.0, 1000.0, 701)

fig, axes = fig_collage(ncols=3, nrows=1, height_cm=6.5)

for (letter, n0, n1, n2, d), ax in zip(cases, axes[0]):
    R_an = np.array([R_airy(w, d, n0, n1, n2) for w in wl])
    R_tm = np.array([tmm_R([n0, n1, n2], [d], w, 0.0, 's') for w in wl])
    ax.plot(wl, R_tm, color=CB10[1], lw=1.0, label='TMM')
    ax.plot(wl, R_an, color='k', ls='--', lw=0.7, label='Airy')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_xlim(300, 1000)
    stack_text = (f"$n_0={n0:g}$\n"
                  f"$n_1={n1.real:g}{'+' + str(n1.imag) + 'i' if n1.imag else ''}$\n"
                  f"$n_2={n2.real:g}{'+' + str(n2.imag) + 'i' if n2.imag else ''}$\n"
                  f"$d={d:g}$ nm")
    ax.text(0.97, 0.03, stack_text, transform=ax.transAxes,
            fontsize=6, ha='right', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
    ax.legend(loc='upper right', fontsize=6)
    add_panel_label(ax, letter)
    err = np.max(np.abs(R_tm - R_an))
    print(f"case {letter}: max |R_TMM - R_Airy| = {err:.2e}")

out = Path(__file__).parent / 'figures' / '04_thin_film_airy.pdf'
finalize(fig, out)
