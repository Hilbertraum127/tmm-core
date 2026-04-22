"""
Shared plotting utilities for the examples gallery.

- apply_style()              : sets rcParams (font 7, Arial, direction in, etc.)
- CB10, MARKERS10            : colorblind-safe palette + matching markers
- fig_single()               : 8.5 cm x 6 cm figure with standard margins
- fig_collage(nc, nr, h_cm)  : 18 cm wide GridSpec collage
- add_panel_label(ax, 'a')   : bold 8 pt label at (-0.25, 1.1)
- residual_panel(ax, ...)    : log-scale residual styling helper
- finalize(fig, png_path)    : tight_layout -> savefig(300 dpi PNG) -> plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


CB10 = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#332288",
    "#E69F00", "#882255", "#56B4E9", "#997700", "#000000",
]

MARKERS10 = ["o", "s", "^", "D", "v", "p", "h", "<", ">", "8"]


def apply_style():
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.titlesize': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': False,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })


def fig_single():
    w_cm, h_cm = 8.5, 6.0
    fig, ax = plt.subplots(figsize=(w_cm / 2.54, h_cm / 2.54))
    fig.subplots_adjust(
        left=1.25 / w_cm,
        right=1 - 0.25 / w_cm,
        bottom=1.25 / h_cm,
        top=1 - 0.25 / h_cm,
    )
    return fig, ax


def fig_collage(ncols, nrows, height_cm=None):
    w_cm = 18.0
    panel_h = 6.0
    h_cm = height_cm if height_cm is not None else panel_h * nrows
    fig = plt.figure(figsize=(w_cm / 2.54, h_cm / 2.54))
    gs = GridSpec(
        nrows, ncols, figure=fig,
        left=1.25 / w_cm,
        right=1 - 0.25 / w_cm,
        bottom=1.25 / h_cm,
        top=1 - 0.25 / h_cm,
        wspace=0.45,
        hspace=0.55,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    return fig, axes


def add_panel_label(ax, letter):
    ax.text(-0.25, 1.1, letter, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left')


def residual_panel(ax, x, residual, xlabel=None,
                   ylabel=r"$|\mathrm{TMM} - \mathrm{analytic}|$"):
    ax.semilogy(x, np.abs(residual) + 1e-300, color=CB10[9], lw=0.7)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def finalize(fig, png_path):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"wrote {png_path}")
    if plt.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
        plt.show()
