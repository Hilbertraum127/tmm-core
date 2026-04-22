"""
13 - Fabry-Perot finesse: TMM-extracted vs. analytic F = pi sqrt(R) / (1 - R).

Symmetric etalon: two identical Bragg mirrors around a half-wave air
spacer. For each mirror pair count N (hence each mirror reflectance R_m),
TMM computes T(lambda) across two neighboring resonances; the finesse
    F_meas = FSR / FWHM
uses the empirically measured FSR (distance between two fitted peaks)
because DBR mirrors add a phase-delay-on-reflection contribution that
extends the effective cavity length beyond the geometric spacer. The
analytic F_analytic = pi sqrt(R_m) / (1 - R_m) uses the single-mirror
reflectance computed from tmm_R at lambda0.
"""

from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from tmm_core import tmm_R, tmm_full
from plotter import apply_style, fig_collage, CB10, add_panel_label, finalize

apply_style()

wl0 = 1000.0
n0, n_H, n_L = 1.0, 2.35, 1.46
d_H, d_L = wl0 / (4 * n_H), wl0 / (4 * n_L)
n_cav = 1.0
# Use a long cavity so free spectral range is smaller than the DBR stopband,
# guaranteeing two cavity modes fall inside the high-R window.
d_cav = 10000.0

Ns = [4, 5, 6, 7, 8]


def mirror_pairs(N):
    return [n_H, n_L] * N


def etalon(N):
    mirror = mirror_pairs(N)
    n_list = [n0] + mirror + [n_cav] + mirror[::-1] + [n0]
    d_list = [d_H, d_L] * N + [d_cav] + [d_L, d_H] * N
    return n_list, d_list


def single_mirror_R(N):
    n_list = [n0] + mirror_pairs(N) + [n_cav]
    d_list = [d_H, d_L] * N
    return tmm_R(n_list, d_list, wl0, 0.0, 's')


def lorentzian(wl, A, wl_c, gamma, C):
    return C + A * gamma ** 2 / ((wl - wl_c) ** 2 + gamma ** 2)


# Coarse scan to locate two adjacent peaks near lambda0
def find_two_peaks(n_list, d_list, wl_center, half_span):
    wl = np.linspace(wl_center - half_span, wl_center + half_span, 20001)
    T = np.array([tmm_full(n_list, d_list, w, 0.0, 's')['T'] for w in wl])
    # find local maxima above 0.5
    above = T > 0.5
    peak_idx = []
    for i in range(1, len(T) - 1):
        if T[i] > T[i - 1] and T[i] >= T[i + 1] and above[i]:
            peak_idx.append(i)
    return wl, T, peak_idx


def fit_peak(wl, T, center_idx, window_pts=200):
    lo = max(0, center_idx - window_pts)
    hi = min(len(wl), center_idx + window_pts)
    wl_w, T_w = wl[lo:hi], T[lo:hi]
    p0 = [T_w.max() - T_w.min(), wl[center_idx], 0.05, T_w.min()]
    popt, _ = curve_fit(lorentzian, wl_w, T_w, p0=p0, maxfev=20000)
    return popt[1], 2 * abs(popt[2])


F_meas, F_an, R_m_list = [], [], []
for N in Ns:
    n_list, d_list = etalon(N)
    # expected FSR ~ lambda0^2 / (2 * n_cav * d_cav) = 50 nm for these params
    wl, T, peaks = find_two_peaks(n_list, d_list, wl0, 80.0)
    # keep the two peaks nearest to wl0
    peaks.sort(key=lambda i: abs(wl[i] - wl0))
    p1_idx, p2_idx = sorted(peaks[:2])
    wl_c1, fwhm1 = fit_peak(wl, T, p1_idx)
    wl_c2, fwhm2 = fit_peak(wl, T, p2_idx)
    FSR = abs(wl_c2 - wl_c1)
    fwhm = 0.5 * (fwhm1 + fwhm2)
    F_meas.append(FSR / fwhm)

    Rm = single_mirror_R(N)
    F_an.append(np.pi * np.sqrt(Rm) / (1 - Rm))
    R_m_list.append(Rm)
    print(f"N = {N}: R_mirror = {Rm:.4f}, FSR = {FSR:.3f} nm,"
          f" FWHM = {fwhm:.4f} nm, F_meas = {F_meas[-1]:.1f},"
          f" F_an = {F_an[-1]:.1f}")

F_meas = np.array(F_meas)
F_an = np.array(F_an)

fig, axes = fig_collage(ncols=2, nrows=1, height_cm=6.5)
ax1, ax2 = axes[0][0], axes[0][1]

N_show = Ns[2]
n_list, d_list = etalon(N_show)
wl_fine = np.linspace(wl0 - 80.0, wl0 + 80.0, 4001)
T_fine = np.array([tmm_full(n_list, d_list, w, 0.0, 's')['T'] for w in wl_fine])
ax1.plot(wl_fine, T_fine, color=CB10[0], lw=0.8, label=f'N = {N_show}')
ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Transmittance")
ax1.legend(loc='best')
add_panel_label(ax1, 'a')

ax2.plot(F_an, F_meas, color=CB10[1], marker='o', ls='none', label='TMM')
xmax = float(max(F_an.max(), F_meas.max()) * 1.05)
ax2.plot([0, xmax], [0, xmax], color=CB10[9], lw=0.7, ls='--', label='y = x')
ax2.set_xlabel(r"$F_\mathrm{analytic}$")
ax2.set_ylabel(r"$F_\mathrm{measured}$")
ax2.legend(loc='best')
add_panel_label(ax2, 'b')

out = Path(__file__).parent / 'figures' / '12_fabry_perot_finesse.pdf'
finalize(fig, out)
