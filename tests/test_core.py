# Filename: test_core.py
"""
Self-tests for tmm_core.

Every test is a closed-form identity from `docs/theory/theory.tex`.
The suite covers:

    - single-interface Fresnel
    - Airy formula, symmetric film
    - R_s == R_p at normal incidence
    - Brewster angle
    - total internal reflection
    - thick-absorber limit
    - energy conservation R + T + sum(A) = 1 for lossy stacks
    - lossless -> A = 0, passivity A_j >= 0
    - quarter-wave Bragg mirror peak reflectance

Usage
-----
    python test_core.py        # run as script, verbose output
    pytest test_core.py        # works as a pytest module too
"""

import numpy as np
import pytest

from tmm_core import tmm_r, tmm_R, tmm_full


# --- helpers --------------------------------------------------------------
TOL = 1e-10       # numerical tolerance for double-precision identities
_n_checks = 0     # running count for the final summary


def _fmt(v):
    """Pretty-print a scalar (real or complex) or small array."""
    v = np.asarray(v).ravel()
    if v.size == 1:
        x = complex(v[0])
        if abs(x.imag) < 1e-15:
            return f"{x.real:+.9e}"
        return f"{x.real:+.6e} {x.imag:+.6e}j"
    return np.array2string(v, precision=6, separator=", ")


def _check(name, got, want, atol=TOL):
    """Compare got vs want, print a detailed report, raise on failure."""
    global _n_checks
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    ok = err < atol
    tag = "[OK]  " if ok else "[FAIL]"
    print(f"  {tag} {name}")
    print(f"         got  = {_fmt(got)}")
    print(f"         want = {_fmt(want)}")
    print(f"         err  = {err:.2e}  (tol {atol:.1e})")
    _n_checks += 1
    if not ok:
        raise AssertionError(f"{name}: got {got}, expected {want}")


# ==========================================================================
# tmm_r / tmm_R tests
# ==========================================================================

def test_single_interface_fresnel():
    """Single interface reduces to the Fresnel formulae (V18)."""
    n0, n1 = 1.0, 1.5
    print(f"  setup:  n=[{n0}, {n1}], wl=500 nm, theta=0 deg")
    r_s = (n0 - n1) / (n0 + n1)
    _check("single interface, s-pol",
           tmm_r([n0, n1], [], 500, 0, 's'), r_s)
    # At normal incidence p-pol equals s-pol (Convention A, Lekner/Yeh).
    _check("single interface, p-pol  ( = r_s at theta=0, Convention A)",
           tmm_r([n0, n1], [], 500, 0, 'p'), r_s)


def test_airy_formula():
    """Symmetric lossless film vs. analytical Airy formula (V19)."""
    n_inc, n_f, n_sub = 1.0, 1.5, 1.0
    d_f, wl = 100.0, 500.0
    print(f"  setup:  n=[{n_inc}, {n_f}, {n_sub}], d=[{d_f}] nm, wl={wl} nm, theta=0")
    r01 = (n_inc - n_f) / (n_inc + n_f)
    delta = (2 * np.pi / wl) * n_f * d_f
    r_airy = r01 * (1 - np.exp(+2j * delta)) \
             / (1 - r01 ** 2 * np.exp(+2j * delta))
    _check("Airy formula, symmetric film (s-pol)",
           tmm_r([n_inc, n_f, n_sub], [d_f], wl, 0, 's'), r_airy)


def test_normal_incidence_pol_symmetry():
    """At theta = 0 both polarisations give identical reflectance (V17)."""
    n_stack = [1.0, 1.5, 2.0, 1.8, 1.6]
    d_stack = [80.0, 120.0, 50.0]
    print(f"  setup:  n={n_stack}, d={d_stack}, wl=500 nm")
    _check("R_s = R_p at theta = 0  (5-layer stack)",
           tmm_R(n_stack, d_stack, 500, 0, 's'),
           tmm_R(n_stack, d_stack, 500, 0, 'p'))


def test_brewster_angle():
    """p-polarisation vanishes at Brewster's angle (V20)."""
    n1, n2 = 1.0, 1.5
    theta_B = np.degrees(np.arctan(n2 / n1))
    print(f"  setup:  n=[{n1}, {n2}], theta_B = arctan(n2/n1) = {theta_B:.4f} deg")
    _check("Brewster angle, r_p = 0",
           tmm_r([n1, n2], [], 500, theta_B, 'p'), 0.0, atol=1e-12)


def test_total_internal_reflection():
    """Above the critical angle |r| = 1 for both polarisations (7.4)."""
    n1, n2 = 1.5, 1.0
    theta_c = np.degrees(np.arcsin(n2 / n1))
    theta_tir = theta_c + 10.0
    print(f"  setup:  n=[{n1}, {n2}], theta_c = {theta_c:.4f} deg, theta = {theta_tir:.4f} deg")
    _check("TIR |r_s| = 1",
           abs(tmm_r([n1, n2], [], 500, theta_tir, 's')), 1.0)
    _check("TIR |r_p| = 1",
           abs(tmm_r([n1, n2], [], 500, theta_tir, 'p')), 1.0)


def test_thick_absorber_reflection_limit():
    """Thick absorber: exp(+2i delta) -> 0, so r -> r_01 (V19 limit)."""
    n_inc, n_f, n_sub = 1.0, 2.0 + 0.5j, 3.0
    d_f, wl = 5000.0, 500.0
    print(f"  setup:  n=[{n_inc}, {n_f}, {n_sub}], d=[{d_f}] nm, wl={wl} nm")
    r_expected = (n_inc - n_f) / (n_inc + n_f)
    _check("thick absorber, r -> r_01",
           tmm_r([n_inc, n_f, n_sub], [d_f], wl, 0, 's'),
           r_expected, atol=1e-8)


def test_energy_conservation_lossless_film():
    """Closed-form Airy transmission: R + T = 1 (section 7.5)."""
    n0, n1, n2 = 1.0, 1.5, 1.0
    d_f, wl = 100.0, 500.0
    print(f"  setup:  n=[{n0}, {n1}, {n2}], d=[{d_f}] nm, wl={wl} nm")
    R = tmm_R([n0, n1, n2], [d_f], wl, 0, 's')
    r01 = (n0 - n1) / (n0 + n1)
    delta = (2 * np.pi / wl) * n1 * d_f
    T = (1 - r01 ** 2) ** 2 / (1 - 2 * r01 ** 2 * np.cos(2 * delta) + r01 ** 4)
    print(f"          R = {R:.6f},  T (analytical) = {T:.6f}")
    _check("R + T = 1", R + T, 1.0)


def test_reflectance_bounds_lossless():
    """0 <= R <= 1 for any lossless stack, all angles and polarisations."""
    global _n_checks
    n_stack = [1.0, 2.0, 1.5, 1.8]
    d_stack = [100.0, 150.0]
    print(f"  setup:  n={n_stack}, d={d_stack}, wl=500 nm")
    for theta in (0.0, 30.0, 60.0, 80.0):
        for p in ('s', 'p'):
            R_val = tmm_R(n_stack, d_stack, 500, theta, p)
            ok = 0.0 <= R_val <= 1.0 + TOL
            tag = "[OK]  " if ok else "[FAIL]"
            print(f"  {tag} theta={theta:5.1f} deg, pol={p}:  R = {R_val:.6f}")
            if not ok:
                raise AssertionError(f"0<=R<=1 violated: R={R_val}")
            _n_checks += 1


# ==========================================================================
# tmm_full tests  (R, T, per-layer absorption)
# ==========================================================================

def test_full_matches_tmm_r():
    """tmm_full must reproduce the reflection coefficient from tmm_r."""
    n_stack = [1.0, 1.5 + 0.1j, 2.0, 1.8 + 0.05j, 1.52]
    d_stack = [80.0, 120.0, 50.0]
    print(f"  setup:  n={n_stack}, d={d_stack}, wl=550 nm, theta=30, pol=p")
    out = tmm_full(n_stack, d_stack, 550, 30, 'p')
    _check("tmm_full['r'] == tmm_r",
           out['r'], tmm_r(n_stack, d_stack, 550, 30, 'p'))


def test_energy_conservation_lossy():
    """R + T + sum(A) = 1 for a general lossy stack at machine precision."""
    n_stack = [1.0, 1.5, 2.0 + 0.1j, 1.8 + 0.02j, 1.52]
    d_stack = [80.0, 120.0, 50.0]
    print(f"  setup:  n={n_stack}, d={d_stack}, wl=550 nm")
    for pol in ('s', 'p'):
        for theta in (0.0, 25.0, 60.0):
            out = tmm_full(n_stack, d_stack, 550, theta, pol)
            total = out['R'] + out['T'] + out['A'].sum()
            print(f"    pol={pol}, theta={theta:4.1f} deg:  "
                  f"R={out['R']:.4f}  T={out['T']:.4f}  "
                  f"sum(A)={out['A'].sum():.4f}")
            _check(f"energy conservation, pol={pol}, theta={theta:.0f}",
                   total, 1.0, atol=1e-12)


def test_lossless_absorption_zero():
    """In a lossless stack every layer absorption is zero and R + T = 1."""
    n_stack = [1.0, 1.5, 2.0, 1.8, 1.52]
    d_stack = [80.0, 120.0, 50.0]
    print(f"  setup:  n={n_stack}, d={d_stack}, wl=550 nm, theta=30, pol=s")
    out = tmm_full(n_stack, d_stack, 550, 30, 's')
    print(f"          A = {out['A']}")
    _check("lossless stack: max(A_j) == 0",
           out['A'].max(), 0.0, atol=1e-14)
    _check("lossless stack: R + T == 1",
           out['R'] + out['T'], 1.0)


def test_thick_absorber_full():
    """Thick absorber: T = 0 and A_layer = 1 - R (substrate invisible)."""
    n_stack = [1.0, 2.0 + 0.5j, 1.5]
    d_stack = [5000.0]
    print(f"  setup:  n={n_stack}, d={d_stack} nm, wl=500 nm, theta=0, pol=s")
    out = tmm_full(n_stack, d_stack, 500, 0, 's')
    print(f"          R={out['R']:.6f}  T={out['T']:.3e}  A[0]={out['A'][0]:.6f}")
    _check("thick absorber: T == 0",
           out['T'], 0.0, atol=1e-12)
    _check("thick absorber: A == 1 - R",
           out['A'][0], 1.0 - out['R'], atol=1e-12)


def test_passivity():
    """All layer absorptions must be >= 0 in passive media."""
    global _n_checks
    n_stack = [1.0, 1.5 + 0.02j, 2.3 + 0.3j, 1.7 + 0.01j, 1.52]
    d_stack = [100.0, 60.0, 90.0]
    print(f"  setup:  n={n_stack}, d={d_stack} nm, wl=600 nm")
    for theta in (0.0, 45.0):
        for pol in ('s', 'p'):
            out = tmm_full(n_stack, d_stack, 600, theta, pol)
            A_min = out['A'].min()
            ok = A_min >= -1e-14
            tag = "[OK]  " if ok else "[FAIL]"
            print(f"  {tag} theta={theta:4.1f} deg, pol={pol}:  "
                  f"A = {np.array2string(out['A'], precision=5)}")
            if not ok:
                raise AssertionError(f"negative absorption {A_min}")
            _n_checks += 1


def test_bragg_mirror_peak():
    """Quarter-wave Bragg mirror at the centre wavelength.

    Each quarter-wave layer transforms the admittance as Y' = n^2 / Y.
    With L adjacent to the substrate and N HL pairs stacked toward the
    incident medium the effective admittance seen from air becomes
        Y_eff = (n_H / n_L)^(2N) * n_sub,
    giving R_peak = ((n_inc - Y_eff) / (n_inc + Y_eff))^2.
    """
    n_inc, n_sub = 1.0, 1.52
    n_H, n_L = 2.35, 1.38                      # typical TiO2 / SiO2
    wl = 550.0
    N_pairs = 5
    n_qw = [n_inc] + [n_H, n_L] * N_pairs + [n_sub]
    d_qw = [wl / (4 * n_H), wl / (4 * n_L)] * N_pairs
    Y_eff = (n_H / n_L) ** (2 * N_pairs) * n_sub
    R_peak = ((n_inc - Y_eff) / (n_inc + Y_eff)) ** 2
    print(f"  setup:  {N_pairs} HL pairs, n_H={n_H}, n_L={n_L}, n_sub={n_sub}, wl={wl} nm")
    print(f"          Y_eff = {Y_eff:.4f},  R_peak (analytical) = {R_peak:.6f}")
    _check("quarter-wave Bragg mirror peak",
           tmm_R(n_qw, d_qw, wl, 0, 's'), R_peak, atol=1e-10)


def test_reciprocity_T():
    """Lorentz reciprocity: T is invariant under stack reversal (both pols).

    Reverses an asymmetric absorbing stack and verifies T_fwd == T_rev for
    both polarisations over a wavelength sweep. The lossless half-spaces on
    either side ensure the reciprocity identity holds.
    """
    global _n_checks
    n_stack = [1.0, 1.5 + 0.02j, 2.5 + 0.1j, 3.5 + 0.05j, 1.8]
    d_stack = [120.0, 80.0, 50.0]
    print(f"  setup:  n={n_stack}, d={d_stack} nm")
    # At normal incidence the tangential wavevector vanishes, so the reverse
    # incidence angle is also 0 and reciprocity is a direct comparison.
    for pol in ('s', 'p'):
        max_dev = 0.0
        for wl in (400.0, 550.0, 750.0, 1000.0):
            T_fwd = tmm_full(n_stack, d_stack, wl, 0.0, pol)['T']
            T_rev = tmm_full(n_stack[::-1], d_stack[::-1], wl, 0.0, pol)['T']
            max_dev = max(max_dev, abs(T_fwd - T_rev))
        ok = max_dev < 1e-12
        tag = "[OK]  " if ok else "[FAIL]"
        print(f"  {tag} reciprocity T_fwd = T_rev, pol={pol}:  "
              f"max|deltaT| = {max_dev:.2e}")
        if not ok:
            raise AssertionError(f"reciprocity violated for pol={pol}: {max_dev}")
        _n_checks += 1


def test_fresnel_transmittance_single_interface():
    """Fresnel intensity transmittance T at a single interface, both pols.

    Closed form (real n_0, n_1, angle theta_0):
        T_s = 4 n0 n1 cos(t0) cos(t1) / (n0 cos(t0) + n1 cos(t1))**2
        T_p = 4 n0 n1 cos(t0) cos(t1) / (n1 cos(t0) + n0 cos(t1))**2
    """
    global _n_checks
    n0, n1 = 1.0, 1.5
    wl = 500.0
    print(f"  setup:  n=[{n0}, {n1}], wl={wl} nm")
    for theta in (0.0, 30.0, 60.0, 80.0):
        th0 = np.radians(theta)
        th1 = np.arcsin(n0 / n1 * np.sin(th0))
        c0, c1 = np.cos(th0), np.cos(th1)
        T_s_an = 4 * n0 * n1 * c0 * c1 / (n0 * c0 + n1 * c1) ** 2
        T_p_an = 4 * n0 * n1 * c0 * c1 / (n1 * c0 + n0 * c1) ** 2
        T_s = tmm_full([n0, n1], [], wl, theta, 's')['T']
        T_p = tmm_full([n0, n1], [], wl, theta, 'p')['T']
        _check(f"Fresnel T_s, theta={theta:.0f} deg", T_s, T_s_an)
        _check(f"Fresnel T_p, theta={theta:.0f} deg", T_p, T_p_an)


def test_halfwave_invisibility():
    """A lossless film of thickness d = wl0/(2 n_f) returns the bare-substrate
    reflectance at wl0."""
    n0, n_f, n_sub = 1.0, 2.0, 1.5
    wl0 = 550.0
    d_hw = wl0 / (2 * n_f)
    print(f"  setup:  n=[{n0}, {n_f}, {n_sub}], d_HW={d_hw:.2f} nm, wl0={wl0}")
    R_bare = ((n0 - n_sub) / (n0 + n_sub)) ** 2
    R_film = tmm_R([n0, n_f, n_sub], [d_hw], wl0, 0, 's')
    _check("half-wave film -> bare substrate R at wl0",
           R_film, R_bare, atol=1e-14)


def test_quarterwave_AR_zero():
    """Index-matched quarter-wave film (n_f = sqrt(n0 n_s)) gives R(wl0) = 0."""
    n0, n_sub = 1.0, 2.25
    n_f = np.sqrt(n0 * n_sub)
    wl0 = 550.0
    d_qw = wl0 / (4 * n_f)
    print(f"  setup:  n_f = sqrt(n0 n_sub) = {n_f:.6f}, d_QW={d_qw:.2f} nm")
    R_film = tmm_R([n0, n_f, n_sub], [d_qw], wl0, 0, 's')
    _check("index-matched quarter-wave AR -> R(wl0) = 0",
           R_film, 0.0, atol=1e-20)


def test_input_validation():
    """ValueError on mismatched list lengths and invalid polarisation string."""
    global _n_checks
    # len(n_list) != len(d_list) + 2
    with pytest.raises(ValueError):
        tmm_r([1.0, 1.5, 1.0], [100.0, 50.0], 500, 0, 's')
    print("  [OK]   ValueError raised for mismatched list lengths")
    _n_checks += 1
    # invalid polarisation
    with pytest.raises(ValueError):
        tmm_r([1.0, 1.5], [], 500, 0, 'x')
    print("  [OK]   ValueError raised for invalid polarisation 'x'")
    _n_checks += 1
    # same for tmm_full
    with pytest.raises(ValueError):
        tmm_full([1.0, 1.5, 1.0], [100.0, 50.0], 500, 0, 's')
    print("  [OK]   ValueError raised (tmm_full) for mismatched lists")
    _n_checks += 1


def test_pseudo_dielectric_bulk_inversion():
    """Bulk-substrate ellipsometric inversion returns the substrate index.

    For a bare interface air / substrate at theta != 0, the Aspnes formula
        <eps> = sin**2 theta (1 + tan**2 theta [(1 - rho_B)/(1 + rho_B)]**2)
    must equal N_sub**2 to floating-point precision, where the Wikipedia/
    Aspnes sign convention rho_B = -rho is applied to the TMM ratio
    rho = r_p/r_s.
    """
    global _n_checks
    theta = 70.0
    wl = 633.0
    th = np.radians(theta)
    for N_sub in (3.882 + 0.019j, 1.5 + 0.0j, 0.425 + 2.367j):
        r_s = tmm_r([1.0, N_sub], [], wl, theta, 's')
        r_p = tmm_r([1.0, N_sub], [], wl, theta, 'p')
        rho_B = -(r_p / r_s)
        eps_pseudo = (np.sin(th) ** 2
                      * (1 + np.tan(th) ** 2
                         * ((1 - rho_B) / (1 + rho_B)) ** 2))
        _check(f"pseudo-eps -> N^2 for N={N_sub}",
               eps_pseudo, N_sub ** 2, atol=1e-12)


def test_brewster_psi_zero():
    """At Brewster, r_p = 0, so tan(psi) = |r_p|/|r_s| = 0."""
    n0, n1 = 1.0, 1.5
    theta_B = np.degrees(np.arctan(n1 / n0))
    print(f"  setup:  n=[{n0}, {n1}], theta_B = {theta_B:.4f} deg")
    r_s = tmm_r([n0, n1], [], 500, theta_B, 's')
    r_p = tmm_r([n0, n1], [], 500, theta_B, 'p')
    psi = np.degrees(np.arctan(abs(r_p) / abs(r_s)))
    _check("Brewster: tan(psi) = 0 at theta_B",
           psi, 0.0, atol=1e-10)


def test_brewster_phase_jump():
    """arg(r_p) jumps by pi across the Brewster angle.

    Below theta_B, r_p and r_s have opposite signs (air/glass, Convention A);
    above theta_B, r_p has flipped sign, so arg(r_p) differs by pi.
    """
    n0, n1 = 1.0, 1.5
    theta_B = np.degrees(np.arctan(n1 / n0))
    r_below = tmm_r([n0, n1], [], 500, theta_B - 5.0, 'p')
    r_above = tmm_r([n0, n1], [], 500, theta_B + 5.0, 'p')
    phase_below = np.angle(r_below)
    phase_above = np.angle(r_above)
    delta_phase = abs((phase_above - phase_below + np.pi) % (2 * np.pi) - np.pi)
    # wrap to [0, pi]; the jump should give delta_phase close to 0 (mod 2pi)
    # after the +pi shift. Equivalently |phase_above - phase_below| ~ pi.
    raw_gap = abs(phase_above - phase_below)
    print(f"  setup:  theta_B = {theta_B:.4f} deg, +/- 5 deg around")
    print(f"          arg(r_p) below = {np.degrees(phase_below):+.3f} deg")
    print(f"          arg(r_p) above = {np.degrees(phase_above):+.3f} deg")
    print(f"          |phase jump|   = {np.degrees(raw_gap):.3f} deg")
    _check("Brewster: |arg(r_p) jump| = pi",
           raw_gap, np.pi, atol=1e-6)


# ==========================================================================
# Script runner
# ==========================================================================
if __name__ == "__main__":
    tests = [
        test_single_interface_fresnel,
        test_airy_formula,
        test_normal_incidence_pol_symmetry,
        test_brewster_angle,
        test_total_internal_reflection,
        test_thick_absorber_reflection_limit,
        test_energy_conservation_lossless_film,
        test_reflectance_bounds_lossless,
        test_full_matches_tmm_r,
        test_energy_conservation_lossy,
        test_lossless_absorption_zero,
        test_thick_absorber_full,
        test_passivity,
        test_bragg_mirror_peak,
        test_reciprocity_T,
        test_fresnel_transmittance_single_interface,
        test_halfwave_invisibility,
        test_quarterwave_AR_zero,
        test_input_validation,
        test_pseudo_dielectric_bulk_inversion,
        test_brewster_psi_zero,
        test_brewster_phase_jump,
    ]

    line = "=" * 72
    print("\n" + line)
    print("TMM self-tests  (see docs/theory/theory.tex)")
    print(line)
    for i, t in enumerate(tests, 1):
        header = f" {i:2d}. {t.__name__} "
        print("\n" + header + "-" * max(3, 72 - len(header)))
        if t.__doc__:
            # first non-empty line of the docstring
            doc = next((ln.strip() for ln in t.__doc__.splitlines() if ln.strip()), "")
            print(f"  {doc}")
        t()
    print("\n" + line)
    print(f"  {len(tests)} test block(s) passed, {_n_checks} checks total.")
    print(line + "\n")
