# Filename: tmm_core.py
"""
Transfer Matrix Method (TMM) for the optical response of a planar multilayer.

Given refractive indices, layer thicknesses, wavelength, angle of incidence
and polarisation, this module returns

    tmm_r    -> complex reflection coefficient r
    tmm_R    -> reflectance R = |r|^2
    tmm_full -> dict with r, R, T and per-layer absorption A

Physics reference: docs/theory/theory.tex  (step-by-step derivation).

Conventions
-----------
- Time convention   : exp(-i*omega*t)
- Refractive index  : N_j = n_j + i*kappa_j  with kappa_j >= 0 for absorbing media
- Branch cut        : Im(q_j) >= 0  (forward wave decays into absorber)
- Stack ordering    : light enters from n_list[0], substrate = n_list[-1]
- p-pol sign        : Convention A (Lekner/Yeh): r_s = r_p at normal incidence
                      r_p = -(Q_j - Q_{j+1}) / (Q_j + Q_{j+1})
- Total matrix      : M = T^(0,1) * P_1 * T^(1,2) * ... * P_N * T^(N,N+1)
                      maps substrate amplitudes to incident amplitudes (right to left)

Interface
---------
    r    = tmm_r   (n_list, d_list, wl, theta_0_deg=0.0, pol='s')  -> complex
    R    = tmm_R   (n_list, d_list, wl, theta_0_deg=0.0, pol='s')  -> float
    out  = tmm_full(n_list, d_list, wl, theta_0_deg=0.0, pol='s')  -> dict
           out = {'r', 'R', 'T', 'A'}

with
    n_list : [N_0, N_1, ..., N_N, N_{N+1}]   (length N+2, complex allowed)
    d_list : [d_1, ..., d_N]                  (length N, same units as wl)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Main routine: complex reflection coefficient r
# ---------------------------------------------------------------------------
def tmm_r(n_list, d_list, wl, theta_0_deg=0.0, pol='s'):
    """
    Complex reflection coefficient r of a multilayer stack.

    Parameters
    ----------
    n_list : array_like, complex
        Refractive indices [n_inc, n_1, ..., n_N, n_sub].  Use n = n' + i*kappa
        with kappa >= 0 for absorbing media.
    d_list : array_like, float
        Layer thicknesses [d_1, ..., d_N] in the same length unit as wl.
        Must satisfy  len(d_list) == len(n_list) - 2.
    wl : float
        Vacuum wavelength (same unit as d_list, e.g. nm).
    theta_0_deg : float, optional
        Angle of incidence in degrees, measured in the incident medium.
    pol : {'s', 'p'}, optional
        's' for TE (E perpendicular to the plane of incidence),
        'p' for TM (E in the plane of incidence).

    Returns
    -------
    complex
        Reflection coefficient  r = M[1, 0] / M[0, 0]   (eq. r-t-from-M)

    Notes
    -----
    See docs/theory/theory.tex for the full derivation.  Internal symbols:
      beta  = conserved tangential wavenumber (eq. snell)
      q_j   = normal wavenumber in layer j (eq. kz, eq. branchcut)
      Q_j   = q_j / N_j^2  (p-pol admittance, eq. qQ-defs)
      phi_j = q_j * d_j    (layer phase, eq. phi-def)
      T_j   = interface matrix T^(j,j+1) (eq. Tj-box)
      P_j   = propagation matrix P_j (eq. Pj-box)
      M     = total transfer matrix (eq. M-total)
    Convention A (Lekner/Yeh): r_s = r_p at normal incidence.
    """
    # --- 0. input handling ------------------------------------------------
    N = np.asarray(n_list, dtype=np.complex128)
    d = np.asarray(d_list, dtype=np.float64)

    if len(N) != len(d) + 2:
        raise ValueError(
            f"Dimension mismatch: {len(N)} indices for "
            f"{len(d)} layers (expected len(n_list) == len(d_list) + 2)."
        )
    if pol not in ('s', 'p'):
        raise ValueError("pol must be 's' (TE) or 'p' (TM)")

    # --- 1. wave-vector components in every medium -----------------------
    # Vacuum wavenumber k0 = 2pi/lambda_0; conserved beta = k_x (eq. snell).
    k0 = 2.0 * np.pi / wl
    beta = k0 * N[0] * np.sin(np.radians(theta_0_deg))

    # Normal wavenumber q_j = sqrt((k0*N_j)^2 - beta^2)    (eq. kz)
    # Physical branch cut: Im(q_j) >= 0, Re > 0 if Im == 0 (eq. branchcut)
    q = np.sqrt((k0 * N) ** 2 - beta ** 2)
    flip = (q.imag < 0) | ((q.imag == 0) & (q.real < 0))
    q = np.where(flip, -q, q)

    # --- 2. p-pol admittance Q_j = q_j / N_j^2  (eq. qQ-defs) ----------
    # For s-pol the admittance is q_j itself.
    if pol == 'p':
        Q = q / N ** 2

    # --- 3. total transfer matrix M  (eq. M-total) -----------------------
    # Build  M = T^(0,1) * P_1 * T^(1,2) * P_2 * ... * T^(N,N+1)
    # by sweeping from j = 1 (first interface) to j = N+1 (substrate).
    # No propagation matrix for the semi-infinite substrate.
    M = np.eye(2, dtype=np.complex128)

    for j in range(1, len(N)):

        # Interface matrix T^(j-1,j)  (eq. Tj-box)
        # Fresnel coefficients (eq. rs-q, rp-Q, ts, tp), Convention A:
        if pol == 's':
            q_prev = q[j - 1]
            q_curr = q[j]
            r_j = (q_prev - q_curr) / (q_prev + q_curr)
            t_j = 2.0 * q_prev / (q_prev + q_curr)
        else:  # pol == 'p'
            Q_prev = Q[j - 1]
            Q_curr = Q[j]
            r_j = -(Q_prev - Q_curr) / (Q_prev + Q_curr)
            t_j = (N[j - 1] / N[j]) * 2.0 * Q_prev / (Q_prev + Q_curr)
        T_j = np.array([[1.0,  r_j],
                        [r_j,  1.0]], dtype=np.complex128) / t_j
        M = M @ T_j

        # Propagation matrix P_j  (eq. Pj-box)
        # Not applied to the semi-infinite substrate (j == N+1).
        # phi_j = q_j * d_j  (eq. phi-def); Im(phi_j) >= 0 ensures
        # |exp(+i*phi_j)| <= 1 for absorbing media (physical decay).
        if j < len(N) - 1:
            phi = q[j] * d[j - 1]
            P_j = np.array([[np.exp(-1j * phi), 0.0],
                            [0.0,  np.exp(+1j * phi)]], dtype=np.complex128)
            M = M @ P_j

    # --- 4. reflection coefficient  (eq. r-t-from-M) ---------------------
    return M[1, 0] / M[0, 0]


# ---------------------------------------------------------------------------
# Convenience: reflectance R = |r|^2
# ---------------------------------------------------------------------------
def tmm_R(n_list, d_list, wl, theta_0_deg=0.0, pol='s'):
    """Reflectance R = |r|^2.  Parameters are identical to tmm_r."""
    return np.abs(tmm_r(n_list, d_list, wl, theta_0_deg, pol)) ** 2


# ---------------------------------------------------------------------------
# Full result: r, R, T, per-layer absorption
# ---------------------------------------------------------------------------
def tmm_full(n_list, d_list, wl, theta_0_deg=0.0, pol='s'):
    """
    Complete TMM result: r, R, T and layer-resolved absorption.

    Parameters
    ----------
    Same as tmm_r.  Additional requirement: the incident medium must be
    lossless (Im(n_list[0]) == 0) so that the incident Poynting flux is
    a well-defined normaliser.

    Returns
    -------
    dict with keys
        'r' : complex            reflection coefficient r     (eq. r-t-from-M)
        'R' : float              reflectance  R = |r|^2
        'T' : float              transmittance T = S_N^(R)     (eq. sum-rule)
        'A' : ndarray, shape (N,)
              absorbed fraction A_j in each finite layer j = 1..N
                                                              (eq. abs-per-layer)

    Energy conservation (passive media, lossless incidence):
        R + T + A.sum() == 1                                  (eq. RTA-sum)

    Notes
    -----
    See docs/theory/theory.tex for the full derivation.  Internal symbols:
      beta    = conserved tangential wavenumber k_x           (eq. snell)
      q_j     = normal wavenumber in layer j                  (eq. kz, eq. qQ-defs)
      Q_j     = q_j / N_j^2   (p-pol natural variable)        (eq. qQ-defs)
      qbar_j  = |N_j|^2 * conj(Q_j)   (p-pol flux prefactor)  (eq. qbar-def)
      phi_j   = q_j * d_j     (layer phase)                   (eq. phi-def)
      alpha_j^(L/R), beta_j^(L/R)
              forward / backward plane-wave amplitudes at the left / right
              edge of layer j                                 (eq. edge-amp-L/R)
      T^(j,j+1) interface matrix                              (eq. Tj-box)
      P_j     propagation matrix                              (eq. Pj-box)
      M       total transfer matrix                           (eq. M-total)

    The transfer matrix is built identically to tmm_r (Convention A: Lekner/Yeh,
    r_s = r_p at normal incidence).  Per-layer absorption is the Poynting-flux
    balance (eq. abs-per-layer)
        A_j = [S_j^(L) - S_j^(R)] / Re(adm_inc),
    evaluated with the symmetric flux formulas (eq. S-s-split, eq. S-p-split)
        S_s^(j)(z) = Re(q_j)   (|E_f|^2 - |E_b|^2) - 2 Im(q_j)    Im(E_f E_b*)
        S_p^(j)(z) = Re(qbar_j)(|E_f|^2 - |E_b|^2) + 2 Im(qbar_j) Im(E_f E_b*)
    at (E_f, E_b) = (alpha_j^(L), beta_j^(L)) and (alpha_j^(R), beta_j^(R)).
    """
    # --- 0. input handling ------------------------------------------------
    N = np.asarray(n_list, dtype=np.complex128)
    d = np.asarray(d_list, dtype=np.float64)

    if len(N) != len(d) + 2:
        raise ValueError(
            f"Dimension mismatch: {len(N)} indices for "
            f"{len(d)} layers (expected len(n_list) == len(d_list) + 2)."
        )
    if pol not in ('s', 'p'):
        raise ValueError("pol must be 's' (TE) or 'p' (TM)")
    if N[0].imag != 0.0:
        raise ValueError("tmm_full requires a lossless incident medium "
                         "(Im(n_list[0]) == 0).")

    Nlay = len(d)

    # --- 1. wave-vector components in every medium -----------------------
    # Identical to tmm_r (eq. snell, eq. kz, eq. branchcut).
    k0 = 2.0 * np.pi / wl
    beta = k0 * N[0] * np.sin(np.radians(theta_0_deg))
    q = np.sqrt((k0 * N) ** 2 - beta ** 2)
    flip = (q.imag < 0) | ((q.imag == 0) & (q.real < 0))
    q = np.where(flip, -q, q)

    # --- 2. p-pol admittance Q and flux prefactor qbar  -----------------
    # Q_j = q_j / N_j^2  (eq. qQ-defs); qbar_j = |N_j|^2 * conj(Q_j) (eq. qbar-def).
    if pol == 'p':
        Q = q / N ** 2
        qbar = (np.abs(N) ** 2) * np.conj(Q)

    # --- 3. total transfer matrix M, keep interface matrices T_list ------
    # Identical construction to tmm_r (Convention A, eq. M-total).
    # T_list[j-1] corresponds to T^(j-1,j).
    M = np.eye(2, dtype=np.complex128)
    T_list = []

    for j in range(1, len(N)):

        # Interface matrix T^(j-1,j)  (eq. Tj-box)
        # Fresnel coefficients (eq. rs-q, rp-Q, ts, tp), Convention A:
        if pol == 's':
            q_prev = q[j - 1]
            q_curr = q[j]
            r_j = (q_prev - q_curr) / (q_prev + q_curr)
            t_j = 2.0 * q_prev / (q_prev + q_curr)
        else:  # pol == 'p'
            Q_prev = Q[j - 1]
            Q_curr = Q[j]
            r_j = -(Q_prev - Q_curr) / (Q_prev + Q_curr)
            t_j = (N[j - 1] / N[j]) * 2.0 * Q_prev / (Q_prev + Q_curr)
        T_j = np.array([[1.0,  r_j],
                        [r_j,  1.0]], dtype=np.complex128) / t_j
        T_list.append(T_j)
        M = M @ T_j

        # Propagation matrix P_j  (eq. Pj-box), skipped for substrate half-space
        if j < len(N) - 1:
            phi = q[j] * d[j - 1]                                       # eq. phi-def
            P_j = np.array([[np.exp(-1j * phi), 0.0],
                            [0.0,  np.exp(+1j * phi)]], dtype=np.complex128)
            M = M @ P_j

    # --- 4. r, R, t, T  (eq. r-t-from-M, eq. sum-rule) -------------------
    r = M[1, 0] / M[0, 0]
    t_stack = 1.0 / M[0, 0]                      # forward amplitude in substrate
    R = float(np.abs(r) ** 2)
    # Flux prefactors Re(adm) at incident / substrate half-spaces:
    # adm = q (s) or qbar (p).
    if pol == 's':
        adm_inc_re = q[0].real
        adm_sub_re = q[-1].real
    else:
        adm_inc_re = qbar[0].real
        adm_sub_re = qbar[-1].real
    T = float(adm_sub_re * np.abs(t_stack) ** 2 / adm_inc_re)

    # --- 5. forward-propagate amplitudes, per-layer absorption -----------
    # Starting state at z = z_0:  (alpha_0^R, beta_0^R) = (A_0, B_0) = (1, r)
    #                                                            (eq. bc, eq. chain)
    # For each layer j = 1..N:
    #   (alpha_j^L, beta_j^L) = T^(j-1,j)^{-1} * (alpha_{j-1}^R, beta_{j-1}^R)
    #                                                            (eq. Tj-box, inverted)
    #   (alpha_j^R, beta_j^R) = diag(exp(+i phi_j), exp(-i phi_j))
    #                           * (alpha_j^L, beta_j^L)          (eq. prop-LR)
    v = np.array([1.0, r], dtype=np.complex128)
    A_layers = np.zeros(Nlay, dtype=np.float64)

    for j in range(1, Nlay + 1):

        # left-edge amplitudes of layer j (interface inverse, stable solve)
        v = np.linalg.solve(T_list[j - 1], v)
        alpha_L, beta_L = v

        # right-edge amplitudes of layer j  (eq. prop-LR)
        phi_j = q[j] * d[j - 1]
        alpha_R = alpha_L * np.exp(+1j * phi_j)
        beta_R  = beta_L  * np.exp(-1j * phi_j)

        # absorption A_j = [S_j^(L) - S_j^(R)] / Re(adm_inc)   (eq. abs-per-layer)
        # Flux formulas:  S_s = Re(q)   (|a|^2 - |b|^2) - 2 Im(q)    Im(a b*)
        #                 S_p = Re(qbar)(|a|^2 - |b|^2) + 2 Im(qbar) Im(a b*)
        # (eq. S-s-split, eq. S-p-split).
        adm_j = q[j] if pol == 's' else qbar[j]
        sign = -1.0  # both pols: r_s=r_p convention flips (E_f +/- E_b) in S_p
        cross_L = (alpha_L * np.conj(beta_L)).imag               # Im(E_f E_b*)
        cross_R = (alpha_R * np.conj(beta_R)).imag
        S_L = adm_j.real * (np.abs(alpha_L) ** 2 - np.abs(beta_L) ** 2) \
              + sign * 2.0 * adm_j.imag * cross_L
        S_R = adm_j.real * (np.abs(alpha_R) ** 2 - np.abs(beta_R) ** 2) \
              + sign * 2.0 * adm_j.imag * cross_R
        A_layers[j - 1] = (S_L - S_R) / adm_inc_re

        # pass right-edge amplitudes to the next interface
        v = np.array([alpha_R, beta_R])

    return {'r': r, 'R': R, 'T': T, 'A': A_layers}
