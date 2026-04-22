# Nomenclature

All symbols used in `docs/theory/theory.pdf` and in the code. Equation tags refer to the labels in `theory.tex`.

## Conventions

| Item | Choice |
|---|---|
| Time dependence | `exp(-i ω t)` (physics convention) |
| Complex refractive index | `N_j = n_j + i κ_j`, with `κ_j ≥ 0` for absorbing (passive) media |
| Branch cut for `q_j` | `Im(q_j) ≥ 0`; if `Im(q_j) = 0`, then `Re(q_j) > 0` |
| Sign convention for `r_p` | `r_s = r_p` at normal incidence (no sign flip between polarisations); matches Lekner / Yeh / Byrnes Appendix A "Convention A" |
| Stack ordering | `n_list[0]` is the semi-infinite incident medium; `n_list[-1]` is the semi-infinite substrate |
| Units | Any consistent length unit for `wl` and `d_list` (e.g. both in nm) |
| Polarisation | `'s'` = TE (E-field perpendicular to plane of incidence), `'p'` = TM (E-field in plane of incidence) |

## Symbol table

| Symbol | Meaning | Unit | Range |
|---|---|---|---|
| `λ` (`wl`) | Vacuum wavelength | length | `> 0` |
| `k₀ = 2π/λ` | Vacuum wave number | 1/length | `> 0` |
| `θ₀` | Angle of incidence (in incident medium) | degrees | `0 ≤ θ₀ < 90` |
| `N_j = n_j + i κ_j` | Complex refractive index of medium `j` | dimensionless | `κ_j ≥ 0` |
| `β = k_x = k₀ n₀ sin θ₀` | Transverse wave-vector component (conserved by Snell's law) | 1/length | real in a lossless incident medium |
| `q_j = √((k₀ N_j)² − β²)` | Normal wave-vector component in layer `j` | 1/length | `Im(q_j) ≥ 0` enforced |
| `Q_j = q_j / N_j²` | `p`-polarisation auxiliary variable | 1/length | — |
| `r_s = (q_{j-1} − q_j)/(q_{j-1} + q_j)` | Fresnel reflection coefficient, `s`-polarisation | dimensionless | `|r| ≤ 1` (passive) |
| `r_p = −(Q_{j-1} − Q_j)/(Q_{j-1} + Q_j)` | Fresnel reflection coefficient, `p`-polarisation | dimensionless | `|r| ≤ 1` (passive) |
| `t_s, t_p` | Fresnel transmission coefficients | dimensionless | — |
| `φ_j = q_j d_j` | Complex phase accumulated across layer `j` | dimensionless (rad) | `Im(φ_j) ≥ 0` |
| `P_j` | Propagation matrix of layer `j` | dimensionless 2×2 | `diag(e^{−iφ_j}, e^{+iφ_j})` |
| `T^{(j,j+1)}` | Interface matrix across boundary `j → j+1` | dimensionless 2×2 | — |
| `M = T^{(0,1)} ∏_j (P_j T^{(j,j+1)})` | Full stack transfer matrix (right → left) | dimensionless 2×2 | — |
| `r = M[1,0] / M[0,0]` | Complex amplitude reflection coefficient of the stack | dimensionless | `|r| ≤ 1` (passive) |
| `t = 1 / M[0,0]` | Complex amplitude transmission coefficient of the stack | dimensionless | — |
| `R = |r|²` | Reflectance (intensity) | dimensionless | `0 ≤ R ≤ 1` (passive) |
| `T` | Transmittance (intensity) | dimensionless | `0 ≤ T` |
| `A_j` | Absorbed fraction in layer `j` | dimensionless | `A_j ≥ 0`; `R + T + Σ A_j = 1` for passive stacks |
| `ρ = r_p / r_s`, `ψ`, `Δ` | Ellipsometric ratio and angles | dimensionless / degrees | `ρ = tan(ψ) e^{iΔ}` |
