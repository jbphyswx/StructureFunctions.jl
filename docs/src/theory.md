# Theory

What the package computes, in the conventions the code uses. Every formula here is one the tests hold
the code to; the [Validation](validation.md) page lists the oracle for each.

## Definition

For two points ``x`` and ``x' = x + \vec r`` of a field ``u``, the **increment** is
``δu = u(x') - u(x)``. A structure function is the average of a polynomial in the increment over all
pairs whose separation ``|\vec r|`` falls in a bin, and the package's operators are those
polynomials. With ``\hat r = \vec r / |\vec r|`` the unit separation:

| operator | value on one pair | shorthand |
|---|---|---|
| `SecondOrderStructureFunctionType` | ``‖δu‖²`` | `S2SFType` |
| `ProjectedStructureFunctionType{2,0}` | ``δu_L^2``, ``δu_L = δu · \hat r`` | `L2SFType` |
| `ProjectedStructureFunctionType{0,2}` | ``‖δu_T‖² = ‖δu‖² - δu_L^2`` | `T2SFType` |
| `ThirdOrderStructureFunctionType` | ``δu_L ‖δu‖²`` | `S3SFType` |
| `ProjectedStructureFunctionType{3,0}` | ``δu_L^3`` | `L3SFType` |
| `ProjectedStructureFunctionType{1,2}` | ``δu_L ‖δu_T‖²`` | `L1T2SFType` |
| `ProjectedStructureFunctionType{2,1}`, `{0,3}` | ``δu_L^2\, (δu·\hat n)``, ``(δu·\hat n)^3`` | `L2T1SFType`, `T3SFType` |
| `ProjectedStructureFunctionType{NL,NT}` | ``δu_L^{NL}`` times ``‖δu_T‖^{NT}`` for ``NT = 2``, ``(δu·\hat n)^{NT}`` otherwise | |
| `TransverseComponentSecondOrderStructureFunctionType`, `LongitudinalTransverseComponentThirdOrderStructureFunctionType` | ``‖δu_T‖²/(D-1)``, ``δu_L ‖δu_T‖²/(D-1)`` — per transverse component | `T2ComponentSFType`, `L1T2ComponentSFType` |
| `FullVectorStructureFunctionType{NF}` | ``‖δu‖^{NF}`` | |
| `ScalarStructureFunctionType{P}` | ``(δθ)^P`` on a scalar channel | `ScalarSFType` |
| `MixedStructureFunctionType{NL,NT,P}` | ``δu_L^{NL} ‖δu_T‖^{NT} (δθ)^P`` | `MixedSFType` |
| `VectorDotStructureFunctionType(a, b)`, `ScalarDotStructureFunctionType(a, b)` | ``δu^{(a)} · δu^{(b)}``, ``δθ^{(a)}\, δθ^{(b)}`` across channels | |
| `MomentTensorOperator{P}` | the whole tensor ``δu_{i_1} ⋯ δu_{i_P}`` | |

``δu_L`` is the **longitudinal** increment, the component along the separation, and
``‖δu_T‖² = ‖δu‖² - δu_L^2`` is the **transverse** energy, the squared norm of everything
perpendicular to it — in any dimension. The signed transverse component ``δu · \hat n`` that the
odd transverse operators read needs a direction ``\hat n`` perpendicular to ``\hat r``, which is a
convention: in two dimensions ``\hat n = \hat z × \hat r = (-\hat r_2, \hat r_1)``, and in three the
`CanonicalTransverseBasis` is ``\hat n = \hat z × \hat r / |\hat z × \hat r|`` continued about
``\hat x`` where ``\hat r ∥ \hat z``, with ``\hat r × \hat n`` as the second transverse direction.
`ReferenceAxisTransverseBasis(a)` is the same rule about an axis ``a`` of the caller's choosing. The
convention travels with the operator into every result, and every rule must give a first transverse
vector that is **odd** under ``\hat r ↦ -\hat r``: reading a pair from its other end flips ``δu`` and
``\hat r`` together, and the transverse component has to flip with them so that the operator's value
is the same from either end.

That swap invariance holds for every operator above except one class. A scalar increment
``δθ = θ(x') - θ(x)`` changes sign when the pair is read backwards, so an operator odd in a scalar
increment (`ScalarSFType{3}`, `MixedSFType{1,0,1}`, …) has no value on an unordered pair without a
rule. The package reads every pair from the lower to the upper end along the first coordinate that
separates the two points — on a sphere from south to north, and along one parallel from west to
east by the shorter way — so those moments are independent of how the input happens to be ordered.
Where neither end comes first (a lag half-turning a periodic direction) the two readings are averaged
and the odd moment is zero.

### Weights

With one weight per point or cell, every statistic is
``\sum_{ij} w_i w_j\, \mathrm{sf}(δu_{ij}, \hat r_{ij}) / \sum_{ij} w_i w_j`` over the pairs of a bin,
and the counts become the weighted pair mass. `cell_measure(grid)` gives a grid's cell areas as
weights, which turns a lat-lon sum into an area average.

## Second order and spectra

For a homogeneous isotropic field in ``D`` dimensions the second-order trace and the shell spectrum
``E(k)`` (``∫_0^∞ E\,dk`` the variance) are related by

```math
S_2(r) = 2 ∫_0^∞ E(k) \left[1 - \mathrm{kernel}_D(kr)\right] dk, \qquad
\mathrm{kernel}_1 = \cos x, \quad \mathrm{kernel}_2 = J_0(x), \quad \mathrm{kernel}_3 = \frac{\sin x}{x},
```

the kernel being the angular average of ``\cos(\vec k · \vec r)``. The package inverts this in
`isotropic_spectrum` for the density ``P(k)`` over ``d^D k`` (``E = Ω_D k^{D-1} P``, `shell_spectrum`),
and the ``k = 0`` mode is not recoverable: a structure function is blind to the mean and the
variance. On a grid the same information is available **exactly** through the lag space
(`gridded_spectrum`): the transform of the unbiased autocovariance ``C(h) = σ̂² - D(h)/2`` over every
lag, with no direction averaged and no separation binned. With cells missing that is still the
exact transform of the exact structure function, equal to the complete field's spectrum in
expectation; a bounded direction gives the windowed (Blackman–Tukey) estimate on a padded lag grid,
with an optional taper.

For a two-dimensional vector field the longitudinal and transverse functions split the spectrum into
its gradient (divergent, ``E``) and curl (rotational, ``B``) parts:

```math
D_{LL} + D_{TT} = 2 ∫ (E_E + E_B)\,[1 - J_0(kr)]\,dk, \qquad
D_{LL} - D_{TT} = 2 ∫ (E_E - E_B)\, J_2(kr)\,dk,
```

inverted by `helmholtz_spectra(L2, T2, k)`. On a sphere the counterparts are Legendre and Wigner
series: ``C_l = -π ∫_0^π D(σ) P_l(\cos σ) \sin σ\, dσ`` for a scalar or the trace, and
``C^E_l ∓ C^B_l`` from ``D_{LL} ∓ D_{TT}`` through ``d^l_{1,-1}`` and ``d^l_{11}``.

## Third order, exact laws and fluxes

In the inertial range of homogeneous isotropic turbulence the third-order functions are linear in
``r`` with the dissipation as the slope — Kolmogorov's four-fifths law ``⟨δu_L^3⟩ = -\tfrac45 ε r``,
the four-thirds law ``⟨δu_L ‖δu‖²⟩ = -\tfrac43 ε r`` and Yaglom's law
``⟨δu_L (δθ)^2⟩ = -\tfrac43 ε_θ r``. Each is stated for one specific moment; the
[KHM](khm.md) page inverts them.

Scale by scale, the interscale energy flux of a two-dimensional flow follows from the advective
structure function ``SF_A = ⟨δu · δ\mathcal{A}_u⟩``, ``\mathcal{A}_u = u·∇u``, without any isotropy
assumption, and — for an isotropic flow — from the third-order functions themselves, with the
boundary terms their integrations by parts leave at the largest separation ``R``:

```math
\begin{aligned}
Π_K &= -\tfrac{K}{2} ∫_0^R SF_A\, J_1(Kr)\, dr \\
Π_K &= -\tfrac{K^2}{4} ∫_0^R S_3\, J_2(Kr)\, dr - \tfrac{K}{4} S_3(R) J_1(KR) \\
Π_K &= -\tfrac{K^3}{12} ∫_0^R L_3\, J_3(Kr)\, r\, dr - \tfrac{K^2}{12}\left[R\, L_3(R) J_2(KR) + \tfrac{3}{K} S_3(R) J_1(KR)\right]
\end{aligned}
```

with ``S_3 = ⟨δu_L‖δu‖²⟩`` and ``L_3 = ⟨δu_L^3⟩``, and the same forms for the scalar-variance and
enstrophy fluxes (`spectral_flux`, `enstrophy_flux`). The boundary terms are not small: without them
the estimate can change sign. The sign convention is positive towards small scales.

## Exact algorithms

### Polynomial moments on a grid by transform

Every operator above except the odd norms is a polynomial of degree ``p`` in the increment, so its
pair sum over a lag ``h`` of a uniform grid is a contraction of the increment moment tensor

```math
M_{c_1 ⋯ c_p}(h) = \sum_x m(x) m(x+h) \prod_{k=1}^p \left[u_{c_k}(x+h) - u_{c_k}(x)\right]
 = \sum_{S ⊆ \{1..p\}} (-1)^{p-|S|}\; X\!\left[m\,μ_{S^c},\; m\,μ_S\right](h),
```

a signed sum of cross-correlations ``X[f, g](h) = \sum_x f(x) g(x+h)`` of the **masked monomials**
``m\,μ_S = m \prod_{k∈S} u_{c_k}`` — each a Fourier product — with the counts ``X[m, m]``. This is
exact at every lag that has a pair, for any mask ``m`` and any order, on periodic or zero-padded
bounded directions. A grid with some uniform directions and some enumerated ones (a stretched
axis, or the latitude rows of a lat-lon grid) is handled by enumerating index pairs on the latter and
transforming along the former, with the pair frame applied per lag; on the sphere that frame depends
only on the two latitudes and the longitude offset, so no rotation of the grid is needed. The
tensor itself is available (`calculate_structure_function_tensor` on a grid), and the transform is
compared against the direct lag sweep and the pair loop to round-off on every schedule.

### The sphere by spherical harmonics

For any point set with weights ``w_i`` and mask ``m_i``, the pseudo-coefficients
``\tilde a_{lm} = \sum_i w_i m_i f_i Y_{lm}(x_i)`` of a scalar give, as an identity,

```math
\sum_{l ≤ L} \frac{2l+1}{4π}\, b_l\, \tilde C_l\, P_l(\cos β) = \sum_{ij} w_i w_j m_i m_j f_i f_j\, K_L(γ_{ij}, β),
\qquad K_L(γ, β) = \frac{1}{16π^2} \sum_{l ≤ L} (2l+1)\, b_l\, P_l(\cos γ) P_l(\cos β),
```

the pair sum with a soft kernel of width ``≈ π/L`` in place of a hard bin (`HarmonicNodes`). Vectors
ride the same identity through the spin-1 quantity ``u_θ + i u_φ`` and Wigner ``d^l_{ss'}`` kernels,
and the higher-order operators through the spin-weighted monomials their polynomials expand into. On
a Gauss–Legendre grid with exact quadrature weights and a band-limited field the series terminates
and the route returns the continuous rotation average exactly.

### Points on a line

For one-dimensional coordinates every bin of every point is one index range once the points are
sorted, so the polynomial moments are prefix-sum differences of the monomials: exact, in
``O(N \log N + N\, n_{\mathrm{bins}})`` rather than ``O(N^2)``. The CPU backends take this route for
every polynomial operator.

### Scattered points by non-uniform FFT — soft bins

Scattered points mapped into a padded periodic box and a mode grid give, through type-1 non-uniform
FFTs of the masked monomials, the same transform the gridded engine expects, and the inverse products
are pair sums with the periodic Dirichlet kernel of the mode set in place of a delta at each lag: a
**soft bin** of width about one cell of the mode grid, with sidelobes a `GaussianTaper` trades for
width. This route (`ScatteredModesSchedule`, `ModeBinEdges`) is **not exact**; it converges to the
hard-binned pair sum as the mode count grows, and its results are marked so they cannot be mistaken
for pair counts. It is never selected automatically.

## Fitting instead of inverting

The relations above invert a structure function directly. On sparse or noisy data an estimator with
a stated prior is the alternative: a forward model from values on wavenumber bins to the structure
function at the measured separations — `SpectrumForwardModel` for ``E(k)`` from ``S_2``,
`HelmholtzForwardModel` for ``(E_E, E_B)`` from ``(D_{LL}, D_{TT})``, `FluxForwardModel` for a
piecewise-constant flux ``F(k) = -ε + \sum_j ξ_j Δk_j H(k - k_j)`` from ``S_3``, exact for that
``F`` through ``S_3(r) = 2εr - \sum_j 4\,ξ_j Δk_j J_1(k_j r)/k_j`` — fitted by regularised least
squares with a posterior covariance, by non-negative least squares, or by a continuous segmented
power law fitted on the relative residual. These are estimators; the transforms are the exact
relations they approximate, and the tests hold the two to each other.

## Curved geometry: structure functions on a sphere

Everything above in flat space treats ``\hat r`` as one vector and ``δu`` as a plain difference. On a
sphere neither holds. The two points have *different* tangent planes, so "the separation direction"
is two different vectors, and subtracting velocities that live in different tangent spaces is not
defined.

### The transported frame

The separation is the great-circle arc. With ``\hat p, \hat q`` the unit position vectors,
``c = \hat p · \hat q``, ``\vec w = \hat p × \hat q``, ``s = |\vec w| = \sin σ``:

```math
σ = 2\,\mathrm{atan2}\!\left(|\hat q - \hat p|,\ |\hat q + \hat p|\right), \qquad r = Rσ,
\qquad
\hat t_A = \frac{\hat q - c\,\hat p}{s}, \qquad
\hat t_B = \frac{c\,\hat q - \hat p}{s}, \qquad
\hat m   = \frac{\vec w}{s}.
```

``\hat t_A`` and ``\hat t_B`` are the geodesic tangents at each endpoint and ``\hat m`` the
great-circle normal, the same vector at both endpoints, so the transverse direction needs no
transport. The increments are

```math
δu_L = \vec u_B · \hat t_B - \vec u_A · \hat t_A, \qquad δu_T = (\vec u_B - \vec u_A) · \hat m .
```

Projecting each velocity onto its own geodesic frame before differencing **is** parallel transport:
a geodesic parallel-transports its own tangent, and transport on ``S^2`` is an orientation-preserving
isometry, so the frame carried from ``A`` arrives at ``B`` rotated by exactly the difference of forward
azimuths. In this frame the components of an increment are the same whichever end the pair is read
from, so the odd-rank tensor takes no reading sign on a sphere.

### Why a flat frame is wrong, and by how much

Using one flat direction for both endpoints ignores the meridian convergence, the angle
``ψ ≈ (r/R)\tan φ`` between the two local frames: 0.9° at 100 km and 45° latitude, 9° at 1000 km,
25° at 1000 km and 70°. ``D_{LT}`` vanishes identically under reflection symmetry, so this leaks
``O(ψ)(D_{TT} - D_{LL})/2`` into a quantity whose true value is zero, and the third-order cascade
diagnostics inherit ``O(ψ)``. For a solid-body rotation, which has no strain, the transported frame
gives ``\sum D_{LL} / \sum S_2 ≈ 5×10^{-32}`` while a flat lon/lat frame puts 36 % of the signal into
``D_{LL}``. Below about 10 km a flat tangent plane is fine; beyond about 100 km, or poleward of about
60°, use a spherical metric.

### Conventions and limits

- **Radial component.** For a thin shell (``D = 3``, ``u = (\mathrm{east}, \mathrm{north}, \mathrm{up})``) the
  radial component is differenced as a scalar and never transported: the geodesic frame is tangent to
  the shell, hence orthogonal to ``\hat p`` at both endpoints, so radial motion cannot leak into
  ``δu_L`` or ``δu_T``.
- **Coordinates.** A point on a shell is located by two numbers: `x` is `(2, N)` lon/lat while `u` may
  be `(2, N)` or `(3, N)`; the shell radius belongs to the metric, `SphericalDistance(R)`.
- **Angle units follow the metric.** `Distances.Haversine` is degrees, `Distances.SphericalAngle` and
  `SphericalDistance` radians. Mixing them rescales every separation by about 57.
- **Degenerate pairs are excluded.** Coincident points have no direction, and antipodal points are
  joined by infinitely many great circles. Both are skipped rather than given a `NaN`.
- **Precision.** Sub-metre separations on Earth need `Float64` input: at 45°, one `Float32` ulp of
  latitude is already about 4 m on the ground.
- **Isotropy relations are planar.** The relation ``D_{TT} = \mathrm d(r D_{LL})/\mathrm dr``
  (Lindborg 1999, eq. 53) holds on the sphere only to ``O((r/R)^2)``.

## Kolmogorov theory

At scales ``η ≪ r ≪ L`` between the dissipation scale and the integral scale, Kolmogorov's 1941
theory predicts ``S_2(r) ∼ ε^{2/3} r^{2/3}`` and, more generally, ``S_n(r) ∼ r^{ζ_n}`` with
``ζ_n = n/3``. Real turbulence is intermittent: the measured exponents deviate,
``ζ_n = n/3 + δ_n``, increasingly so for ``n > 3``, which is what the higher-order operators measure.
A prescribed shell spectrum ``E(k) ∼ k^{-(ζ+1)}`` implies ``S_2 ∼ r^{ζ}`` — the two are one statement
about a field, and the [Walkthrough](walkthrough.md) shows both on the same data.

## References

Foundations

- Kolmogorov, A. N. (1941). The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers. *Dokl. Akad. Nauk SSSR*, 30, 301–305.
- Landau, L. D., & Lifshitz, E. M. (1987). *Fluid Mechanics* (2nd ed.). Pergamon.
- Frisch, U. (1995). *Turbulence: The Legacy of A. N. Kolmogorov.* Cambridge University Press.
- She, Z.-S., & Leveque, E. (1994). Universal scaling laws in fully developed turbulence. *Physical Review Letters*, 72, 336.
- Lindborg, E. (1999). Can the atmospheric kinetic energy spectrum be explained by two-dimensional turbulence? *Journal of Fluid Mechanics*, 388, 259–288. [doi:10.1017/S0022112099004851](https://doi.org/10.1017/S0022112099004851)

Fluxes and fits

- Xie, J.-H., & Bühler, O. (2018). Exact third-order structure functions for two-dimensional turbulence. *Journal of Fluid Mechanics*, 851, 672–686.
- Pearson, B., Wagner, G. L., Fox-Kemper, B., & Samelson, R. (2025). Bessel-function estimators of the interscale kinetic energy and enstrophy fluxes from structure functions. *Journal of Physical Oceanography*, 55(9), 1335–1352. [doi:10.1175/JPO-D-24-0211.1](https://doi.org/10.1175/JPO-D-24-0211.1) — the ``J_1``, ``J_2`` and ``J_3`` relations; see the validation page on the printed boundary terms.
- Balwada, D., Xie, J.-H., Marino, R., & Feraco, F. (2022). Direct observational evidence of an oceanic dual kinetic energy cascade and its seasonality. *Science Advances*, 8(41). [arXiv:2202.08637](https://arxiv.org/abs/2202.08637) — the non-negative flux fit.
- Gutierrez-Villanueva, M. O., Cornuelle, B., Gille, S., Mazloff, M., & Balwada, D. (2026). Regularised least-squares estimates of the spectral kinetic energy flux from third-order structure functions. *Journal of Atmospheric and Oceanic Technology*, 43(3), 355–372 — the regularised flux fit and its posterior covariance.
- Bhattacharjee, A., Jones, J., Balwada, D., Elipot, S., & Gutierrez-Villanueva, M. O. (2026). Segmented power-law spectra from second-order structure functions. [arXiv:2604.27200](https://arxiv.org/abs/2604.27200) — the segmented power-law fit.

Grids, masks and scattered points

- Marcotte, D. (1996). Fast variogram computation with FFT. *Computers & Geosciences*, 22(10), 1175–1186. [doi:10.1016/S0098-3004(96)00026-X](https://doi.org/10.1016/S0098-3004(96)00026-X) — the second-order case of the masked-monomial expansion.
- Slepian, Z., & Eisenstein, D. J. (2016). A practical computational method for the anisotropic redshift-space three-point correlation function. *MNRAS*, 455, L31. [arXiv:1506.04746](https://arxiv.org/abs/1506.04746) — binned pair counts as FFT convolutions and the padding remark.
- Campagne, J.-E. (2026). Kernel-weighted correlation of irregularly sampled series by non-uniform FFT. [arXiv:2609.03866](https://arxiv.org/abs/2609.03866) — the soft-binned route's flat-space precedent.
- Szapudi, I., Prunet, S., & Colombi, S. (2001). Fast analysis of inhomogeneous megapixel CMB maps. *ApJ*, 561, L11. Chon, G., Challinor, A., Prunet, S., Colombi, S., & Szapudi, I. (2004). Fast estimation of polarization power spectra using correlation functions. *MNRAS*, 350, 914 — the kernel-binned harmonic statistic and its E/B split.

Curved geometry

- Kamionkowski, M., Kosowsky, A., & Stebbins, A. (1997). Statistics of cosmic microwave background polarization. *Physical Review D*, 55, 7368 — the transported-frame convention for two-point vector statistics on a sphere.
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). *Optimization Algorithms on Matrix Manifolds.* Princeton University Press, Ex. 8.1.1 — closed-form parallel transport on ``S^2``.
- Aluie, H. (2019). Convolutions on the sphere: commutation with differential operators. *GEM — International Journal on Geomathematics*, 10, 9. [doi:10.1007/s13137-019-0123-9](https://doi.org/10.1007/s13137-019-0123-9) — normal and tangent character preserved, which is why the radial component is differenced rather than transported.
- Balwada, D., LaCasce, J. H., & Speer, K. G. (2016). Scale-dependent distribution of kinetic energy from surface drifters in the Gulf of Mexico. *Geophysical Research Letters*, 43. [doi:10.1002/2016GL069405](https://doi.org/10.1002/2016GL069405) — the local-tangent-plane convention of regional studies.
