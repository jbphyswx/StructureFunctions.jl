# Validation

Every calculation in this package is checked against something that knows the answer independently.
This page lists those oracles, strongest first, and says what each one can and cannot catch.

The ordering matters. An oracle that is exact to round-off will catch a defect that a
percent-level physical check never notices, and most of the defects found in this package's history
were of exactly that kind — a plausible number that was wrong by a constant factor.

## The oracles

| # | Oracle | Agreement | What it gates | Where |
|---|---|---|---|---|
| 1 | Closed-form Fourier modes | round-off | binning, operators, lag enumeration | `test/test_known_truth.jl` |
| 2 | Transform vs direct sweep | round-off | the two gridded algorithms against each other | `test/test_gridded_fft.jl` |
| 3 | Gridded vs unstructured pair loop | exact counts | the whole gridded path against the reference loop | `test/test_gridded.jl` |
| 4 | Culled vs uncalled | exact counts | culling changes cost, never results | `test/test_cpu_pair_blocking.jl` |
| 5 | Backend agreement | round-off | serial, threaded, distributed, MPI, GPU | `test/test_parallel_equivalence.jl` |
| 6 | Unit invariance | round-off | quantities that may not depend on a choice of unit | `test/test_known_truth.jl` |
| 7 | Frame invariance | round-off | the tensor trace against the second-order SF | `test/test_known_truth.jl` |
| 8 | Analytic zeros | machine zero | solid-body rotation, angular folding | `test/test_spherical_geometry.jl` |
| 9 | Exact-law inversion | round-off | the inertial-range laws recover a prescribed constant | `test/test_known_truth.jl` |
| 10 | Analytic spectrum | round-off | the isotropic transform against a closed form | `test/test_transforms.jl` |
| 11 | Gridded transform vs the field's own spectrum | round-off | the lag-space spectral route | `test/test_transforms.jl` |
| 12 | Closed-form quadrature | round-off | the flux relation's kernel and prefactor | `test/test_transforms.jl` |
| 13 | Linearity of the Helmholtz split | round-off | rotational + divergent spectra sum to the trace's | `test/test_transforms.jl` |

### 1. Closed-form Fourier modes

For a single mode ``u(x) = A\,\hat{e}\cos(k\cdot x + \varphi)`` averaged over a full period,

```math
D_{ab}(r) = A^2\, \hat{e}_a \hat{e}_b \left[1 - \cos(k\cdot r)\right],
```

and every odd-order moment vanishes identically. On a periodic grid whose cell count is a multiple
of the mode's period the average over cells *is* the average over a full period, so this holds to
round-off rather than to a sampling tolerance.

Distinct grid harmonics are orthogonal over the cells, so a superposition's cross terms cancel
exactly and the single-mode form simply adds. Giving each mode two orthonormal polarisations
perpendicular to its wavevector makes the field divergence-free and the polarisation sum the
transverse projector, so ``\sum_p (\hat{e}_p\cdot\hat{r})^2 = 1 - (\hat{k}\cdot\hat{r})^2``. That
makes a *prescribed spectrum* recoverable mode by mode, and the package reproduces it to a relative
`1e-12`.

One subtlety is worth stating, because it is easy to get wrong when building such a test field: the
two polarisations of a single mode share a wavevector, so they are **not** orthogonal over the
cells. Their cross term is traceless — it leaves the trace exact — but it does project onto
``\hat{r}``, so it corrupts the longitudinal component. Offsetting their phases by a quarter turn
removes it exactly.

### 2. Transform against direct sweep

On a uniform grid the structure function of any integer order is computable exactly by transform,
because the binomial expansion turns the increment moment into a sum of cross-correlations and the
correlation theorem evaluates all lags at once. That gives two independent algorithms for one
definition, and they must agree to round-off on the same data.

They are genuinely independent: the sweep visits each lag and reduces over cells, while the
transform never forms a lag at all until after the inverse transform. A defect in either shows up
immediately as disagreement.

Bounded directions are zero-padded to at least ``2n-1`` so that the circular correlation equals the
linear one; periodic directions are not padded, because there the circular correlation is exactly
the sum wanted. Counts are computed as exact integer arithmetic rather than by rounding an inverse
transform.

### 3. Gridded against the unstructured pair loop

The pair loop is the reference implementation: it enumerates pairs, computes each separation, and
bins it. On a bounded grid the separations are plain Euclidean, so the pair loop over the same
points must give **exactly equal counts** and matching sums.

There is one systematic difference worth knowing about, and the gridded path has the better
behaviour. All pairs sharing a lag have one exact separation, so the sweep bins them identically,
while the pair loop recomputes each pair's coordinate difference and those differ by an ulp. When a
bin edge falls *on* an achievable separation the two disagree about the whole shell. The tests place
edges between achievable separations so the comparison is unambiguous.

### 6. Unit invariance

A physical quantity may not change when the unit of length changes. This is a sharper test than it
sounds: the Helmholtz decomposition once multiplied a cumulative integral by ``r``, which left the
energy identity ``D_{rot} + D_{div} = D_{LL} + D_{TT}`` passing to `4e-16` — because the spurious
terms cancel in that sum — while producing a 33 % divergent signal on a field with no divergent
component, and moving that signal by a factor of 2600 under a change from metres to millimetres.

The lesson generalises: a conservation identity that the defect preserves is not a gate. Assert the
invariance the defect actually breaks.

### 7. Frame invariance

The trace of the second-order tensor is the second-order structure function, in any frame. On a
curved manifold that is only true if the tensor's components are transported into a common frame
per pair, so `trace(D_ab) == S2SF` is a direct test of the transport. It is silent on a flat
geometry, where a raw coordinate difference and a transported increment coincide — which is why the
test asserts it on a sphere as well as on a plane.

### 10–13. The transforms

![Spectra from structure functions](assets/sf_spectra.png)

Each transform is checked against something with a closed form, because a transform that is wrong by
a constant returns a curve of exactly the right shape.

**The isotropic transform against an analytic spectrum.** A Gaussian correlation
``C(r) = σ^2 e^{-r^2/2\ell^2}`` gives ``S_2(r) = 2σ^2[1 - e^{-r^2/2\ell^2}]`` and the closed-form
density ``σ^2 \ell^D e^{-k^2\ell^2/2} / (2π)^{D/2}``. Because it *decays*, the transform is not
truncation-limited and the comparison is pointwise: agreement to `6.6e-05`, `2.3e-09` and `9.9e-14`
in one, two and three dimensions. A wrong kernel, a wrong solid angle, a wrong `(2π)^D` or a wrong
sign would each break that, so one comparison covers all four.

A discrete spectral line will *not* do for this. Its correlation does not decay, so the truncated
transform is a sinc whose sidelobes fall off like `1/k` and are cut off by any finite range — worth
1.9 % in one dimension, and refining the wavenumber grid does not move it. Lines are used only to
check that peaks land on the right wavenumbers.

**The gridded transform against the field's own spectrum.** Over the whole lag space nothing is
angularly averaged and nothing is radially binned, so this must agree with transforming the field
directly, and it does: `6.4e-17`, `3.0e-16`, `1.3e-16`. That is the standard the isotropic route
cannot meet on a grid, where the lattice's separations are biased toward its axes.

**The flux relation against a closed-form integral.** ``\int_0^R J_1(Kr)dr = (1 - J_0(KR))/K``, so a
constant advective structure function `c` must give ``Π_K = -(c/2)(1 - J_0(KR))`` exactly. That pins
the kernel and the prefactor with nothing left to fit — which matters, because a flux wrong by a
constant, or by a sign, still looks like a cascade.

**The Helmholtz split by linearity.** ``D_{rot} + D_{div} = D_{LL} + D_{TT}`` exactly and the
transform is linear, so the rotational and divergent spectra must sum to the spectrum of the trace,
whatever the field is. This is an identity rather than a physical expectation, which is what makes it
a gate.

### A note on positive-definiteness

A covariance matrix must be positive semi-definite, and one assembled from a sampled covariance
function need not be — **interpolating a positive-definite kernel does not preserve
positive-definiteness**. The error falls as the square of the separation spacing:

| separations over [0, 6] | spacing | most negative eigenvalue |
|---|---|---|
| 60 | 0.102 | −1.2e−03 |
| 1 000 | 0.006 | −3.8e−07 |
| 5 000 | 0.0012 | +8.0e−08 |
| the kernel itself, uninterpolated | — | +1.6e−09 |
| an oscillating function, which is no kernel at all | — | **−9.7** |

So two quite different failures land in the same place, and the check distinguishes them: a
covariance function that is invalid is off by the scale of the matrix itself, while one that is valid
but under-resolved is off by a discretisation error. Tripping the check on a coarse covariance is
useful output — it says the representation cannot support a valid matrix — not a nuisance to be
tuned away.

## What is checked statistically, and why it is not a gate

The isotropic relation

```math
D_{LL}(r) = \int_0^\infty E(k)\, f(kr)\,dk, \qquad
f(x) = 4\left[\tfrac{1}{3} - \frac{\sin x - x\cos x}{x^3}\right]
```

holds for a three-dimensional isotropic field. The kernel itself is confirmed numerically here: the
transverse-projector direction average equals ``\tfrac{1}{2} f(k_0 r)`` at every ``r`` and every
shell radius, to within the spherical quadrature's own error of about `4e-6`.

The relation is nonetheless **not** used as a test assertion, because neither available construction
makes it tight:

- A field built from cubic-grid harmonics in a thin shell is 0.4–5 % anisotropic, and that does not
  improve with shell radius — a shell of 3722 modes still shows 2 %.
- A field built from spherical-quadrature directions evaluated on scattered points converges only as
  the sampling error allows: 5–8 % at 16 000 points.

In both cases the residual is a property of the mode set and the sampling, not of this package, so
an assertion at that tolerance would mostly be testing the test. The exact multi-mode oracle in
section 1 covers the same code path to round-off and is used instead.

The same reasoning applies to the inertial-range laws. A synthetic Gaussian field carries no energy
flux, so its third-order moments are consistent with zero and the four-fifths law has nothing to
recover. What *is* tested is the inversion itself: given a moment that obeys a law exactly, the
corresponding routine returns the prescribed constant, and the routines are not interchangeable —
applying the four-fifths law to the scalar moment is off by exactly 5/3, which is asserted so that
the two cannot be confused.

## Reproducing

Each oracle lives in a targeted test file that can be run on its own:

```julia
julia --project=test test/test_known_truth.jl
julia --project=test test/test_gridded_fft.jl
julia --project=test test/test_spherical_geometry.jl
```
