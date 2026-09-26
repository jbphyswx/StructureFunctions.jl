# Spectra, fluxes, and fitting

A structure-function transform requires a stated relationship between the sampled statistic and the target spectrum or flux. Dimensionality, isotropy, finite separation coverage, and normalization affect that relationship.

## Spectrum calculations

`isotropic_spectrum(result, k, Val(D))` transforms an isotropic second-order trace in dimension `D`. The two-dimensional kernels require `Bessels`. `shell_spectrum` converts a spectral density into a shell-integrated representation; the integration measure must match the chosen dimension and wavenumber convention.

`gridded_spectrum` uses correlations resolved over grid lags. Boundary conditions and tapering affect the available correlations. Mask normalization recovers a mean over observed pairs, whose relation to an unobserved field depends on the sampling process.

`helmholtz_spectra` separates rotational and divergent contributions using longitudinal and transverse second-order functions. Spherical harmonic calculations instead use degree-indexed coefficients and the corresponding spherical kernels.

## Flux calculations

`spectral_flux` supports the package's implemented advective and third-order relations. `S3SFType()` computes `δu_L‖δu‖²`; `L3SFType()` computes `δu_L³`. Select the relation appropriate to the input operator. `enstrophy_flux` uses the corresponding vorticity statistic.

Finite integration limits and boundary terms are part of these estimators. They must be included when interpreting a fitted or transformed flux. The API reference states each convention.

## Fit a forward model

The fitting interfaces compare measured structure functions with a forward model on chosen wavenumber bins:

- `SpectrumForwardModel` models a trace spectrum.
- `HelmholtzForwardModel` models rotational and divergent spectra jointly.
- `FluxForwardModel` models transfer and integrated flux.

`fit_spectrum`, `fit_helmholtz_spectra`, and `fit_flux` select the appropriate model. `RegularizedLeastSquares` uses supplied data covariance and a prior covariance. `NonNegativeLeastSquares` constrains fitted coefficients. `SegmentedPowerLaw` fits a continuous piecewise power law through the optional `LsqFit` extension.

Supply covariance estimates consistent with the estimator and sampling process. Pairs that share observations are generally dependent. A variance inferred from a value histogram also depends on the resolution of its value bins; it is not a replacement for a sampling model.

`tradeoff_curve` reports fit residuals and solution norms across prior strengths. `select_segments` selects the smallest observed residual among the requested segment counts; it does not apply a complexity penalty.

## Validation and production examples

Small analytic checks belong in the test suite. Full synthetic-field experiments, fitting sweeps, and figure generation run separately from the documentation build. See [Validation](validation.md) for the available checks and their assumptions.
