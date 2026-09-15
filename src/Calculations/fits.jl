# Regularised fits of spectra and fluxes to binned structure functions: linear forward models from
# values on wavenumber bins to the structure function at the result's separations, and the inversions
# that fit them.

# ---------------------------------------------------------------------------------------------------
# Wavenumber quadrature
# ---------------------------------------------------------------------------------------------------

"""
    _bin_nodes(k_edges, r_max) -> (k, w, bin)

Gauss–Legendre nodes and weights over every wavenumber bin, enough per bin to integrate a kernel that
oscillates as `cos(k r_max)` across it, with the bin each node belongs to.
"""
function _bin_nodes(k_edges::AbstractVector, r_max::Real)
    nb = length(k_edges) - 1
    k = Float64[]
    w = Float64[]
    bin = Int[]
    for j in 1:nb
        a, b = Float64(k_edges[j]), Float64(k_edges[j + 1])
        n = 12 + ceil(Int, (b - a) * r_max / 2)
        μ, ω = gauss_legendre(n)
        append!(k, (a + b) / 2 .+ (b - a) / 2 .* μ)
        append!(w, (b - a) / 2 .* ω)
        append!(bin, fill(j, n))
    end
    return k, w, bin
end

function _check_wavenumber_edges(k_edges::AbstractVector)
    length(k_edges) >= 2 || throw(ArgumentError("wavenumber edges need at least one bin"))
    issorted(k_edges) || throw(ArgumentError("wavenumber edges must be sorted"))
    first(k_edges) > 0 || throw(ArgumentError(
        "wavenumber edges must be positive; the k = 0 mode is not recoverable from a structure function",
    ))
    return nothing
end

function _check_separations(r::AbstractVector)
    isempty(r) && throw(ArgumentError("no separation to fit at"))
    issorted(r) || throw(ArgumentError("separations must be sorted"))
    first(r) > 0 || throw(ArgumentError("separations must be positive"))
    return nothing
end

"""The bin each column of a forward matrix stands for: its centre and width."""
_bin_centres(k_edges) = [(k_edges[j] + k_edges[j + 1]) / 2 for j in 1:(length(k_edges) - 1)]
_bin_widths(k_edges) = [k_edges[j + 1] - k_edges[j] for j in 1:(length(k_edges) - 1)]

# ---------------------------------------------------------------------------------------------------
# Forward models
# ---------------------------------------------------------------------------------------------------

"""
    AbstractForwardModel

A linear map `y = H x` from values on wavenumber bins to a structure function at fixed separations.
[`forward_matrix`](@ref) returns `H`; `model.r` the separations, `model.k_edges` the bin edges.
"""
abstract type AbstractForwardModel end

"""The matrix of a forward model."""
forward_matrix(m::AbstractForwardModel) = m.H

"""
    SpectrumForwardModel(::Val{D}, r, k_edges)

The second-order trace at the separations `r` from a shell spectrum `E(k)` constant on each of the
wavenumber bins `k_edges`, in `D` dimensions:

```math
S_2(r_i) = 2 ∫_0^∞ E(k)\\,[1 - \\mathrm{kernel}_D(k r_i)]\\,dk = Σ_j H_{ij} E_j,
\\qquad H_{ij} = 2 ∫_{k_j}^{k_{j+1}} [1 - \\mathrm{kernel}_D(k r_i)]\\,dk ,
```

`kernel_D` the angular average of `cos(k·r)` ([`isotropic_kernel`](@ref); `J₀` in two dimensions,
which needs `Bessels`). `E` integrates over `k` to the variance, the convention of
[`shell_spectrum`](@ref). Each bin is integrated by Gauss–Legendre quadrature.
"""
struct SpectrumForwardModel{D, T, RT <: AbstractVector, KT <: AbstractVector} <: AbstractForwardModel
    r::RT
    k_edges::KT
    H::Matrix{T}
end

function SpectrumForwardModel(::Val{D}, r::AbstractVector, k_edges::AbstractVector) where {D}
    _check_separations(r)
    _check_wavenumber_edges(k_edges)
    k, w, bin = _bin_nodes(k_edges, last(r))
    H = zeros(Float64, length(r), length(k_edges) - 1)
    @inbounds for q in eachindex(k), i in eachindex(r)
        H[i, bin[q]] += 2 * w[q] * (1 - isotropic_kernel(Val(D), k[q] * r[i]))
    end
    return SpectrumForwardModel{D, Float64, typeof(r), typeof(k_edges)}(r, k_edges, H)
end

"""
    HelmholtzForwardModel(r, k_edges)

The longitudinal and transverse second-order structure functions of a two-dimensional field, stacked
as `[D_LL; D_TT]` at the separations `r`, from the gradient (`E`) and curl (`B`) shell spectra stacked
as `[E_E; E_B]` on the bins `k_edges`:

```math
D_{LL} + D_{TT} = 2 ∫ (E_E + E_B)\\,[1 - J_0(kr)]\\,dk, \\qquad
D_{LL} - D_{TT} = 2 ∫ (E_E - E_B)\\, J_2(kr)\\,dk ,
```

the relations [`helmholtz_spectra`](@ref) inverts. Needs `Bessels`.
"""
struct HelmholtzForwardModel{T, RT <: AbstractVector, KT <: AbstractVector} <: AbstractForwardModel
    r::RT
    k_edges::KT
    H::Matrix{T}
end

function HelmholtzForwardModel(r::AbstractVector, k_edges::AbstractVector)
    _check_separations(r)
    _check_wavenumber_edges(k_edges)
    k, w, bin = _bin_nodes(k_edges, last(r))
    nr, nk = length(r), length(k_edges) - 1
    H = zeros(Float64, 2nr, 2nk)
    @inbounds for q in eachindex(k), i in 1:nr
        x = k[q] * r[i]
        plus = w[q] * (1 - bessel_kernel(Val(0), x))
        minus = w[q] * bessel_kernel(Val(2), x)
        j = bin[q]
        H[i, j] += plus + minus
        H[i, nk + j] += plus - minus
        H[nr + i, j] += plus - minus
        H[nr + i, nk + j] += plus + minus
    end
    return HelmholtzForwardModel{Float64, typeof(r), typeof(k_edges)}(r, k_edges, H)
end

"""
    FluxForwardModel(r, k_edges)

The third-order structure function `S3 = ⟨δu_L ‖δu‖²⟩` of a two-dimensional isotropic flow at the
separations `r` from a spectral energy flux that is piecewise constant in wavenumber: `x = (ε, ξ₁, …,
ξ_n)` with `F(k) = -ε + Σ_j ξ_j Δk_j H(k - k_j)`, `k_j` the bin centres and `Δk_j` the bin widths, so
`ε` is the flux towards large scales below the first bin and `ξ_j` the injection density
(flux gained per unit wavenumber) in bin `j`. From `S3(r) = -4r ∫_0^∞ F(k) J_2(kr)/k\\,dk`,

```math
S3(r_i) = 2 ε r_i - Σ_j 4\\,\\frac{ξ_j Δk_j}{k_j} J_1(k_j r_i) ,
```

exactly for that `F`. `F` at the bin centres, just past each jump, is `G x` with
`G[i, 1] = -1`, `G[i, j + 1] = Δk_j` for `j ≤ i`; [`flux_matrix`](@ref) returns `G`. The sign is
that of [`spectral_flux`](@ref): positive towards small scales. Needs `Bessels`.
"""
struct FluxForwardModel{T, RT <: AbstractVector, KT <: AbstractVector} <: AbstractForwardModel
    r::RT
    k_edges::KT
    H::Matrix{T}
    G::Matrix{T}
end

function FluxForwardModel(r::AbstractVector, k_edges::AbstractVector)
    _check_separations(r)
    _check_wavenumber_edges(k_edges)
    kc = _bin_centres(k_edges)
    dk = _bin_widths(k_edges)
    nr, nk = length(r), length(kc)
    H = zeros(Float64, nr, nk + 1)
    @inbounds for i in 1:nr
        H[i, 1] = 2 * r[i]
        for j in 1:nk
            H[i, j + 1] = -4 * dk[j] / kc[j] * bessel_kernel(Val(1), kc[j] * r[i])
        end
    end
    G = zeros(Float64, nk, nk + 1)
    @inbounds for i in 1:nk
        G[i, 1] = -1
        for j in 1:i
            G[i, j + 1] = dk[j]
        end
    end
    return FluxForwardModel{Float64, typeof(r), typeof(k_edges)}(r, k_edges, H, G)
end

"""The matrix taking a flux model's `(ε, ξ…)` to the flux at the bin centres."""
flux_matrix(m::FluxForwardModel) = m.G

# ---------------------------------------------------------------------------------------------------
# Data covariance
# ---------------------------------------------------------------------------------------------------

"""
    _whitened(H, y, W) -> (Hw, yw)

`W^{-1/2} H` and `W^{-1/2} y` for a data covariance `W` given as the variances of `y` (a vector), a
full matrix, or `nothing` for the identity.
"""
_whitened(H::AbstractMatrix, y::AbstractVector, ::Nothing) = (Matrix{Float64}(H), Vector{Float64}(y))

function _whitened(H::AbstractMatrix, y::AbstractVector, W::AbstractVector)
    length(W) == length(y) || throw(DimensionMismatch(
        "the data covariance names $(length(W)) variances for $(length(y)) values",
    ))
    all(>(0), W) || throw(ArgumentError("every data variance must be positive"))
    s = 1 ./ sqrt.(W)
    return H .* s, y .* s
end

function _whitened(H::AbstractMatrix, y::AbstractVector, W::AbstractMatrix)
    size(W) == (length(y), length(y)) || throw(DimensionMismatch(
        "the data covariance is $(size(W)) for $(length(y)) values",
    ))
    L = LA.cholesky(LA.Symmetric(Matrix{Float64}(W))).L
    return L \ Matrix{Float64}(H), L \ Vector{Float64}(y)
end

"""
    independent_pair_variance(joint::StructureFunction2DSumsAndCounts) -> Vector

The variance of each distance bin's mean pair value under the assumption that the pairs are
independent: the variance of the pair values in the bin over the bin's count. Read from a
value-binned joint histogram (`InvariantValueAxis`), each value cell contributing its own mean, so
the spread inside a value cell is not seen and the result is a lower bound. A bin holding no pair is
`NaN`. This is the data covariance `W` the fits take.
"""
function independent_pair_variance(joint::SFO.StructureFunction2DSumsAndCounts)
    nb = size(joint.sums, 1)
    out = Vector{Float64}(undef, nb)
    @inbounds for b in 1:nb
        n = 0.0
        s1 = 0.0
        s2 = 0.0
        for v in axes(joint.sums, 2)
            c = float(joint.counts[b, v])
            c > 0 || continue
            m = joint.sums[b, v] / c
            n += c
            s1 += c * m
            s2 += c * m * m
        end
        out[b] = n > 0 ? max(s2 / n - (s1 / n)^2, 0.0) / n : NaN
    end
    return out
end

# ---------------------------------------------------------------------------------------------------
# Inversions
# ---------------------------------------------------------------------------------------------------

"""
    AbstractFitMethod

How values on wavenumber bins are fitted to a structure function through a forward model.
"""
abstract type AbstractFitMethod end

"""
    RegularizedLeastSquares(prior)

Regularised least squares with a Gaussian prior on the fitted values: for `y = H x + e`, `e` of
covariance `W` and a prior covariance `P`,

```math
\\hat x = (Hᵀ W^{-1} H + P^{-1})^{-1} Hᵀ W^{-1} y, \\qquad C_{xx} = (Hᵀ W^{-1} H + P^{-1})^{-1} .
```

`prior` is the prior variance of each value (a vector), a full prior covariance matrix, or `nothing`
for no prior, which is ordinary weighted least squares. The data covariance `W` is required by the
fits taking this method; [`independent_pair_variance`](@ref) supplies one from a joint histogram, and
[`tradeoff_curve`](@ref) shows how the misfit and the norm of `x̂` trade against the prior.
"""
struct RegularizedLeastSquares{P} <: AbstractFitMethod
    prior::P
    function RegularizedLeastSquares(prior::P) where {P <: Union{Nothing, AbstractVector, AbstractMatrix}}
        prior isa AbstractVector && !all(>(0), prior) && throw(ArgumentError(
            "every prior variance must be positive",
        ))
        return new{P}(prior)
    end
end

"""
    NonNegativeLeastSquares()

Least squares with every fitted value constrained non-negative, by the Lawson–Hanson active-set
algorithm on the whitened system. For a flux that is the monotone-flux model, which cannot resolve a
sink; for a spectrum it is the constraint that a spectral density is non-negative. It returns no
covariance.
"""
struct NonNegativeLeastSquares <: AbstractFitMethod end

"""
    SegmentedPowerLaw(segments; slope_bounds = (-4.0, 1.0))

A shell spectrum that is a power law on each of `segments` wavenumber ranges, log-uniform between the
smallest and the largest wavenumber asked for, continuous at the breakpoints, with slopes held in
`slope_bounds` and a non-negative amplitude: `S + 1` parameters `(b₁, α₁, …, α_S)`. Fitted by bounded
Levenberg–Marquardt through `LsqFit` (`using LsqFit`), on the relative residual
`(S₂^{fit} - S₂)/S₂` when no data covariance is given.
"""
struct SegmentedPowerLaw <: AbstractFitMethod
    segments::Int
    slope_bounds::Tuple{Float64, Float64}
    function SegmentedPowerLaw(segments::Integer; slope_bounds = (-4.0, 1.0))
        segments >= 1 || throw(ArgumentError("a segmented power law needs at least one segment"))
        lo, hi = Float64(slope_bounds[1]), Float64(slope_bounds[2])
        lo < hi || throw(ArgumentError("the slope bounds must be ordered; got $slope_bounds"))
        return new(Int(segments), (lo, hi))
    end
end

"""`P⁻¹` of a prior given as variances, a matrix, or nothing (zero)."""
_prior_precision(::Nothing, n::Int) = zeros(Float64, n, n)
function _prior_precision(P::AbstractVector, n::Int)
    length(P) == n || throw(DimensionMismatch("the prior names $(length(P)) variances for $n values"))
    return LA.Diagonal(1 ./ Vector{Float64}(P))
end
function _prior_precision(P::AbstractMatrix, n::Int)
    size(P) == (n, n) || throw(DimensionMismatch("the prior covariance is $(size(P)) for $n values"))
    return inv(LA.Symmetric(Matrix{Float64}(P)))
end

"""
    solve(method, H, y, W) -> (x, covariance)

Fit `y = H x` by `method` under the data covariance `W`. [`RegularizedLeastSquares`](@ref) returns the
posterior covariance; [`NonNegativeLeastSquares`](@ref) returns `nothing` for it.
"""
function solve(m::RegularizedLeastSquares, H::AbstractMatrix, y::AbstractVector, W)
    W === nothing && throw(ArgumentError(
        "regularised least squares weighs the data by their covariance, which was not given; pass " *
        "`W`, the variance of each structure-function value — `independent_pair_variance` reads one " *
        "from a value-binned joint histogram — or a full covariance matrix",
    ))
    Hw, yw = _whitened(H, y, W)
    n = size(H, 2)
    A = LA.Symmetric(Hw' * Hw + _prior_precision(m.prior, n))
    C = inv(A)
    x = C * (Hw' * yw)
    return x, LA.Symmetric(C)
end

function solve(::NonNegativeLeastSquares, H::AbstractMatrix, y::AbstractVector, W)
    Hw, yw = _whitened(H, y, W)
    return _nnls(Hw, yw), nothing
end

"""
    _nnls(A, b) -> x

`argmin ‖A x − b‖₂` over `x ≥ 0` by the Lawson–Hanson active-set algorithm.
"""
function _nnls(A::AbstractMatrix, b::AbstractVector; maxiter::Int = 10 * size(A, 2))
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("A is $(size(A)) and b has $(length(b)) entries"))
    x = zeros(Float64, n)
    passive = falses(n)
    tol = 10 * eps(Float64) * max(LA.opnorm(A, 1), 1.0) * max(m, n)
    w = A' * (b - A * x)
    iter = 0
    while !all(passive) && maximum(w[.!passive]; init = -Inf) > tol && iter < maxiter
        iter += 1
        j = argmax(ifelse.(passive, -Inf, w))
        passive[j] = true
        s = zeros(Float64, n)
        s[passive] = A[:, passive] \ b
        while minimum(s[passive]; init = Inf) <= 0
            α = Inf
            for i in 1:n
                if passive[i] && s[i] <= 0
                    α = min(α, x[i] / (x[i] - s[i]))
                end
            end
            x .+= α .* (s .- x)
            for i in 1:n
                passive[i] && x[i] <= tol && (passive[i] = false; x[i] = 0)
            end
            s .= 0
            any(passive) && (s[passive] = A[:, passive] \ b)
            any(passive) || break
        end
        x .= s
        w = A' * (b - A * x)
    end
    return x
end

# ---------------------------------------------------------------------------------------------------
# The segmented power law
# ---------------------------------------------------------------------------------------------------

"""Log-uniform breakpoints, `S + 1` of them from `k_lo` to `k_hi`."""
_segment_edges(k_lo::Real, k_hi::Real, S::Int) = exp.(range(log(k_lo), log(k_hi); length = S + 1))

"""
    segmented_spectrum(p, k, edges) -> E(k)

The continuous piecewise power law with parameters `p = (b₁, α₁, …, α_S)` on the segments `edges`:
`b_s k^{α_s}` on segment `s`, `b_{s+1} = b_s k_s^{α_s - α_{s+1}}` at each breakpoint. `k` outside the
segments takes the nearest end segment's law.
"""
function segmented_spectrum(p::AbstractVector, k::AbstractVector, edges::AbstractVector)
    S = length(edges) - 1
    length(p) == S + 1 || throw(DimensionMismatch("$S segments take $(S + 1) parameters; got $(length(p))"))
    T = promote_type(eltype(p), eltype(k))
    amps = Vector{T}(undef, S)
    amps[1] = p[1]
    for s in 1:(S - 1)
        amps[s + 1] = amps[s] * T(edges[s + 1])^(p[s + 1] - p[s + 2])
    end
    return [begin
        s = clamp(searchsortedlast(edges, kq), 1, S)
        amps[s] * T(kq)^p[s + 1]
    end for kq in k]
end

"""
    _segmented_fit(method::SegmentedPowerLaw, ::Val{D}, r, y, W, k_lo, k_hi)
        -> (p, covariance, edges, converged)

The bounded Levenberg–Marquardt fit of the segmented power law; supplied by the LsqFit extension.
"""
function _segmented_fit(::SegmentedPowerLaw, ::Val, r, y, W, k_lo, k_hi)
    throw(ArgumentError("fitting a segmented power law needs LsqFit: run `using LsqFit`"))
end

"""
    _segmented_design(::Val{D}, r, edges) -> (k, w, K)

Quadrature nodes and weights over the segments — Gauss–Legendre on log-uniform panels of at most a
quarter octave, where a power law is close to linear — and the kernel matrix
`K[i, q] = 2 w_q [1 - kernel_D(k_q r_i)]`, so the model is `K * E(p)(k)` with `E` the spectrum at the
nodes.
"""
function _segmented_design(::Val{D}, r::AbstractVector, edges::AbstractVector) where {D}
    panels = Float64[]
    for s in 1:(length(edges) - 1)
        n = max(1, ceil(Int, 4 * log2(edges[s + 1] / edges[s])))
        append!(panels, exp.(range(log(edges[s]), log(edges[s + 1]); length = n + 1))[1:end - 1])
    end
    push!(panels, Float64(last(edges)))
    k, w, _ = _bin_nodes(panels, last(r))
    K = Matrix{Float64}(undef, length(r), length(k))
    @inbounds for q in eachindex(k), i in eachindex(r)
        K[i, q] = 2 * w[q] * (1 - isotropic_kernel(Val(D), k[q] * r[i]))
    end
    return k, w, K
end

# ---------------------------------------------------------------------------------------------------
# Fits of result objects
# ---------------------------------------------------------------------------------------------------

"""The separations and values a result holds, restricted to the bins with a value."""
function _fit_samples(sf::SFO.AbstractStructureFunction)
    r, vals, keep = _binned(sf)
    isempty(keep) && throw(ArgumentError("no structure function value to fit"))
    return r[keep], vals[keep], keep
end

"""The data covariance over the bins holding a value, from one given over those bins or over all `n_all` bins."""
_keep_variances(::Nothing, keep, n_all) = nothing
function _keep_variances(W::AbstractVector, keep, n_all)
    length(W) == length(keep) && return Vector{Float64}(W)
    length(W) == n_all && return Vector{Float64}(W[keep])
    throw(DimensionMismatch(
        "the data covariance names $(length(W)) variances; the result holds $(length(keep)) values in $n_all bins",
    ))
end
function _keep_variances(W::AbstractMatrix, keep, n_all)
    size(W, 1) == size(W, 2) || throw(DimensionMismatch("a data covariance matrix must be square; got $(size(W))"))
    size(W, 1) == length(keep) && return Matrix{Float64}(W)
    size(W, 1) == n_all && return Matrix{Float64}(W[keep, keep])
    throw(DimensionMismatch(
        "the data covariance is $(size(W)); the result holds $(length(keep)) values in $n_all bins",
    ))
end

"""
    fit_spectrum(sf, k_edges, method, ::Val{D}; W = nothing) -> (k, E, covariance, …)

The shell spectrum on the wavenumber bins `k_edges` that reproduces the second-order trace `sf`
through [`SpectrumForwardModel`](@ref), fitted by `method`. `W` is the data covariance: the variance
of each bin's value (a vector over the bins holding a value, or over all bins), or a full matrix.
Returns the bin centres `k`, the fitted `E` on the bins and the posterior `covariance` (`nothing` for
a method that gives none). A [`SegmentedPowerLaw`](@ref) returns `E` evaluated at `k`, and in
addition the fitted `parameters`, their `covariance`, the segment `breakpoints` and `converged`.
"""
function fit_spectrum(sf::SFO.AbstractStructureFunction, k_edges::AbstractVector, method::AbstractFitMethod,
                      ::Val{D}; W = nothing) where {D}
    assert_invertible(sf.operator)
    r, y, keep = _fit_samples(sf)
    return _fit_spectrum(method, Val(D), r, y, k_edges, _keep_variances(W, keep, _value_count(sf)))
end

function _fit_spectrum(method::AbstractFitMethod, ::Val{D}, r, y, k_edges, W) where {D}
    model = SpectrumForwardModel(Val(D), r, k_edges)
    x, C = solve(method, model.H, y, W)
    return (k = _bin_centres(k_edges), E = x, covariance = C)
end

function _fit_spectrum(method::SegmentedPowerLaw, ::Val{D}, r, y, k_edges, W) where {D}
    _check_wavenumber_edges(k_edges)
    p, C, edges, converged = _segmented_fit(method, Val(D), r, y, W, first(k_edges), last(k_edges))
    k = _bin_centres(k_edges)
    return (k = k, E = segmented_spectrum(p, k, edges), covariance = C, parameters = p,
            breakpoints = edges, converged = converged)
end

"""
    fit_helmholtz_spectra(L2, T2, k_edges, method; W = nothing) -> (k, E, B, covariance)

The gradient (`E`) and curl (`B`) shell spectra on the bins `k_edges` that reproduce the longitudinal
and transverse second-order structure functions of a two-dimensional field through
[`HelmholtzForwardModel`](@ref). `L2` and `T2` share their bins; `W` covers the stacked values
`[D_LL; D_TT]` over the bins both hold (or over all bins of each). The `covariance` is over the
stacked `[E; B]`.
"""
function fit_helmholtz_spectra(L2::SFO.AbstractStructureFunction, T2::SFO.AbstractStructureFunction,
                               k_edges::AbstractVector, method::AbstractFitMethod; W = nothing)
    _assert_projection(L2.operator, 2, 0, "longitudinal")
    _assert_projection(T2.operator, 0, 2, "transverse")
    r, dll, dtt = _paired_bins(L2, T2)
    model = HelmholtzForwardModel(r, k_edges)
    y = vcat(dll, dtt)
    Wk = _stacked_variances(W, L2, T2, r)
    x, C = solve(method, model.H, y, Wk)
    nk = length(k_edges) - 1
    return (k = _bin_centres(k_edges), E = x[1:nk], B = x[(nk + 1):end], covariance = C)
end

_stacked_variances(::Nothing, L2, T2, r) = nothing
function _stacked_variances(W::AbstractVector, L2, T2, r)
    n = length(r)
    length(W) == 2n && return Vector{Float64}(W)
    _, _, ka = _binned(L2)
    _, _, kb = _binned(T2)
    keep = intersect(ka, kb)
    nb = _value_count(L2)
    length(W) == 2nb || throw(DimensionMismatch(
        "the data covariance must cover the $(2n) stacked values the two results share, or the $(2nb) of every bin",
    ))
    return vcat(Float64.(W[keep]), Float64.(W[nb .+ keep]))
end
_stacked_variances(W::AbstractMatrix, L2, T2, r) = (size(W, 1) == 2length(r) || throw(DimensionMismatch(
    "a full data covariance must cover the $(2length(r)) stacked values the two results share",
)); Matrix{Float64}(W))

_value_count(sf::SFO.StructureFunction) = length(sf.values)
_value_count(sf::SFO.StructureFunctionSumsAndCounts) = length(sf.sums)

"""
    fit_flux(sf, k_edges, method; W = nothing) -> (k, F, ξ, ε, covariance, flux_covariance)

The piecewise-constant spectral energy flux that reproduces the third-order structure function
`sf = ⟨δu_L ‖δu‖²⟩` (`S3SFType`) of a two-dimensional isotropic flow through [`FluxForwardModel`](@ref).
`F` is the flux at the bin centres `k` (positive towards small scales), `ξ` the injection density on
the bins and `ε` the flux towards large scales below the first bin; `covariance` is over `(ε, ξ…)`
and `flux_covariance = G C Gᵀ` over `F`.
"""
function fit_flux(sf::SFO.AbstractStructureFunction, k_edges::AbstractVector, method::AbstractFitMethod;
                  W = nothing)
    sf.operator isa SFT.S3SFType || throw(ArgumentError(
        "the flux is fitted to ⟨δu_L ‖δu‖²⟩ (S3SFType); got $(nameof(typeof(sf.operator)))",
    ))
    r, y, keep = _fit_samples(sf)
    model = FluxForwardModel(r, k_edges)
    x, C = solve(method, model.H, y, _keep_variances(W, keep, _value_count(sf)))
    F = model.G * x
    CF = C === nothing ? nothing : LA.Symmetric(model.G * C * model.G')
    return (k = _bin_centres(k_edges), F = F, ξ = x[2:end], ε = x[1], covariance = C, flux_covariance = CF)
end

"""
    tradeoff_curve(model, y, W, priors) -> (misfit, norm)

For each prior in `priors` (each a prior variance shared by every value, or a vector of them), the
regularised least-squares fit of `y` through `model` under the data covariance `W`, reporting the
`W`-normalised misfit `‖W^{-1/2}(H x̂ - y)‖²` and `‖x̂‖₂`. The curve is returned for the caller to
choose from; nothing is chosen.
"""
function tradeoff_curve(model::AbstractForwardModel, y::AbstractVector, W, priors::AbstractVector)
    H = forward_matrix(model)
    n = size(H, 2)
    Hw, yw = _whitened(H, y, W)
    misfit = Vector{Float64}(undef, length(priors))
    norm = Vector{Float64}(undef, length(priors))
    for (i, p) in enumerate(priors)
        prior = p isa AbstractVector ? p : fill(Float64(p), n)
        x, _ = solve(RegularizedLeastSquares(prior), H, y, W)
        misfit[i] = sum(abs2, Hw * x - yw)
        norm[i] = LA.norm(x)
    end
    return (misfit = misfit, norm = norm)
end

"""
    select_segments(sf, k_edges, S_range, ::Val{D}; W = nothing, slope_bounds = (-4.0, 1.0))
        -> (segments, misfits, fits)

Fit a [`SegmentedPowerLaw`](@ref) with each number of segments in `S_range` and report the one
minimising the mean relative misfit `mean(|S₂^{fit} - S₂| / S₂)`, with every misfit and every fit.
"""
function select_segments(sf::SFO.AbstractStructureFunction, k_edges::AbstractVector, S_range, ::Val{D};
                         W = nothing, slope_bounds = (-4.0, 1.0)) where {D}
    assert_invertible(sf.operator)
    r, y, keep = _fit_samples(sf)
    Wk = _keep_variances(W, keep, _value_count(sf))
    fits = [_fit_spectrum(SegmentedPowerLaw(S; slope_bounds), Val(D), r, y, k_edges, Wk) for S in S_range]
    misfits = [begin
        kq, _, K = _segmented_design(Val(D), r, f.breakpoints)
        yfit = K * segmented_spectrum(f.parameters, kq, f.breakpoints)
        sum(abs.(yfit .- y) ./ abs.(y)) / length(y)
    end for f in fits]
    best = argmin(misfits)
    return (segments = collect(S_range)[best], misfits = misfits, fits = fits)
end
