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
    all(isfinite, k_edges) && all(>(0), diff(k_edges)) ||
        throw(ArgumentError("wavenumber edges must be finite and strictly increasing"))
    first(k_edges) > 0 || throw(ArgumentError(
        "wavenumber edges must be positive; the k = 0 mode is not recoverable from a structure function",
    ))
    return nothing
end

function _check_separations(r::AbstractVector)
    isempty(r) && throw(ArgumentError("no separation to fit at"))
    all(isfinite, r) && all(>(0), diff(r)) ||
        throw(ArgumentError("separations must be finite and strictly increasing"))
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

function _check_fit_data(H, y)
    size(H, 1) == length(y) || throw(DimensionMismatch("H rows must match the number of observations"))
    size(H, 1) > 0 && size(H, 2) > 0 || throw(ArgumentError("the fit needs observations and parameters"))
    all(isfinite, H) && all(isfinite, y) || throw(ArgumentError("fit data must be finite"))
    return nothing
end

function _covariance_factor(W, label)
    size(W, 1) == size(W, 2) || throw(DimensionMismatch("$label must be square"))
    all(isfinite, W) && LA.issymmetric(W) ||
        throw(ArgumentError("$label must be finite and symmetric"))
    factor = LA.cholesky(LA.Symmetric(Matrix{Float64}(W)); check=false)
    LA.issuccess(factor) || throw(ArgumentError("$label must be positive definite"))
    return factor
end

"""
    _whitening(W, n) -> whiten

The map `v -> W^{-1/2} v` over vectors and matrices of `n` rows, for a data covariance `W` given as the variances of
the `n` values (a vector), a full matrix, or `nothing` for the identity.
"""
_whitening(::Nothing, n::Int) = v -> Float64.(v)

function _whitening(W::AbstractVector, n::Int)
    length(W) == n || throw(DimensionMismatch("the data covariance names $(length(W)) variances for $n values"))
    all(v -> isfinite(v) && v > 0, W) || throw(ArgumentError("every data variance must be finite and positive"))
    s = 1 ./ sqrt.(Vector{Float64}(W))
    return v -> v .* s
end

function _whitening(W::AbstractMatrix, n::Int)
    size(W) == (n, n) || throw(DimensionMismatch("the data covariance is $(size(W)) for $n values"))
    L = _covariance_factor(W, "data covariance").L
    return v -> L \ Float64.(v)
end

"""
    _whitened(H, y, W) -> (Hw, yw)

`W^{-1/2} H` and `W^{-1/2} y` ([`_whitening`](@ref)).
"""
function _whitened(H::AbstractMatrix, y::AbstractVector, W)
    _check_fit_data(H, y)
    whiten = _whitening(W, length(y))
    return whiten(H), whiten(y)
end

"""
    independent_pair_variance(joint::StructureFunction2DSumsAndCounts) -> Array

The variance of each distance bin's mean pair value, of shape `(n_distance, batch...)`, under the assumption that the
pairs are independent: the variance of the pair values in the bin over the number of pairs in it. Pairs sharing a point
are correlated, which this does not account for. Read from a value-binned joint histogram (`InvariantValueAxis`) of
integer pair counts, each value cell contributing its own mean, so the spread inside a value cell is not seen and the
result is a lower bound. A bin holding no pair is `NaN`. This is the data covariance `W` the fits take.
"""
function independent_pair_variance(
    joint::SFO.StructureFunction2DSumsAndCounts{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray{<:Integer},
                                                <:InvariantValueAxis},
)
    c = float.(joint.counts)
    s = joint.sums
    n = sum(c; dims = 2)
    s1 = sum(s; dims = 2)
    s2 = sum(ifelse.(c .> 0, s .^ 2 ./ c, zero(eltype(c))); dims = 2)
    return dropdims(ifelse.(n .> 0, max.(s2 ./ n .- (s1 ./ n) .^ 2, 0) ./ n, NaN); dims = 2)
end

independent_pair_variance(
    ::SFO.StructureFunction2DSumsAndCounts{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SeparationAngleAxis},
) = throw(ArgumentError(
    "independent_pair_variance reads the spread of the pair values in each distance bin from the value cells of a " *
    "joint histogram; this histogram bins the angle of the separation, whose cells carry no value. Bin the operator " *
    "value (second_axis = InvariantValueAxis()).",
))

independent_pair_variance(
    ::SFO.StructureFunction2DSumsAndCounts{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractArray{<:AbstractFloat},
                                           <:InvariantValueAxis},
) = throw(ArgumentError(
    "independent_pair_variance divides by the number of independent pairs; floating-point counts hold a weighted or " *
    "split pair mass, which is not a number of pairs. An unweighted histogram with an integer count type gives it.",
))

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
        prior isa AbstractVector && !all(v -> isfinite(v) && v > 0, prior) && throw(ArgumentError(
            "every prior variance must be finite and positive",
        ))
        prior isa AbstractMatrix && _covariance_factor(prior, "prior covariance")
        return new{P}(prior)
    end
end

"""
    NonNegativeLeastSquares()

Least squares with every fitted value constrained non-negative, by the Lawson–Hanson active-set
algorithm on the whitened system. For a flux that is the monotone-flux model, which cannot resolve a
sink; for a spectrum it is the constraint that a spectral density is non-negative. It returns no
covariance. `solve(method, H, y, W; maxiter, return_info=true)` also returns a
third value containing `converged`, `iterations`, `kkt_residual`, and `tolerance`.
The default two-value form throws on nonconvergence.
"""
struct NonNegativeLeastSquares <: AbstractFitMethod end

"""
    SegmentedPowerLaw(segments; slope_bounds = (-4.0, 1.0))

A shell spectrum that is a power law on each of `segments` wavenumber ranges, log-uniform between the
smallest and the largest wavenumber asked for, continuous at the breakpoints, with slopes held in
`slope_bounds` and a non-negative amplitude: `S + 1` parameters `(b₁, α₁, …, α_S)`. Fitted by bounded
Levenberg–Marquardt through `LsqFit` (`using LsqFit`), on the relative residual
`(S₂^{fit} - S₂)/scale` when no data covariance is given. Here
`scale = max(abs(S₂), sqrt(eps(Float64))*maximum(abs, S₂))` elementwise;
an all-zero observation vector uses unit scale. This finite floor permits zero
observations. Supply `W` when an observation-error model is available.
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

# A zero-mean Gaussian prior contributes rows L⁻¹ to the least-squares system,
# where P = LLᵀ. QR then avoids forming the squared-condition normal matrix.
_prior_rows(::Nothing, n::Int) = zeros(Float64, 0, n)
function _prior_rows(P::AbstractVector, n::Int)
    length(P) == n || throw(DimensionMismatch("the prior names $(length(P)) variances for $n values"))
    all(v -> isfinite(v) && v > 0, P) || throw(ArgumentError("prior variances must be finite and positive"))
    return LA.Diagonal(1 ./ sqrt.(Vector{Float64}(P)))
end
function _prior_rows(P::AbstractMatrix, n::Int)
    size(P) == (n, n) || throw(DimensionMismatch("the prior covariance is $(size(P)) for $n values"))
    return _covariance_factor(P, "prior covariance").L \ Matrix{Float64}(LA.I, n, n)
end

"""
    solve(method, H, y, W) -> (x, covariance)

Fit `y = H x` by `method` under the data covariance `W`. [`RegularizedLeastSquares`](@ref) returns the
posterior covariance; [`NonNegativeLeastSquares`](@ref) returns `nothing` for it.
The regularized solve uses a pivoted QR factorization of the whitened system
augmented by the prior. Numerically rank-deficient systems require a prior or a
smaller model. A dense covariance can lose its smallest eigenvalues to rounding
when the fitted system is poorly conditioned.
"""
function solve(m::RegularizedLeastSquares, H::AbstractMatrix, y::AbstractVector, W)
    _require_covariance(m, W)
    Hw, yw = _whitened(H, y, W)
    return _solve_prepared(m, _prepared(m, Hw), yw)
end

function solve(::NonNegativeLeastSquares, H::AbstractMatrix, y::AbstractVector, W;
               maxiter::Int = 10 * size(H, 2), return_info::Bool = false)
    Hw, yw = _whitened(H, y, W)
    info = _nnls(Hw, yw; maxiter, return_info=true)
    return_info && return (info.x, nothing, info)
    info.converged || error("NNLS did not converge in $(info.iterations) iterations (KKT residual $(info.kkt_residual))")
    return info.x, nothing
end

_require_covariance(::AbstractFitMethod, W) = nothing
_require_covariance(::RegularizedLeastSquares, ::Nothing) = throw(ArgumentError(
    "regularised least squares weighs the data by their covariance, which was not given; pass " *
    "`W`, the variance of each structure-function value — `independent_pair_variance` reads one " *
    "from a value-binned joint histogram — or a full covariance matrix",
))

"""
    _prepared(method, Hw) -> prepared

What `method` keeps of the whitened system `Hw` for any number of whitened right-hand sides: for regularised least
squares the pivoted QR factorization of `Hw` augmented by the prior's rows, checked for rank, and the posterior
covariance it implies; for non-negative least squares the system itself.
"""
function _prepared(m::RegularizedLeastSquares, Hw::AbstractMatrix)
    n = size(Hw, 2)
    penalty = _prior_rows(m.prior, n)
    A = vcat(Hw, penalty)
    size(A, 1) >= n || throw(ArgumentError("fit is underdetermined; supply a positive definite prior"))
    factor = LA.qr(A, LA.ColumnNorm())
    R = factor.R
    threshold = maximum(abs, LA.diag(R)) * max(size(A)...) * eps(Float64)
    minimum(abs, LA.diag(R)) > threshold ||
        throw(ArgumentError("fit is numerically rank deficient; supply a positive definite prior or reduce the model"))
    inverse_R = LA.UpperTriangular(R) \ Matrix{Float64}(LA.I, n, n)
    C = Matrix{Float64}(undef, n, n)
    C[factor.p, factor.p] = inverse_R * inverse_R'
    return (; factor, n_prior = size(penalty, 1), covariance = LA.Symmetric(C))
end

_prepared(::NonNegativeLeastSquares, Hw::AbstractMatrix) = Hw

"""The fit `(x, covariance)` of the whitened values `yw` through what [`_prepared`](@ref) kept."""
_solve_prepared(::RegularizedLeastSquares, p, yw::AbstractVector) =
    (p.factor \ vcat(yw, zeros(p.n_prior)), p.covariance)
_solve_prepared(::NonNegativeLeastSquares, Hw, yw::AbstractVector) = (_nnls(Hw, yw), nothing)

"""
    _nnls(A, b; maxiter=10size(A, 2), return_info=false) -> x

Minimize `‖A x − b‖₂` over `x ≥ 0` with a bounded Lawson–Hanson active-set iteration.
Failure to satisfy the KKT conditions throws. With `return_info=true`, return the
iterate, convergence flag, iteration count, KKT residual, and tolerance instead.
"""
function _nnls(A::AbstractMatrix, b::AbstractVector;
               maxiter::Int = 10 * size(A, 2), return_info::Bool = false)
    _check_fit_data(A, b)
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    A, b = Matrix{Float64}(A), Vector{Float64}(b)
    m, n = size(A)
    x = zeros(n)
    passive = falses(n)
    tol = 10 * eps(Float64) * max(m, n) * LA.opnorm(A, 1) * LA.norm(b)
    w = A' * b
    iterations = 0
    exhausted = false
    while maximum(w[.!passive]; init=-Inf) > tol
        if iterations >= maxiter
            exhausted = true
            break
        end
        passive[argmax(ifelse.(passive, -Inf, w))] = true
        while true
            if iterations >= maxiter
                exhausted = true
                break
            end
            iterations += 1
            candidate = zeros(n)
            # SVD gives a bounded minimum-norm solve when active columns are dependent.
            candidate[passive] = LA.svd(A[:, passive]) \ b
            all(>(0), candidate[passive]) && (x = candidate; break)
            alpha = 1.0
            for i in eachindex(x)
                if passive[i] && candidate[i] <= 0
                    denominator = x[i] - candidate[i]
                    alpha = min(alpha, denominator > 0 ? x[i] / denominator : 0.0)
                end
            end
            x .+= alpha .* (candidate .- x)
            xtol = 10 * eps(Float64) * maximum(abs, x; init=0.0)
            for i in eachindex(x)
                if passive[i] && x[i] <= xtol
                    passive[i] = false
                    x[i] = 0
                end
            end
            any(passive) || break
        end
        w = A' * (b - A * x)
        exhausted && break
    end
    # KKT: nonnegative x, zero gradient on its support, nonnegative gradient elsewhere.
    dual_violation = max(maximum(w[.!passive]; init=0.0), 0.0)
    active_violation = maximum(abs, w[passive]; init=0.0)
    kkt_residual = max(dual_violation, active_violation)
    converged = !exhausted && kkt_residual <= tol
    info = (; x, converged, iterations, kkt_residual, tolerance=tol)
    return_info && return info
    converged || error("NNLS did not converge in $iterations iterations (KKT residual $kkt_residual, tolerance $tol)")
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
    be = digitize_plan(edges)
    amps = Vector{T}(undef, S)
    amps[1] = p[1]
    for s in 1:(S - 1)
        amps[s + 1] = amps[s] * T(be[s + 1])^(p[s + 1] - p[s + 2])
    end
    return [begin
        s = clamp(searchsortedlast(be, kq), 1, S)
        amps[s] * T(kq)^p[s + 1]
    end for kq in k]
end

function _relative_scales(y)
    all(isfinite, y) || throw(ArgumentError("observations must be finite"))
    scale = maximum(abs, y; init=0.0)
    scale == 0 && return ones(Float64, length(y))
    return max.(abs.(y), sqrt(eps(Float64)) * scale)
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
    return r[keep], Array(_take(vals, keep)), keep
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

"""The values of a result on its distance bins, one column per slice of its batch axes, `NaN` where a bin holds no pair."""
_value_columns(sf::SFO.StructureFunction) = reshape(Float64.(collect(sf.values)), size(sf.values, 1), :)
function _value_columns(sf::SFO.StructureFunctionSumsAndCounts)
    s, c = collect(sf.sums), collect(sf.counts)
    return reshape(ifelse.(c .> 0, Float64.(s) ./ Float64.(c), NaN), size(s, 1), :)
end

"""The batch axes of a result: the axes of its values after the distance axis."""
_batch_axes(sf::SFO.StructureFunction) = size(sf.values)[2:end]
_batch_axes(sf::SFO.StructureFunctionSumsAndCounts) = size(sf.sums)[2:end]

"""One data covariance per slice of a batch: an array over the batch axes of variance vectors or covariance matrices."""
const SliceCovariances = AbstractArray{<:AbstractVecOrMat}

_slice_covariance(W::SliceCovariances, I) = W[I]
_slice_covariance(W, I) = W

_check_slice_covariances(W::SliceCovariances, bax) = size(W) == bax || throw(DimensionMismatch(
    "one data covariance per slice is an array over the batch axes $bax; got $(size(W))",
))
_check_slice_covariances(W, bax) = nothing

"""Fits of every slice as one fit for a result without batch axes, else the array of them over its batch axes."""
_slice_result(fits, bax::Tuple{}) = only(fits)
_slice_result(fits, bax::Tuple) = fits

"""
    _fit_slices(sf, W) -> slices

The values of `sf` to fit, one entry per slice of its batch axes (one for a result without them): the separations `r`
and values `y` of the bins holding a value, those bins `keep`, and the data covariance `W` over them. `W` is one data
covariance for every slice (a vector of variances or a matrix, over the bins holding a value or over all bins, or
`nothing`), or one per slice ([`SliceCovariances`](@ref)).
"""
function _fit_slices(sf::SFO.AbstractStructureFunction, W)
    bax = _batch_axes(sf)
    _check_slice_covariances(W, bax)
    r_all = collect(midpoints(sf.distance))
    Y = _value_columns(sf)
    nb = size(Y, 1)
    return map(CartesianIndices(bax)) do I
        y = view(Y, :, LinearIndices(bax)[I])
        keep = findall(isfinite, y)
        isempty(keep) && throw(ArgumentError("no structure function value to fit"))
        (; r = r_all[keep], y = y[keep], keep, W = _keep_variances(_slice_covariance(W, I), keep, nb))
    end
end

"""
    _paired_slices(L2, T2, W) -> slices

As [`_fit_slices`](@ref) for two results on one set of bins and batch axes, the values of each slice stacked as
`[L2; T2]` over the bins both hold, `W` covering the stacked values.
"""
function _paired_slices(L2::SFO.AbstractStructureFunction, T2::SFO.AbstractStructureFunction, W)
    L2.distance == T2.distance || throw(ArgumentError("the two structure functions must share their distance bins"))
    bax = _batch_axes(L2)
    _batch_axes(T2) == bax || throw(DimensionMismatch("the two structure functions must share their batch axes"))
    _check_slice_covariances(W, bax)
    r_all = collect(midpoints(L2.distance))
    A, B = _value_columns(L2), _value_columns(T2)
    nb = size(A, 1)
    return map(CartesianIndices(bax)) do I
        j = LinearIndices(bax)[I]
        a, b = view(A, :, j), view(B, :, j)
        keep = intersect(findall(isfinite, a), findall(isfinite, b))
        isempty(keep) && throw(ArgumentError("no bin holds a value in both structure functions"))
        (; r = r_all[keep], y = vcat(a[keep], b[keep]), keep, W = _stacked_variances(_slice_covariance(W, I), keep, nb))
    end
end

_stacked_variances(::Nothing, keep, nb) = nothing
function _stacked_variances(W::AbstractVector, keep, nb)
    n = length(keep)
    length(W) == 2n && return Vector{Float64}(W)
    length(W) == 2nb || throw(DimensionMismatch(
        "the data covariance must cover the $(2n) stacked values the two results share, or the $(2nb) of every bin",
    ))
    return vcat(Float64.(W[keep]), Float64.(W[nb .+ keep]))
end
_stacked_variances(W::AbstractMatrix, keep, nb) = (size(W, 1) == 2length(keep) || throw(DimensionMismatch(
    "a full data covariance must cover the $(2length(keep)) stacked values the two results share",
)); Matrix{Float64}(W))

"""
    _linear_fits(model_for, method, slices, W) -> fits

`(; model, x, covariance)` for each of `slices` ([`_fit_slices`](@ref)), fitted by `method` through the forward model
`model_for(r)` of its kept bins: one model per distinct set of kept bins, and with one data covariance `W` for every
slice, one whitened system and one [`_prepared`](@ref) solve per model.
"""
function _linear_fits(model_for, method::AbstractFitMethod, slices, W)
    _require_covariance(method, W)
    models = _slice_models(model_for, slices)
    prepared = Dict(map(unique(s.keep for s in slices)) do keep
        s = slices[findfirst(t -> t.keep == keep, slices)]
        whiten = _whitening(s.W, length(s.y))
        keep => (whiten, _prepared(method, whiten(forward_matrix(models[keep]))))
    end)
    return map(slices) do s
        model = models[s.keep]
        _check_fit_data(forward_matrix(model), s.y)
        whiten, p = prepared[s.keep]
        x, C = _solve_prepared(method, p, whiten(s.y))
        (; model, x, covariance = C)
    end
end

function _linear_fits(model_for, method::AbstractFitMethod, slices, W::SliceCovariances)
    models = _slice_models(model_for, slices)
    return map(slices) do s
        _require_covariance(method, s.W)
        model = models[s.keep]
        Hw, yw = _whitened(forward_matrix(model), s.y, s.W)
        x, C = _solve_prepared(method, _prepared(method, Hw), yw)
        (; model, x, covariance = C)
    end
end

"""The forward model `model_for(r)` of each distinct set of kept bins among `slices`."""
_slice_models(model_for, slices) =
    Dict(keep => model_for(slices[findfirst(t -> t.keep == keep, slices)].r) for keep in unique(s.keep for s in slices))

"""
    fit_spectrum(sf, k_edges, method, ::Val{D}; W = nothing) -> (k, E, covariance, …)

The shell spectrum on the wavenumber bins `k_edges` that reproduces the second-order trace `sf`
through [`SpectrumForwardModel`](@ref), fitted by `method`. `W` is the data covariance: the variance
of each bin's value (a vector over the bins holding a value, or over all bins), or a full matrix.
Returns the bin centres `k`, the fitted `E` on the bins and the posterior `covariance` (`nothing` for
a method that gives none). A [`SegmentedPowerLaw`](@ref) returns `E` evaluated at `k`, and in
addition the fitted `parameters`, their `covariance`, the segment `breakpoints` and `converged`.

A result with batch axes is fitted slice by slice, returning an array of these over its batch axes; `W` is then one
data covariance for every slice, or an array over the batch axes of one per slice ([`SliceCovariances`](@ref)). Slices
holding values in the same bins share one forward model, and with one data covariance, one factorization.
"""
function fit_spectrum(sf::SFO.AbstractStructureFunction, k_edges::AbstractVector, method::AbstractFitMethod,
                      ::Val{D}; W = nothing) where {D}
    assert_invertible(sf.operator)
    return _slice_result(_fit_spectra(method, Val(D), _fit_slices(sf, W), k_edges, W), _batch_axes(sf))
end

function _fit_spectra(method::AbstractFitMethod, ::Val{D}, slices, k_edges, W) where {D}
    fits = _linear_fits(r -> SpectrumForwardModel(Val(D), r, k_edges), method, slices, W)
    return map(f -> (k = _bin_centres(k_edges), E = f.x, covariance = f.covariance), fits)
end

_fit_spectra(method::SegmentedPowerLaw, ::Val{D}, slices, k_edges, W) where {D} =
    map(s -> _fit_spectrum(method, Val(D), s.r, s.y, k_edges, s.W), slices)

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
stacked `[E; B]`. Results with batch axes are fitted slice by slice, as [`fit_spectrum`](@ref) fits them.
"""
function fit_helmholtz_spectra(L2::SFO.AbstractStructureFunction, T2::SFO.AbstractStructureFunction,
                               k_edges::AbstractVector, method::AbstractFitMethod; W = nothing)
    _assert_projection(L2.operator, 2, 0, "longitudinal")
    _assert_projection(T2.operator, 0, 2, "transverse")
    nk = length(k_edges) - 1
    fits = _linear_fits(r -> HelmholtzForwardModel(r, k_edges), method, _paired_slices(L2, T2, W), W)
    return _slice_result(map(f -> (k = _bin_centres(k_edges), E = f.x[1:nk], B = f.x[(nk + 1):end],
                                   covariance = f.covariance), fits), _batch_axes(L2))
end

_value_count(sf::SFO.StructureFunction) = length(sf.values)
_value_count(sf::SFO.StructureFunctionSumsAndCounts) = length(sf.sums)

"""
    fit_flux(sf, k_edges, method; W = nothing) -> (k, F, ξ, ε, covariance, flux_covariance)

The piecewise-constant spectral energy flux that reproduces the third-order structure function
`sf = ⟨δu_L ‖δu‖²⟩` (`S3SFType`) of a two-dimensional isotropic flow through [`FluxForwardModel`](@ref).
`F` is the flux at the bin centres `k` (positive towards small scales), `ξ` the injection density on
the bins and `ε` the flux towards large scales below the first bin; `covariance` is over `(ε, ξ…)`
and `flux_covariance = G C Gᵀ` over `F`. A result with batch axes is fitted slice by slice, as
[`fit_spectrum`](@ref) fits it.
"""
function fit_flux(sf::SFO.AbstractStructureFunction, k_edges::AbstractVector, method::AbstractFitMethod;
                  W = nothing)
    sf.operator isa SFT.S3SFType || throw(ArgumentError(
        "the flux is fitted to ⟨δu_L ‖δu‖²⟩ (S3SFType); got $(nameof(typeof(sf.operator)))",
    ))
    fits = _linear_fits(r -> FluxForwardModel(r, k_edges), method, _fit_slices(sf, W), W)
    return _slice_result(map(fits) do f
        G, x, C = f.model.G, f.x, f.covariance
        CF = C === nothing ? nothing : LA.Symmetric(G * C * G')
        (k = _bin_centres(k_edges), F = G * x, ξ = x[2:end], ε = x[1], covariance = C, flux_covariance = CF)
    end, _batch_axes(sf))
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
    _require_covariance(RegularizedLeastSquares(nothing), W)
    Hw, yw = _whitened(H, y, W)
    misfit = Vector{Float64}(undef, length(priors))
    norm = Vector{Float64}(undef, length(priors))
    for (i, p) in enumerate(priors)
        method = RegularizedLeastSquares(p isa AbstractVector ? p : fill(Float64(p), n))
        x, _ = _solve_prepared(method, _prepared(method, Hw), yw)
        misfit[i] = sum(abs2, Hw * x - yw)
        norm[i] = LA.norm(x)
    end
    return (misfit = misfit, norm = norm)
end

"""
    select_segments(sf, k_edges, S_range, ::Val{D}; W = nothing, slope_bounds = (-4.0, 1.0))
        -> (segments, misfits, fits)

Fit a [`SegmentedPowerLaw`](@ref) with each number of segments in `S_range` and report the one
minimizing the mean absolute residual divided by the same floored observation
scale as the default segmented fit. Returns every misfit and fitted model.
This is an in-sample residual criterion; it does not penalize model complexity
or estimate predictive error.
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
        sum(abs.(yfit .- y) ./ _relative_scales(y)) / length(y)
    end for f in fits]
    best = argmin(misfits)
    return (segments = collect(S_range)[best], misfits = misfits, fits = fits)
end
