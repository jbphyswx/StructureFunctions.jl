"""
    isotropic_kernel(::Val{D}, x)

Angular average of `cos(k·r)` over the directions of `r` in `D` dimensions, at `x = kr`.

`cos` on a line, `J₀` on a plane and `sin(x)/x` in a volume. This is the kernel of the isotropic
Fourier transform, so a single mode of amplitude `A` and wavenumber `k₀` has
`S₂(r) = A²[1 - isotropic_kernel(Val(D), k₀ r)]` after angular averaging.
"""
@inline isotropic_kernel(::Val{1}, x) = cos(x)

@inline function isotropic_kernel(::Val{3}, x)
    ax = abs(x)
    return ax < cbrt(eps(typeof(ax))) ? one(x) - x * x / 6 : sin(x) / x
end

function isotropic_kernel(::Val{D}, x) where {D}
    D == 2 && throw(ArgumentError(
        "the two-dimensional kernel is the Bessel function J₀, which this session has not loaded. " *
        "Run `using Bessels`. One and three dimensions need no such package.",
    ))
    throw(ArgumentError(
        "no isotropic transform kernel for $D dimensions; the angular average of cos(k·r) is " *
        "implemented for D = 1, 2 and 3.",
    ))
end

"""
    assert_variogram(operator)

Refuse a structure function that is not second order, since only a second-order moment is twice a
variogram.
"""
assert_variogram(::SFT.SecondOrderStructureFunctionType) = nothing

function assert_variogram(op::SFT.ProjectedStructureFunctionType{NL, NT}) where {NL, NT}
    NL + NT == 2 && return nothing
    return throw(ArgumentError(
        "a covariance follows from a second-order structure function; order $(NL + NT) does not " *
        "reduce to one.",
    ))
end

function assert_variogram(op::SFT.ScalarStructureFunctionType{P}) where {P}
    P == 2 && return nothing
    return throw(ArgumentError(
        "a covariance follows from a second-order structure function; order $P does not reduce to " *
        "one.",
    ))
end

assert_variogram(op) = throw(ArgumentError(
    "$(nameof(typeof(op))) is not a second-order structure function, so it is not twice a variogram.",
))

"""
    covariance(result, variance) -> (separations, C)

Covariance function from a second-order structure function and the field's variance.

A structure function is twice the variogram, `D(r) = 2[C(0) - C(r)]`, so

```math
C(r) = C(0) - D(r)/2
```

with `C(0)` the variance. **`variance` must be supplied and cannot be recovered from `D`**, which is
blind to it — the same blindness that makes `k = 0` unavailable to [`isotropic_spectrum`](@ref).

Valid only if the field is **second-order stationary**; an intrinsically stationary field has a
variogram but need not have a finite variance.
"""
function covariance(sf::Union{SFO.StructureFunction, SFO.StructureFunctionSumsAndCounts}, variance::Real)
    assert_variogram(sf.operator)
    r, vals, keep = _binned(sf)
    isempty(keep) && throw(ArgumentError("no bin holds a structure function value"))
    return r[keep], variance .- _take(vals, keep) ./ 2
end

"""
    covariance_matrix(points, separations, C; metric, check_posdef = true, posdef_rtol = 1e-6)

Covariance matrix over `points`, evaluating the covariance function `(separations, C)` at each pair
distance by linear interpolation and holding it constant outside the sampled range.

The matrix is checked for positive semi-definiteness, which fails for a covariance function that is
not a valid kernel and for a valid one sampled too coarsely — interpolation does not preserve
positive-definiteness, and its error falls as the square of the separation spacing.
`check_posdef = false` returns the matrix unchecked.
"""
function covariance_matrix(
    points::AbstractMatrix, separations::AbstractVector, C::AbstractVector;
    metric::DI.PreMetric = DI.Euclidean(), check_posdef::Bool = true, posdef_rtol::Real = 1e-6,
)
    length(separations) == length(C) || throw(DimensionMismatch(
        "separations and C must agree in length; got $(length(separations)) and $(length(C))",
    ))
    issorted(separations) || throw(ArgumentError("separations must be sorted"))
    n = size(points, 2)
    FT = float(promote_type(eltype(points), eltype(C)))
    Σ = Matrix{FT}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        d = metric(view(points, :, i), view(points, :, j))
        Σ[i, j] = _interp(separations, C, d)
    end
    Σ .= (Σ .+ transpose(Σ)) ./ 2          # a distance is symmetric; round-off need not be
    if check_posdef
        scale = maximum(abs, Σ)
        λ = minimum(LA.eigvals(LA.Symmetric(Σ)))
        λ < -posdef_rtol * scale && throw(ArgumentError(
            "the covariance function does not give a positive semi-definite matrix on these " *
            "points: most negative eigenvalue $λ, i.e. $(λ / scale) of the matrix scale. Either " *
            "it is not a valid covariance kernel, or it is sampled too coarsely — interpolation " *
            "does not preserve positive-definiteness and its error falls as the square of the " *
            "separation spacing. Refine `separations`, fit a valid model, raise `posdef_rtol`, " *
            "or pass `check_posdef = false`.",
        ))
    end
    return Σ
end

@inline function _interp(x::AbstractVector, y::AbstractVector, q)
    q <= first(x) && return float(first(y))
    q >= last(x) && return float(last(y))
    i = searchsortedlast(x, q)
    i >= length(x) && return float(last(y))
    t = (q - x[i]) / (x[i + 1] - x[i])
    return float(y[i] * (1 - t) + y[i + 1] * t)
end

"""
    helmholtz_spectra(h, wavenumbers; variance)

Rotational and divergent spectral densities from a two-dimensional Helmholtz decomposition, by the
transforms of its longitudinal and transverse functions that `helmholtz_spectra(L2, T2, wavenumbers;
variance)` documents. Bins holding no pair in either function are dropped.
"""
function helmholtz_spectra(h::SFO.HelmholtzDecomposition2D, wavenumbers::AbstractVector; variance::Real)
    keep = findall((Array(h.divergent_counts) .> 0) .& (Array(h.rotational_counts) .> 0))
    isempty(keep) && throw(ArgumentError("no bin holds a value in both the longitudinal and the transverse function"))
    r = collect(midpoints(h.distance_bins))[keep]
    return _helmholtz_spectra(r, _take(h.longitudinal_values, keep), _take(h.transverse_values, keep), wavenumbers,
                              variance)
end

"""
    helmholtz_spectra(L2, T2, wavenumbers; variance)

Rotational and divergent spectral densities of a two-dimensional field from its longitudinal and
transverse second-order structure functions, binned on the same edges.

The sum and the difference of the two projections transform separately,

```math
P_E + P_B = -\\frac{1}{4π} ∫_0^∞ (D_{LL} + D_{TT} - a)\\, J_0(kr)\\, r\\, dr, \\qquad
P_E - P_B = \\frac{1}{4π} ∫_0^∞ (D_{LL} - D_{TT})\\, J_2(kr)\\, r\\, dr,
```

with `E` the divergent (gradient) part and `B` the rotational (curl) part, on the convention that a
density integrates over `d²k` to the variance. The first line is [`isotropic_spectrum`](@ref) of the
trace, whose large-separation limit `a` is twice `variance`, the field's variance summed over its two
components; the second needs no constant, since `D_LL − D_TT` decays on its own. Both kernels need
`Bessels`.

Its error is the truncation of the Hankel integrals at the last separation.
"""
function helmholtz_spectra(
    L2::SFO.AbstractStructureFunction, T2::SFO.AbstractStructureFunction, wavenumbers::AbstractVector;
    variance::Real,
)
    _assert_projection(L2.operator, 2, 0, "longitudinal")
    _assert_projection(T2.operator, 0, 2, "transverse")
    r, dll, dtt = _paired_bins(L2, T2)
    return _helmholtz_spectra(r, dll, dtt, wavenumbers, variance)
end

function _helmholtz_spectra(r, dll, dtt, wavenumbers, variance)
    total = isotropic_spectrum(SFT.S2SFType(), r, dll .+ dtt, wavenumbers, Val(2); variance)
    diff = _hankel(Val(2), r, dll .- dtt, wavenumbers) ./ (4π)
    return (rotational = (total .- diff) ./ 2, divergent = (total .+ diff) ./ 2)
end

function _assert_projection(op, NL::Int, NT::Int, name::String)
    op isa SFT.ProjectedStructureFunctionType{NL, NT} && return nothing
    throw(ArgumentError(
        "the $name argument must be the second-order $name projection, " *
        "ProjectedStructureFunctionType{$NL, $NT}; got $(nameof(typeof(op))).",
    ))
end

"""
    _binned(sf) -> (r, values, keep)

The bin abscissae of a result on the host, its values in the result's own array family (`NaN` in a bin holding no
pair), and the bins holding a value, on the host.
"""
function _binned(sf::SFO.StructureFunction)
    return collect(midpoints(sf.distance)), sf.values, findall(isfinite, Array(sf.values))
end

function _binned(sf::SFO.StructureFunctionSumsAndCounts)
    vals = ifelse.(sf.counts .> 0, sf.sums ./ sf.counts, oftype(float(zero(eltype(sf.sums))), NaN))
    return collect(midpoints(sf.distance)), vals, findall(>(0), Array(sf.counts))
end

"""The entries `keep` (host indices) of `v`, in `v`'s array family."""
_take(v::Array, keep) = v[keep]
_take(v::AbstractArray, keep) = v[_on(v, Int, keep)]

"""`x` with element type `T` in the array family of `like`."""
_on(like::AbstractArray, ::Type{T}, x::AbstractArray) where {T} = copyto!(similar(like, T, size(x)), x)

"""The last `n` entries of `v`, on the host."""
_tail(v::AbstractVector, n::Int) = Array(view(v, (lastindex(v) - n + 1):lastindex(v)))

# Two results on one set of edges, reduced to the bins both hold.
function _paired_bins(a::SFO.AbstractStructureFunction, b::SFO.AbstractStructureFunction)
    a.distance == b.distance || throw(ArgumentError(
        "the two structure functions must share their distance bins",
    ))
    r, va, ka = _binned(a)
    _, vb, kb = _binned(b)
    keep = intersect(ka, kb)
    isempty(keep) && throw(ArgumentError("no bin holds a value in both structure functions"))
    return r[keep], _take(va, keep), _take(vb, keep)
end

"""
    _kernel_transform(kernel, separations, weights, values, wavenumbers, FT)

`Σ_i kernel(k, r_i) w_i values_i` at each wavenumber `k`, for the quadrature weights `w` of the separations `r`: the
matrix `kernel(k_j, r_i) w_i` built on the host, applied in the array family of `values`.
"""
function _kernel_transform(kernel, separations::AbstractVector, weights::AbstractVector, values::AbstractVector,
                           wavenumbers::AbstractVector, ::Type{FT}) where {FT}
    r, w, k = FT.(collect(separations)), FT.(weights), FT.(collect(wavenumbers))
    K = [FT(kernel(kj, ri)) * wi for kj in k, (ri, wi) in zip(r, w)]
    return _on(values, FT, K) * FT.(values)
end

"""`∫₀^R f(r) J_N(kr) r dr` at each wavenumber by the trapezoid rule from the origin, where the integrand vanishes."""
function _hankel(::Val{N}, separations::AbstractVector, values::AbstractVector, wavenumbers::AbstractVector) where {N}
    FT = float(promote_type(eltype(separations), eltype(values), eltype(wavenumbers)))
    r = FT.(collect(separations)) # this allocates
    return _kernel_transform((k, r) -> bessel_kernel(Val(N), k * r) * r, r, _trapezoid_weights(r), values,
                             wavenumbers, FT)
end

"""
    bessel_kernel(::Val{N}, x)

Bessel function of the first kind of order `N`.

Orders 0 through 3 are supported once `Bessels` is loaded.
"""
function bessel_kernel(::Val{N}, x) where {N}
    throw(ArgumentError(
        "the Bessel function J$N is not available; run `using Bessels`. Orders 0 to 3 are " *
        "supported once it is loaded.",
    ))
end

"""
    solid_angle(::Val{D})

Total measure of the unit sphere in `D` dimensions: `2`, `2π`, `4π`.
"""
@inline solid_angle(::Val{1}) = 2.0
@inline solid_angle(::Val{2}) = 2π
@inline solid_angle(::Val{3}) = 4π

"""
    assert_invertible(operator)

Refuse a structure function that does not map to a spectrum by the plain Wiener–Khinchin route.

Only the second-order **trace** `⟨‖δu‖²⟩` does. `L2SF` and `T2SF` are single projections and carry
roughly half the trace, so inverting one as though it were the trace silently returns half the
spectrum; converting from them needs the isotropy relation between longitudinal and transverse
components, which this transform does not apply. Orders other than two are fluxes, not spectra.
"""
assert_invertible(::SFT.SecondOrderStructureFunctionType) = nothing
assert_invertible(::SFT.ScalarStructureFunctionType{2}) = nothing

function assert_invertible(op::SFT.ProjectedStructureFunctionType{NL, NT}) where {NL, NT}
    if NL + NT == 2
        throw(ArgumentError(
            "a spectrum follows from the second-order trace ⟨‖δu‖²⟩ (`S2SFType`), not from " *
            "$(nameof(typeof(op))){$NL,$NT}. A single projection carries about half the trace, so " *
            "inverting it here would return about half the spectrum. Recovering a spectrum from a " *
            "longitudinal or transverse component needs the isotropy relation that links them, " *
            "which this transform does not apply.",
        ))
    end
    return throw(ArgumentError(
        "only the second-order structure function may be inverted to a spectrum; order $(NL + NT) " *
        "is a flux, not a spectrum.",
    ))
end

# Each Helmholtz component is a share of the second-order trace and inverts as the trace does.
assert_invertible(::SFT.RotationalSecondOrderStructureFunctionType) = nothing
assert_invertible(::SFT.DivergentSecondOrderStructureFunctionType) = nothing

assert_invertible(op) = throw(ArgumentError(
    "$(nameof(typeof(op))) has no spectral inverse; a spectrum follows only from the second-order " *
    "trace ⟨‖δu‖²⟩ (`S2SFType`).",
))

"""
    isotropic_spectrum(operator, separations, values, wavenumbers, ::Val{D}; variance)

Power spectral density at each of `wavenumbers`, from a second-order structure function sampled at
`separations`.

`D(r) = 2[C(0) - C(r)]`, so the transform of `D` differs from that of `-2C` only by a constant, whose
transform is confined to `k = 0`. Every returned wavenumber must therefore be nonzero, and the
`k = 0` mode is not recoverable from `D`.

`variance` is the field's variance, summed over the components of a vector field. Twice it is the
large-separation limit of the structure function, subtracted so the integrand decays; the integral
stops at the last separation, so any other constant leaves an error that rings in `k`.

The integral runs from zero separation, where the structure function vanishes, by the trapezoid rule
over the separations supplied and the origin.

Normalised so that integrating the density over `d^D k` returns the field's variance, which fixes
the convention: `∫₀^∞ shell_spectrum(...) dk == var(u)`.
"""
function isotropic_spectrum(
    operator, separations::AbstractVector, values::AbstractVector,
    wavenumbers::AbstractVector, ::Val{D};
    variance::Real,
) where {D}
    assert_invertible(operator)
    length(separations) == length(values) || throw(DimensionMismatch(
        "separations and values must agree in length; got $(length(separations)) and $(length(values))",
    ))
    any(iszero, wavenumbers) && throw(ArgumentError(
        "the k = 0 mode is not recoverable from a structure function, which is blind to the mean " *
        "and to the variance; request nonzero wavenumbers only.",
    ))
    r = collect(separations)
    issorted(r) || throw(ArgumentError("separations must be sorted"))
    first(r) >= 0 || throw(ArgumentError("separations must be non-negative; got $(first(r))"))

    FT = float(promote_type(eltype(separations), eltype(values), eltype(wavenumbers)))
    limit = 2 * FT(variance)
    decaying = FT.(values) .- limit
    if first(r) > 0
        r = vcat(zero(eltype(r)), r)
        decaying = vcat(fill!(similar(decaying, 1), -limit), decaying)
    end
    acc = _kernel_transform((k, ri) -> isotropic_kernel(Val(D), k * ri) * ri^(D - 1), r, _trapezoid_weights(r),
                            decaying, wavenumbers, FT)
    return acc .* (-solid_angle(Val(D)) / (2 * (2 * FT(π))^D))
end

"""
    isotropic_spectrum(result, wavenumbers, ::Val{D}; variance)

Spectral density from a structure function result, taking the operator, the separations and the
values from the result itself.

The abscissa is the bin representative of `result`'s edges. Bins holding no pair are dropped.

The transform averages over the directions of the separation, so it assumes the pairs behind each
bin sample direction uniformly. Scattered points do; a rectilinear grid does **not**, and on gridded
data the separations available at a given `r` are biased toward the lattice axes.
"""
function isotropic_spectrum(sf::Union{SFO.StructureFunction, SFO.StructureFunctionSumsAndCounts},
                            wavenumbers::AbstractVector, ::Val{D}; kwargs...) where {D}
    r, vals, keep = _binned(sf)
    isempty(keep) && throw(ArgumentError("no bin holds a structure function value to transform"))
    return isotropic_spectrum(sf.operator, r[keep], _take(vals, keep), wavenumbers, Val(D); kwargs...)
end

"""
    isotropic_spectrum(result, geometry::SphericalGeometry, lmax) -> (l, C)

Angular power spectrum `C_l`, `l = 1:lmax`, of a scalar or of the trace on a sphere, from a
second-order structure function binned in separation `r = R σ`.

The isotropic correlation is `C(σ) = Σ_l (2l+1)/(4π) C_l P_l(cos σ)` and `D(σ) = 2[C(0) − C(σ)]`, so by
the orthogonality of the Legendre polynomials

```math
C_l = -π ∫_0^π D(σ)\\, P_l(\\cos σ)\\, \\sin σ\\, dσ, \\qquad l ≥ 1,
```

with no variance needed: the constant `C(0)` is orthogonal to every `P_l` but `P_0`, which is the
one degree lost — the sphere's `k = 0`. Each bin's value is taken as constant across the bin and the
kernel is integrated over the bin in closed form, so the only error is that of the binning itself.
Every bin must hold a value: the integral runs over the whole sphere. A kernel-binned result on
[`HarmonicNodes`](@ref) integrates by the nodes' quadrature weights.
"""
function isotropic_spectrum(sf::SFO.AbstractStructureFunction, g::SFH.SphericalGeometry, lmax::Integer)
    assert_invertible(sf.operator)
    D, Q = _legendre_quadrature(sf, sf.distance, g, lmax)   # values, and ∫ P_l sin σ dσ over each value's support, l = 0:lmax
    FT = promote_type(Float64, eltype(D))
    return (l = 1:lmax, C = (transpose(_on(D, FT, Q[:, 2:end])) * FT.(D)) .* -FT(π))
end

"""
    helmholtz_spectra(L2, T2, geometry::SphericalGeometry, lmax; variance) -> (l, E, B)

Gradient (`E`) and curl (`B`) angular power spectra, `l = 1:lmax`, of a tangent vector field on a
sphere, from its longitudinal and transverse second-order structure functions binned in `r = R σ`.

In the geodesic frame `ξ₊ = C_LL + C_TT = Σ_l (2l+1)/(4π) (C^E_l + C^B_l) d^l_{11}(σ)` and
`ξ₋ = C_LL − C_TT = −Σ_l (2l+1)/(4π) (C^E_l − C^B_l) d^l_{1,−1}(σ)`, and the Wigner functions are
orthogonal with `∫₀^π d^l_{ss'} d^{l'}_{ss'} sin σ dσ = 2δ_{ll'}/(2l+1)`, so

```math
C^E_l - C^B_l = π ∫_0^π (D_{LL} - D_{TT})\\, d^l_{1,-1}\\, \\sin σ\\, dσ, \\qquad
C^E_l + C^B_l = 2π ∫_0^π \\Big(⟨‖u‖²⟩ - \\frac{D_{LL} + D_{TT}}{2}\\Big)\\, d^l_{11}\\, \\sin σ\\, dσ .
```

The difference needs no constant because `C_LL(0) = C_TT(0)`; the sum needs the field's mean square
`⟨‖u‖²⟩`, which a structure function cannot supply, so `variance` is required. Bins are integrated in
closed form as in [`isotropic_spectrum`](@ref) on a sphere, nodes by their quadrature weights.
"""
function helmholtz_spectra(
    L2::SFO.AbstractStructureFunction, T2::SFO.AbstractStructureFunction, g::SFH.SphericalGeometry,
    lmax::Integer; variance::Real,
)
    _assert_projection(L2.operator, 2, 0, "longitudinal")
    _assert_projection(T2.operator, 0, 2, "transverse")
    L2.distance == T2.distance || throw(ArgumentError(
        "the two structure functions must share their distance bins",
    ))
    DLL, Gp, Gm = _wigner_quadrature(L2, L2.distance, g, lmax)   # values, ∫ d^l_{11} sin σ dσ and ∫ d^l_{1,-1} sin σ dσ, l = 1:lmax
    DTT, _, _ = _wigner_quadrature(T2, T2.distance, g, lmax)
    FT = promote_type(Float64, eltype(DLL), eltype(DTT))
    diff = (transpose(_on(DLL, FT, Gm)) * FT.(DLL .- DTT)) .* FT(π)
    total = (transpose(_on(DLL, FT, Gp)) * (FT(variance) .- FT.(DLL .+ DTT) ./ 2)) .* (2 * FT(π))
    return (l = 1:lmax, E = (total .+ diff) ./ 2, B = (total .- diff) ./ 2)
end

# Bin values and edges in central angle, every bin holding a value.
function _spherical_bins(sf::SFO.AbstractStructureFunction, g::SFH.SphericalGeometry)
    _, vals, keep = _binned(sf)
    length(keep) == length(vals) || throw(ArgumentError(
        "every bin must hold a value to integrate over the sphere; bins $(setdiff(eachindex(vals), keep)) are empty",
    ))
    edges = collect(sf.distance) ./ g.radius
    last(edges) <= π * (1 + 1e-12) || throw(ArgumentError(
        "the bins reach a central angle of $(last(edges)) on a sphere of radius $(g.radius); the largest " *
        "separation on a sphere is π·R",
    ))
    return vals, edges
end

# Every node's value, in the nodes' order.
function _node_values(sf::SFO.AbstractStructureFunction)
    _, vals, keep = _binned(sf)
    length(keep) == length(vals) || throw(ArgumentError(
        "every node must hold a value to integrate over the sphere; nodes $(setdiff(eachindex(vals), keep)) have none",
    ))
    return vals
end

# The values and, for l = 0:L, `∫ P_l(cos σ) sin σ dσ` over each value's support as (n_values, L + 1):
# closed-form bin integrals for edges, quadrature weights for nodes.
function _legendre_quadrature(sf, edges, g, L)
    D, e = _spherical_bins(sf, g)
    return D, _legendre_bin_integrals(e, L)
end

function _legendre_quadrature(sf, nodes::HarmonicNodes, g, L)
    D = _node_values(sf)
    Q = Matrix{Float64}(undef, length(nodes), L + 1)
    for k in eachindex(D)
        Q[k, :] .= nodes.weights[k] .* _legendre_values(cos(nodes.separations[k]), L)
    end
    return D, Q
end

# Likewise `∫ d^l_{11} sin σ dσ` and `∫ d^l_{1,−1} sin σ dσ`, l = 1:L.
function _wigner_quadrature(sf, edges, g, L)
    D, e = _spherical_bins(sf, g)
    Gp, Gm = _wigner_bin_integrals(e, L)
    return D, Gp, Gm
end

function _wigner_quadrature(sf, nodes::HarmonicNodes, g, L)
    D = _node_values(sf)
    Gp = Matrix{Float64}(undef, length(nodes), L)
    Gm = Matrix{Float64}(undef, length(nodes), L)
    for k in eachindex(D)
        β = nodes.separations[k]
        Gp[k, :] .= nodes.weights[k] .* @view(wigner_d_column(1, 1, β, L)[2:end])
        Gm[k, :] .= nodes.weights[k] .* @view(wigner_d_column(1, -1, β, L)[2:end])
    end
    return D, Gp, Gm
end

# P_0 … P_L at x by the three-term recurrence.
function _legendre_values(x::Real, L::Integer)
    P = Vector{typeof(float(x))}(undef, L + 1)
    P[1] = one(x)
    L >= 1 && (P[2] = x)
    for l in 1:(L - 1)
        P[l + 2] = ((2l + 1) * x * P[l + 1] - l * P[l]) / (l + 1)
    end
    return P
end

# ∫ P_l dx = (P_{l+1} − P_{l−1})/(2l+1) for l ≥ 1, and x for l = 0; `P` holds P_0 … P_{L+1}.
@inline _legendre_antiderivative(P, l, x) = l == 0 ? x : (P[l + 2] - P[l]) / (2l + 1)

# ∫_{σ_i}^{σ_{i+1}} P_l(cos σ) sin σ dσ for every bin and l = 0:L, as (n_bins, L + 1).
function _legendre_bin_integrals(edges::AbstractVector, L::Integer)
    x = cos.(edges)
    nb = length(edges) - 1
    Q = Matrix{Float64}(undef, nb, L + 1)
    Plo = _legendre_values(x[1], L + 1)
    for i in 1:nb
        Phi = _legendre_values(x[i + 1], L + 1)
        for l in 0:L
            Q[i, l + 1] = _legendre_antiderivative(Plo, l, x[i]) - _legendre_antiderivative(Phi, l, x[i + 1])
        end
        Plo = Phi
    end
    return Q
end

# ∫_{σ_i}^{σ_{i+1}} d^l_{11} sin σ dσ and ∫ d^l_{1,−1} sin σ dσ for l = 1:L, from
# d^l_{11} = P_l + (1 − x) P_l'/(l(l+1)), d^l_{1,−1} = −P_l + (1 + x) P_l'/(l(l+1)) and
# ∫ x P_l' dx = x P_l − ∫ P_l dx.
function _wigner_bin_integrals(edges::AbstractVector, L::Integer)
    x = cos.(edges)
    nb = length(edges) - 1
    Gp = Matrix{Float64}(undef, nb, L)
    Gm = Matrix{Float64}(undef, nb, L)
    anti(P, l, xx) = begin
        Ql = _legendre_antiderivative(P, l, xx)
        Pl = P[l + 1]
        (Ql + ((1 - xx) * Pl + Ql) / (l * (l + 1)), -Ql + ((1 + xx) * Pl - Ql) / (l * (l + 1)))
    end
    Plo = _legendre_values(x[1], L + 1)
    for i in 1:nb
        Phi = _legendre_values(x[i + 1], L + 1)
        for l in 1:L
            plo, mlo = anti(Plo, l, x[i])
            phi, mhi = anti(Phi, l, x[i + 1])
            Gp[i, l] = plo - phi
            Gm[i, l] = mlo - mhi
        end
        Plo = Phi
    end
    return Gp, Gm
end

"""
    shell_spectrum(P, wavenumbers, ::Val{D})

Shell-integrated spectrum `E(k) = Ω_D k^(D-1) P(k)` from a power spectral density.

`E` integrates over `k` to the variance the density integrates over `d^D k` to, which is the form
the inertial-range scaling laws are stated in.
"""
shell_spectrum(P::AbstractVector, wavenumbers::AbstractVector, ::Val{D}) where {D} =
    solid_angle(Val(D)) .* _on(P, float(eltype(wavenumbers)), wavenumbers) .^ (D - 1) .* P

"""
    assert_advective(operator)

Refuse a structure function that is not a cross-field moment, so cannot be an advective one.

A flux relation consumes `⟨δφ δ𝓐_φ⟩` for a quantity `φ` and its advection `𝓐_φ = u·∇φ`, which is a
moment across two fields. The diagonal `(a, a)` is a variance — `VectorDotSFType(1,1)` is `S2SF` —
and carries no flux.

Whether the second field really holds the advection of the first is the caller's construction, not
something a moment can report; this checks only that two distinct fields were asked for.
"""
function assert_advective(op::Union{SFT.VectorDotStructureFunctionType,
                                    SFT.ScalarDotStructureFunctionType})
    op.a == op.b && throw(ArgumentError(
        "$(nameof(typeof(op)))($(op.a), $(op.b)) is a diagonal moment, which is a variance and not " *
        "a flux. A flux relation needs two distinct fields, a quantity and its advection.",
    ))
    return nothing
end

assert_advective(op) = throw(ArgumentError(
    "$(nameof(typeof(op))) is not a cross-field moment. A spectral flux follows from an " *
    "advective structure function ⟨δφ δ𝓐_φ⟩, built with `VectorDotSFType(a, b)` or " *
    "`ScalarDotSFType(a, b)` over a field carrying the quantity and its advection as two fields. " *
    "The third-order routes are the `spectral_flux` methods on `S3SFType`, `L3SFType` (with `S3`) " *
    "and `MixedSFType{1,0,2}`.",
))

"""
    spectral_flux(operator, separations, values, wavenumbers)

Interscale flux at each of `wavenumbers`, from an advective structure function sampled at
`separations`.

```
Π_K = -(K/2) ∫₀^∞ SF_A(r) J₁(Kr) dr
```

`SF_A` is the advective structure function averaged over the directions of the separation, so the
same uniform-direction assumption [`isotropic_spectrum`](@ref) carries applies here. The relation
itself assumes no isotropy of the flow, which is what these estimators are for. The integral is the
trapezoid rule over the samples from the origin, where the integrand vanishes, to the last separation.

`J₁` peaks at `Kr ≈ 1.84`, so the flux at `K` is weighted mostly by separations near `1.84/K`.

The companion relations on third-order structure functions are the methods on `S3SFType`,
`L3SFType` and `MixedSFType{1,0,2}`, each with the boundary term its integration by parts leaves at
the last separation.
"""
function spectral_flux(
    operator, separations::AbstractVector, values::AbstractVector,
    wavenumbers::AbstractVector,
)
    assert_advective(operator)
    FT = _flux_samples(separations, values)
    integral = _flux_integrals((K, r) -> bessel_kernel(Val(1), K * r), separations, values, wavenumbers, FT)
    return _on(integral, FT, wavenumbers) .* integral ./ -2
end

function spectral_flux(sf::Union{SFO.StructureFunction, SFO.StructureFunctionSumsAndCounts}, wavenumbers::AbstractVector)
    r, vals, keep = _binned(sf)
    isempty(keep) && throw(ArgumentError("no bin holds a structure function value to transform"))
    return spectral_flux(sf.operator, r[keep], _take(vals, keep), wavenumbers)
end

# One sorted abscissa with every series sampled on it; returns the common float type.
function _flux_samples(separations::AbstractVector, series::AbstractVector...)
    for v in series
        length(v) == length(separations) || throw(DimensionMismatch(
            "separations and values must agree in length; got $(length(separations)) and $(length(v))",
        ))
    end
    issorted(collect(separations)) || throw(ArgumentError("separations must be sorted"))
    isempty(separations) && throw(ArgumentError("no separation to integrate over"))
    return float(promote_type(eltype(separations), map(eltype, series)...))
end

"""Weights of the trapezoid rule over the sorted samples `r` from the origin, the interval before the first sample a
triangle with the integrand zero at the origin: `∫₀^R f dr ≈ Σ_i t_i f(r_i)`."""
function _trapezoid_weights(r::AbstractVector)
    n = length(r)
    return [((i == n ? r[n] : r[i + 1]) - (i == 1 ? zero(eltype(r)) : r[i - 1])) / 2 for i in 1:n]
end

"""`∫₀^R values(r) g(K, r) dr` at each wavenumber `K` by the trapezoid rule from the origin, in the array family of
`values`; `g` is the kernel with its powers of `r`."""
function _flux_integrals(g, separations::AbstractVector, values::AbstractVector, wavenumbers::AbstractVector,
                         ::Type{FT}) where {FT}
    r = FT.(collect(separations))
    return _kernel_transform(g, r, _trapezoid_weights(r), values, wavenumbers, FT)
end

# Π_K = −(K²/4) ∫₀^R S(r) J₂(Kr) dr − (K/4) S(R) J₁(KR), for S = ⟨δu_L ‖δu‖²⟩ or ⟨δu_L (δθ)²⟩.
function _j2_flux(separations::AbstractVector, S::AbstractVector, wavenumbers::AbstractVector)
    FT = _flux_samples(separations, S)
    R, SR = FT(last(collect(separations))), FT(only(_tail(S, 1)))
    integral = _flux_integrals((K, r) -> bessel_kernel(Val(2), K * r), separations, S, wavenumbers, FT)
    K = _on(integral, FT, wavenumbers)
    boundary = _on(integral, FT, [FT(K0) * SR * bessel_kernel(Val(1), FT(K0) * R) / 4 for K0 in wavenumbers])
    return .-K .^ 2 .* integral ./ 4 .- boundary
end

"""
    spectral_flux(::S3SFType, separations, S3, wavenumbers)

Interscale energy flux of a two-dimensional isotropic flow from `S3 = ⟨δu_L ‖δu‖²⟩`, sampled at
`separations` up to `R`:

```
Π_K = -(K²/4) ∫₀^R S3(r) J₂(Kr) dr - (K/4) S3(R) J₁(KR)
```

The `J₁` relation integrated by parts with `SF_A = (1/2r) d(r S3)/dr`, the isotropic divergence of
`⟨δu ‖δu‖²⟩`. The boundary term at `R` does not vanish at any finite `R` and is kept. `J₂` peaks at
`Kr ≈ 3.05`, so the flux at `K` is reported on mostly by separations near `3.05/K`.
"""
spectral_flux(::SFT.S3SFType, separations::AbstractVector, S3::AbstractVector, wavenumbers::AbstractVector) =
    _j2_flux(separations, S3, wavenumbers)

"""
    spectral_flux(::MixedSFType{1,0,2}, separations, S, wavenumbers)

Interscale flux of the variance of a scalar `θ` (enstrophy, for the vorticity) in a two-dimensional
isotropic flow from `S = ⟨δu_L (δθ)²⟩`, sampled at `separations` up to `R`:

```
Π_K = -(K²/4) ∫₀^R S(r) J₂(Kr) dr - (K/4) S(R) J₁(KR)
```

The `J₁` relation on `⟨δθ δ𝓐_θ⟩ = (1/2r) d(r S)/dr` integrated by parts, boundary term kept.
"""
spectral_flux(::SFT.MixedStructureFunctionType{1, 0, 2}, separations::AbstractVector, S::AbstractVector,
              wavenumbers::AbstractVector) = _j2_flux(separations, S, wavenumbers)

"""
    spectral_flux(::L3SFType, separations, L3, S3, wavenumbers)

Interscale energy flux of a two-dimensional isotropic flow from `L3 = ⟨δu_L³⟩`, with `S3 = ⟨δu_L ‖δu‖²⟩`
on the same `separations` for the boundary term:

```
Π_K = -(K³/12) ∫₀^R L3(r) J₃(Kr) r dr - (K²/12) [ R L3(R) J₂(KR) + (3/K) S3(R) J₁(KR) ]
```

The `S3` relation integrated by parts once more with `S3 = (1/3r²) d(r³ L3)/dr`, which holds for an
isotropic incompressible two-dimensional flow. Both boundary terms are kept. `J₃` peaks at `Kr ≈ 4.20`.
"""
function spectral_flux(::SFT.L3SFType, separations::AbstractVector, L3::AbstractVector, S3::AbstractVector,
                       wavenumbers::AbstractVector)
    FT = _flux_samples(separations, L3, S3)
    R, LR, SR = FT(last(collect(separations))), FT(only(_tail(L3, 1))), FT(only(_tail(S3, 1)))
    integral = _flux_integrals((K, r) -> bessel_kernel(Val(3), K * r) * r, separations, L3, wavenumbers, FT)
    K = _on(integral, FT, wavenumbers)
    boundary = _on(integral, FT, [begin
        k = FT(K0)
        k^2 * (R * LR * bessel_kernel(Val(2), k * R) + 3 * SR * bessel_kernel(Val(1), k * R) / k) / 12
    end for K0 in wavenumbers])
    return .-K .^ 3 .* integral ./ 12 .- boundary
end

"""
    spectral_flux(L3, S3, wavenumbers)

The `L3` route on two result objects sharing their distance bins, over the bins both hold.
"""
function spectral_flux(L3::SFO.AbstractStructureFunction, S3::SFO.AbstractStructureFunction,
                       wavenumbers::AbstractVector)
    L3.operator isa SFT.L3SFType || throw(ArgumentError(
        "the first structure function must be ⟨δu_L³⟩ (L3SFType); got $(nameof(typeof(L3.operator)))",
    ))
    S3.operator isa SFT.S3SFType || throw(ArgumentError(
        "the second structure function must be ⟨δu_L ‖δu‖²⟩ (S3SFType); got $(nameof(typeof(S3.operator)))",
    ))
    r, vL, vS = _paired_bins(L3, S3)
    return spectral_flux(L3.operator, r, vL, vS, wavenumbers)
end

"""
    enstrophy_flux(operator::VectorDotSFType, separations, SF_Au, wavenumbers)
    enstrophy_flux(result, wavenumbers)

Interscale enstrophy flux of a two-dimensional isotropic flow from the velocity's advective structure
function `SF_Au = ⟨δu · δ𝓐_u⟩`, `𝓐_u = u·∇u`, sampled at `separations` up to `R`:

```
Π^ω_K = (K³/2) ∫₀^R SF_Au(r) [J₃(Kr) - J₁(Kr)]/2 dr + (K²/2) [ SF_Au(R) J₂(KR) + (1/K) SF_Au'(R) J₁(KR) ]
```

The `J₁` relation on `⟨δω δ𝓐_ω⟩ = -∇² SF_Au` integrated by parts twice; `[J₃ - J₁]/2 = -J₂'`. The
slope `SF_Au'(R)` is the one-sided difference of the last two samples, so the boundary term is
sensitive to noise in the last bins. `operator` must name two distinct fields, the velocity and
its advection.
"""
function enstrophy_flux(op::SFT.VectorDotStructureFunctionType, separations::AbstractVector,
                        SF_Au::AbstractVector, wavenumbers::AbstractVector)
    assert_advective(op)
    FT = _flux_samples(separations, SF_Au)
    length(separations) >= 2 || throw(ArgumentError(
        "the boundary term needs the slope of SF_Au at the last separation; give at least two",
    ))
    rs = collect(separations)
    last_two = FT.(_tail(SF_Au, 2))
    R, AR = FT(rs[end]), last_two[2]
    dA = (last_two[2] - last_two[1]) / (R - FT(rs[end - 1]))
    integral = _flux_integrals((K, r) -> (bessel_kernel(Val(3), K * r) - bessel_kernel(Val(1), K * r)) / 2,
                               separations, SF_Au, wavenumbers, FT)
    K = _on(integral, FT, wavenumbers)
    boundary = _on(integral, FT, [begin
        k = FT(K0)
        k^2 * (AR * bessel_kernel(Val(2), k * R) + dA * bessel_kernel(Val(1), k * R) / k) / 2
    end for K0 in wavenumbers])
    return K .^ 3 .* integral ./ 2 .+ boundary
end

function enstrophy_flux(sf::SFO.AbstractStructureFunction, wavenumbers::AbstractVector)
    r, vals, keep = _binned(sf)
    isempty(keep) && throw(ArgumentError("no bin holds a structure function value to transform"))
    return enstrophy_flux(sf.operator, r[keep], _take(vals, keep), wavenumbers)
end

"""
    AbstractMissingLagPolicy

What a lag-space spectrum does with a lag no pair of held cells names. `RefuseMissingLags()` throws,
since the structure function is undefined there. `ZeroDeviationAtMissingLags()` sets the
autocovariance to zero at that lag.
"""
abstract type AbstractMissingLagPolicy end
struct RefuseMissingLags <: AbstractMissingLagPolicy end
struct ZeroDeviationAtMissingLags <: AbstractMissingLagPolicy end

"""
    gridded_spectrum(u, schedule, ::Val{D}, spectral_backend; valid, weights, taper, missing_lags) -> (wavenumbers, density)

Spectral density of a gridded field, by transforming its structure function over the whole lag
space.

No direction is averaged over and no separation is binned, so this carries none of the angular
assumption [`isotropic_spectrum`](@ref) makes. It reads the field only through its structure function
and the variance of its held cells, so it accepts a field with cells missing.

On a complete periodic grid the result is the field's own spectrum to round-off. With cells missing,
or on a bounded direction, it is an **estimate**: the exact transform of the exact masked structure
function, equal to the complete field's spectrum in expectation, with a statistical error set by the
pair count behind each lag. A bounded direction is padded so the transform of the lags `|h| < n` is
their linear transform; `wavenumbers` then has the padded length along it. `taper` weights the lags
(see [`AbstractTaper`](@ref)) and `missing_lags` says what to do with a lag no held pair names (see
[`AbstractMissingLagPolicy`](@ref)). `weights`, one per cell, makes every pair sum and pair count a
`w_k w_kp`-weighted one and the variance a weighted variance, so the estimate is that of the weighted
structure function.

`wavenumbers` is one angular-wavenumber vector per grid direction; `density` is the `Dg`-dimensional
array over those, on the same convention as [`isotropic_spectrum`](@ref) — integrating it over
`d^D k` returns the variance of the held cells, so `sum(density) * prod(step)` does exactly, with
`step` the wavenumber spacing of each direction. The `k = 0` value is the sum of the weighted
autocovariance over the lags: zero to round-off on a complete periodic grid, where no pair sees the
mean, and the estimator's own low-wavenumber value otherwise.
"""
function gridded_spectrum(u, schedule, ::Val{D}, spectral_backend; valid = AllValid(), weights = nothing,
                          taper::AbstractTaper = NoTaper(),
                          missing_lags::AbstractMissingLagPolicy = RefuseMissingLags()) where {D}
    throw(ArgumentError(
        "no method computes a gridded spectrum for $(nameof(typeof(schedule))) with " *
        "$(nameof(typeof(spectral_backend))). The AbstractFFTs extension supplies one " *
        "combination, `UniformLagSchedule` with a fast-Fourier tag: reaching this message with " *
        "that schedule means the extension is not loaded (`using FFTW` on CPU), and reaching it " *
        "with any other means the spectrum has no method for that schedule.",
    ))
end

"""
    shell_average(wavenumbers, density, edges)

Radially bin a `Dg`-dimensional spectral density onto `edges`, returning `(midpoints, E)` with `E`
the shell-integrated spectrum: summing `E` over the shells returns what summing `density` over the
wavenumber cells returns.
"""
function shell_average(wavenumbers::NTuple{Dg, <:AbstractVector}, density::AbstractArray{FT, Dg},
                       edges::AbstractVector) where {FT, Dg}
    nb = length(edges) - 1
    acc = zeros(FT, nb)
    be = digitize_plan(edges)
    width = FT[be[b + 1] - be[b] for b in 1:nb]
    @inbounds for I in CartesianIndices(density)
        k = sqrt(sum(abs2, ntuple(d -> wavenumbers[d][I[d]], Val(Dg))))
        b = searchsortedlast(be, k)
        (1 <= b <= nb) || continue
        acc[b] += density[I]
    end
    mids = FT[(be[b] + be[b + 1]) / 2 for b in 1:nb]
    return mids, acc ./ width
end
