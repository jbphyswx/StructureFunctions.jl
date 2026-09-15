# Transforms between a binned structure function and spectral space.

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

Valid only if the field is **second-order stationary**. A field that is merely *intrinsically*
stationary has a variogram but need not have a finite variance, and then no covariance exists to
compute; supplying a number anyway produces a curve with no referent.
"""
function covariance(sf::SFO.StructureFunction, variance::Real)
    assert_variogram(sf.operator)
    r = midpoints(sf.distance)
    keep = findall(isfinite, sf.values)
    isempty(keep) && throw(ArgumentError("no finite structure function value"))
    return collect(r)[keep], [variance - sf.values[i] / 2 for i in keep]
end

function covariance(sf::SFO.StructureFunctionSumsAndCounts, variance::Real)
    assert_variogram(sf.operator)
    r = midpoints(sf.distance)
    keep = findall(>(0), sf.counts)
    isempty(keep) && throw(ArgumentError("every bin is empty"))
    return collect(r)[keep], [variance - sf.sums[i] / sf.counts[i] / 2 for i in keep]
end

"""
    covariance_matrix(points, separations, C; metric, check_posdef = true, posdef_rtol = 1e-6)

Covariance matrix over `points`, evaluating the covariance function `(separations, C)` at each pair
distance by linear interpolation and holding it constant outside the sampled range.

A covariance matrix must be positive semi-definite, and one built this way need not be, so it is
checked rather than assumed. Two different things trip the check and the message distinguishes them:
a covariance function that is not a valid kernel at all, and one that is valid but sampled too
coarsely — interpolating a kernel does not preserve positive-definiteness, and the error falls as
the square of the separation spacing. `check_posdef = false` returns the matrix regardless.
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
    helmholtz_spectra(h, wavenumbers; rotational_asymptote, divergent_asymptote)

Rotational and divergent kinetic-energy spectra from a two-dimensional Helmholtz decomposition.

Each component of the decomposition is the share of the second-order trace carried by one of the two
fields, so each transforms to a spectrum by the same route the trace does — in two dimensions, with
the `J₀` kernel, hence `Bessels`. Returns a `NamedTuple` of the two densities, on the convention
[`isotropic_spectrum`](@ref) documents.

Bins holding no pair are dropped. The asymptotes default to each component's own largest value.
"""
function helmholtz_spectra(
    h::SFO.HelmholtzDecomposition2D, wavenumbers::AbstractVector;
    rotational_asymptote = nothing, divergent_asymptote = nothing,
)
    r = collect(midpoints(h.distance_bins))
    rot = _component_spectrum(SFT.RotationalSecondOrderStructureFunctionType(), r,
                              h.rotational_sums, h.rotational_counts, wavenumbers,
                              rotational_asymptote)
    div = _component_spectrum(SFT.DivergentSecondOrderStructureFunctionType(), r,
                              h.divergent_sums, h.divergent_counts, wavenumbers,
                              divergent_asymptote)
    return (rotational = rot, divergent = div)
end

"""
    helmholtz_spectra(L2, T2, wavenumbers; asymptote)

Rotational and divergent spectral densities of a two-dimensional field from its longitudinal and
transverse second-order structure functions, binned on the same edges.

The sum and the difference of the two projections transform separately,

```math
P_E + P_B = -\\frac{1}{4π} ∫_0^∞ (D_{LL} + D_{TT} - a)\\, J_0(kr)\\, r\\, dr, \\qquad
P_E - P_B = \\frac{1}{4π} ∫_0^∞ (D_{LL} - D_{TT})\\, J_2(kr)\\, r\\, dr,
```

with `E` the divergent (gradient) part and `B` the rotational (curl) part, on the convention that a
density integrates over `d²k` to the variance. The first line is [`isotropic_spectrum`](@ref) of the
trace and needs the trace's large-separation limit `a`, which defaults to its largest value; the
second needs none, since `D_LL − D_TT` decays on its own. Both kernels need `Bessels`.

Its error is the truncation of the Hankel integrals at the last separation, which is a different
error from the one the real-space route through [`helmholtz_decompose_2d`](@ref) carries.
"""
function helmholtz_spectra(
    L2::SFO.AbstractStructureFunction, T2::SFO.AbstractStructureFunction, wavenumbers::AbstractVector;
    asymptote = nothing,
)
    _assert_projection(L2.operator, 2, 0, "longitudinal")
    _assert_projection(T2.operator, 0, 2, "transverse")
    r, dll, dtt = _paired_bins(L2, T2)
    asym = asymptote === nothing ? maximum(dll .+ dtt) : asymptote
    total = isotropic_spectrum(SFT.S2SFType(), r, dll .+ dtt, wavenumbers, Val(2); asymptote = asym)
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

"""Abscissa and values of a result, restricted to bins that hold a value."""
function _binned(sf::SFO.StructureFunction)
    keep = findall(isfinite, sf.values)
    return collect(midpoints(sf.distance)), collect(sf.values), keep
end

function _binned(sf::SFO.StructureFunctionSumsAndCounts)
    keep = findall(>(0), sf.counts)
    vals = [c > 0 ? s / c : oftype(float(s), NaN) for (s, c) in zip(sf.sums, sf.counts)]
    return collect(midpoints(sf.distance)), vals, keep
end

# Two results on one set of edges, reduced to the bins both hold.
function _paired_bins(a::SFO.AbstractStructureFunction, b::SFO.AbstractStructureFunction)
    a.distance == b.distance || throw(ArgumentError(
        "the two structure functions must share their distance bins",
    ))
    r, va, ka = _binned(a)
    _, vb, kb = _binned(b)
    keep = intersect(ka, kb)
    isempty(keep) && throw(ArgumentError("no bin holds a value in both structure functions"))
    return r[keep], va[keep], vb[keep]
end

"""`∫₀^∞ f(r) J_N(kr) r dr` at each wavenumber, by the same quadrature as [`isotropic_spectrum`](@ref)."""
function _hankel(::Val{N}, separations::AbstractVector, values::AbstractVector, wavenumbers::AbstractVector) where {N}
    FT = float(promote_type(eltype(separations), eltype(values), eltype(wavenumbers)))
    out = Vector{FT}(undef, length(wavenumbers))
    @inbounds for (j, k) in pairs(wavenumbers)
        acc = zero(FT)
        for i in eachindex(separations)
            r = FT(separations[i])
            acc += FT(values[i]) * bessel_kernel(Val(N), k * r) * r * _quad_width(separations, i)
        end
        out[j] = acc
    end
    return out
end

function _component_spectrum(op, r, sums, counts, wavenumbers, asymptote)
    keep = findall(>(0), counts)
    isempty(keep) && throw(ArgumentError("every bin of the $(nameof(typeof(op))) component is empty"))
    values = [sums[i] / counts[i] for i in keep]
    asym = asymptote === nothing ? maximum(values) : asymptote
    return isotropic_spectrum(op, r[keep], values, wavenumbers, Val(2); asymptote = asym)
end

"""
    bessel_kernel(::Val{N}, x)

Bessel function of the first kind of order `N`.

Orders 0 through 3 are what the flux relations use. Every one needs `Bessels`, which core does not
depend on, so this is the single point where that package is reached.
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

# Each Helmholtz component is the part of the second-order trace carried by one of the two fields the
# decomposition separates, so each inverts by the same route as the trace itself.
assert_invertible(::SFT.RotationalSecondOrderStructureFunctionType) = nothing
assert_invertible(::SFT.DivergentSecondOrderStructureFunctionType) = nothing

assert_invertible(op) = throw(ArgumentError(
    "$(nameof(typeof(op))) has no spectral inverse; a spectrum follows only from the second-order " *
    "trace ⟨‖δu‖²⟩ (`S2SFType`).",
))

"""
    isotropic_spectrum(operator, separations, values, wavenumbers, ::Val{D}; asymptote)

Power spectral density at each of `wavenumbers`, from a second-order structure function sampled at
`separations`.

`D(r) = 2[C(0) - C(r)]`, so the transform of `D` differs from that of `-2C` only by a constant, whose
transform is confined to `k = 0`. Every returned wavenumber must therefore be nonzero, and the
`k = 0` mode is not recoverable — no variance argument would help, because `D` does not carry it.

`asymptote` is the large-separation limit of the structure function, subtracted so the integrand
decays; it defaults to the largest value supplied.

Normalised so that integrating the density over `d^D k` returns the field's variance, which fixes
the convention: `∫₀^∞ shell_spectrum(...) dk == var(u)`.
"""
function isotropic_spectrum(
    operator, separations::AbstractVector, values::AbstractVector,
    wavenumbers::AbstractVector, ::Val{D};
    asymptote = maximum(values),
) where {D}
    assert_invertible(operator)
    length(separations) == length(values) || throw(DimensionMismatch(
        "separations and values must agree in length; got $(length(separations)) and $(length(values))",
    ))
    any(iszero, wavenumbers) && throw(ArgumentError(
        "the k = 0 mode is not recoverable from a structure function, which is blind to the mean " *
        "and to the variance; request nonzero wavenumbers only.",
    ))
    issorted(separations) || throw(ArgumentError("separations must be sorted"))

    FT = float(promote_type(eltype(separations), eltype(values), eltype(wavenumbers)))
    decaying = FT[v - asymptote for v in values]
    Ω = solid_angle(Val(D))
    out = Vector{FT}(undef, length(wavenumbers))
    @inbounds for (j, k) in pairs(wavenumbers)
        acc = zero(FT)
        for i in eachindex(separations)
            r = FT(separations[i])
            acc += decaying[i] * isotropic_kernel(Val(D), k * r) * r^(D - 1) * _quad_width(separations, i)
        end
        out[j] = -Ω * acc / (2 * (2 * FT(π))^D)
    end
    return out
end

"""
    isotropic_spectrum(result, wavenumbers, ::Val{D}; asymptote)

Spectral density from a structure function result, taking the operator, the separations and the
values from the result itself.

The abscissa is the bin representative of `result`'s edges. Bins holding no pair are dropped rather
than carried as `NaN`, which would otherwise propagate through the quadrature into every wavenumber.

The transform averages over the directions of the separation, so it assumes the pairs behind each
bin sample direction uniformly. Scattered points do; a rectilinear grid does **not**, and on gridded
data the separations available at a given `r` are biased toward the lattice axes.
"""
function isotropic_spectrum(sf::SFO.StructureFunction, wavenumbers::AbstractVector, ::Val{D};
                            kwargs...) where {D}
    r = midpoints(sf.distance)
    keep = findall(isfinite, sf.values)
    isempty(keep) && throw(ArgumentError("no finite structure function value to transform"))
    return isotropic_spectrum(sf.operator, collect(r)[keep], collect(sf.values)[keep],
                              wavenumbers, Val(D); kwargs...)
end

function isotropic_spectrum(sf::SFO.StructureFunctionSumsAndCounts, wavenumbers::AbstractVector,
                            ::Val{D}; kwargs...) where {D}
    r = midpoints(sf.distance)
    keep = findall(>(0), sf.counts)
    isempty(keep) && throw(ArgumentError("every bin is empty; nothing to transform"))
    values = [sf.sums[i] / sf.counts[i] for i in keep]
    return isotropic_spectrum(sf.operator, collect(r)[keep], values, wavenumbers, Val(D); kwargs...)
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
    C = [-π * sum(D[i] * Q[i, l + 1] for i in eachindex(D)) for l in 1:lmax]
    return (l = 1:lmax, C = C)
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
    E = Vector{Float64}(undef, lmax)
    B = Vector{Float64}(undef, lmax)
    for l in 1:lmax
        diff = π * sum((DLL[i] - DTT[i]) * Gm[i, l] for i in eachindex(DLL))
        total = 2π * sum((variance - (DLL[i] + DTT[i]) / 2) * Gp[i, l] for i in eachindex(DLL))
        E[l] = (total + diff) / 2
        B[l] = (total - diff) / 2
    end
    return (l = 1:lmax, E = E, B = B)
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
    [solid_angle(Val(D)) * k^(D - 1) * p for (k, p) in zip(wavenumbers, P)]

"""
    assert_advective(operator)

Refuse a structure function that is not a cross-channel moment, so cannot be an advective one.

A flux relation consumes `⟨δφ δ𝓐_φ⟩` for a quantity `φ` and its advection `𝓐_φ = u·∇φ`, which is a
moment across two channels. The diagonal `(a, a)` is a variance — `VectorDotSFType(1,1)` is `S2SF` —
and carries no flux.

Whether the second channel really holds the advection of the first is the caller's construction, not
something a moment can report; this checks only that two distinct channels were asked for.
"""
function assert_advective(op::Union{SFT.VectorDotStructureFunctionType,
                                    SFT.ScalarDotStructureFunctionType})
    op.a == op.b && throw(ArgumentError(
        "$(nameof(typeof(op)))($(op.a), $(op.b)) is a diagonal moment, which is a variance and not " *
        "a flux. A flux relation needs two distinct channels, a quantity and its advection.",
    ))
    return nothing
end

assert_advective(op) = throw(ArgumentError(
    "$(nameof(typeof(op))) is not a cross-channel moment. A spectral flux follows from an " *
    "advective structure function ⟨δφ δ𝓐_φ⟩, built with `VectorDotSFType(a, b)` or " *
    "`ScalarDotSFType(a, b)` over a field carrying the quantity and its advection as two channels. " *
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

The kernel's first peak sets which separations carry the most weight at a given wavenumber: `J₁`
peaks at `Kr ≈ 1.84`, so the flux at `K` is reported on mostly by separations near `1.84/K`. That is
a statement about weighting, not about where `Π` itself is largest — the explicit factor of `K`
means `|Π|` keeps growing with `K` for a fixed feature.

This is the `J₁` relation, which takes the advective structure function and assumes no isotropy of
the flow. The companion relations on third-order structure functions are the methods on
`S3SFType`, `L3SFType` and `MixedSFType{1,0,2}`, each with the boundary term its integration by
parts leaves at the last separation.
"""
function spectral_flux(
    operator, separations::AbstractVector, values::AbstractVector,
    wavenumbers::AbstractVector,
)
    assert_advective(operator)
    FT = _flux_samples(separations, values)
    return [begin
        K = FT(K0)
        -K * _flux_quadrature(r -> bessel_kernel(Val(1), K * r), separations, values, FT) / 2
    end for K0 in wavenumbers]
end

function spectral_flux(sf::SFO.StructureFunction, wavenumbers::AbstractVector)
    r = midpoints(sf.distance)
    keep = findall(isfinite, sf.values)
    isempty(keep) && throw(ArgumentError("no finite structure function value to transform"))
    return spectral_flux(sf.operator, collect(r)[keep], collect(sf.values)[keep], wavenumbers)
end

function spectral_flux(sf::SFO.StructureFunctionSumsAndCounts, wavenumbers::AbstractVector)
    r = midpoints(sf.distance)
    keep = findall(>(0), sf.counts)
    isempty(keep) && throw(ArgumentError("every bin is empty; nothing to transform"))
    return spectral_flux(sf.operator, collect(r)[keep], [sf.sums[i] / sf.counts[i] for i in keep],
                         wavenumbers)
end

# One sorted abscissa with every series sampled on it; returns the common float type.
function _flux_samples(separations::AbstractVector, series::AbstractVector...)
    for v in series
        length(v) == length(separations) || throw(DimensionMismatch(
            "separations and values must agree in length; got $(length(separations)) and $(length(v))",
        ))
    end
    issorted(separations) || throw(ArgumentError("separations must be sorted"))
    isempty(separations) && throw(ArgumentError("no separation to integrate over"))
    return float(promote_type(eltype(separations), map(eltype, series)...))
end

# ∫₀^R values(r) g(r) dr by the trapezoid rule over the samples from the origin, where every flux
# integrand vanishes; `g` is the kernel with its powers of r.
function _flux_quadrature(g, separations::AbstractVector, values::AbstractVector, ::Type{FT}) where {FT}
    acc = zero(FT)
    r0 = zero(FT)
    f0 = zero(FT)
    @inbounds for i in eachindex(separations)
        r = FT(separations[i])
        f = FT(values[i]) * g(r)
        acc += (f0 + f) * (r - r0) / 2
        r0, f0 = r, f
    end
    return acc
end

# Π_K = −(K²/4) ∫₀^R S(r) J₂(Kr) dr − (K/4) S(R) J₁(KR), for S = ⟨δu_L ‖δu‖²⟩ or ⟨δu_L (δθ)²⟩.
function _j2_flux(separations::AbstractVector, S::AbstractVector, wavenumbers::AbstractVector)
    FT = _flux_samples(separations, S)
    R, SR = FT(last(separations)), FT(last(S))
    return [begin
        K = FT(K0)
        integral = _flux_quadrature(r -> bessel_kernel(Val(2), K * r), separations, S, FT)
        -K^2 * integral / 4 - K * SR * bessel_kernel(Val(1), K * R) / 4
    end for K0 in wavenumbers]
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
    R, LR, SR = FT(last(separations)), FT(last(L3)), FT(last(S3))
    return [begin
        K = FT(K0)
        integral = _flux_quadrature(r -> bessel_kernel(Val(3), K * r) * r, separations, L3, FT)
        -K^3 * integral / 12 -
        K^2 * (R * LR * bessel_kernel(Val(2), K * R) + 3 * SR * bessel_kernel(Val(1), K * R) / K) / 12
    end for K0 in wavenumbers]
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
sensitive to noise in the last bins. `operator` must name two distinct channels, the velocity and
its advection.
"""
function enstrophy_flux(op::SFT.VectorDotStructureFunctionType, separations::AbstractVector,
                        SF_Au::AbstractVector, wavenumbers::AbstractVector)
    assert_advective(op)
    FT = _flux_samples(separations, SF_Au)
    length(separations) >= 2 || throw(ArgumentError(
        "the boundary term needs the slope of SF_Au at the last separation; give at least two",
    ))
    R, AR = FT(last(separations)), FT(last(SF_Au))
    dA = (FT(SF_Au[end]) - FT(SF_Au[end - 1])) / (R - FT(separations[end - 1]))
    return [begin
        K = FT(K0)
        integral = _flux_quadrature(r -> (bessel_kernel(Val(3), K * r) - bessel_kernel(Val(1), K * r)) / 2,
                                    separations, SF_Au, FT)
        K^3 * integral / 2 + K^2 * (AR * bessel_kernel(Val(2), K * R) + dA * bessel_kernel(Val(1), K * R) / K) / 2
    end for K0 in wavenumbers]
end

function enstrophy_flux(sf::SFO.AbstractStructureFunction, wavenumbers::AbstractVector)
    r, vals, keep = _binned(sf)
    isempty(keep) && throw(ArgumentError("no finite structure function value to transform"))
    return enstrophy_flux(sf.operator, r[keep], vals[keep], wavenumbers)
end

"""
    AbstractMissingLagPolicy

What a lag-space spectrum does with a lag no pair of held cells names. `RefuseMissingLags()` throws,
since the structure function is undefined there. `ZeroDeviationAtMissingLags()` sets the
autocovariance to zero at that lag, which is what a taper does at the lags beyond its reach.
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
and the variance of its held cells, so it accepts a field with cells missing, which is the case where
the field's own transform is meaningless while the pair average is still unbiased.

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
        "no method transforms a gridded structure function with $(typeof(spectral_backend)). " *
        "Load an AbstractFFTs implementation — `using FFTW` on CPU.",
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
    width = FT[edges[b + 1] - edges[b] for b in 1:nb]
    @inbounds for I in CartesianIndices(density)
        k = sqrt(sum(abs2, ntuple(d -> wavenumbers[d][I[d]], Val(Dg))))
        b = searchsortedlast(edges, k)
        (1 <= b <= nb) || continue
        acc[b] += density[I]
    end
    mids = FT[(edges[b] + edges[b + 1]) / 2 for b in 1:nb]
    return mids, acc ./ width
end

@inline function _quad_width(r::AbstractVector, i::Integer)
    n = length(r)
    n == 1 && return one(eltype(r))
    i == 1 && return r[2] - r[1]
    i == n && return r[n] - r[n - 1]
    return (r[i + 1] - r[i - 1]) / 2
end
