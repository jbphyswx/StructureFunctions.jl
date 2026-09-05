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
    "`ScalarDotSFType(a, b)` over a field carrying the quantity and its advection as two channels.",
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
itself assumes no isotropy of the flow, which is what these estimators are for.

The kernel's first peak sets which separations carry the most weight at a given wavenumber: `J₁`
peaks at `Kr ≈ 1.84`, so the flux at `K` is reported on mostly by separations near `1.84/K`. That is
a statement about weighting, not about where `Π` itself is largest — the explicit factor of `K`
means `|Π|` keeps growing with `K` for a fixed feature.

This is the `J₁` relation, which takes the advective structure function. The companion relations
built on third-order structure functions use `J₂` and `J₃` with their own prefactors and carry
boundary terms that do not generally vanish, so they are not reachable by swapping the kernel order
here.
"""
function spectral_flux(
    operator, separations::AbstractVector, values::AbstractVector,
    wavenumbers::AbstractVector,
)
    assert_advective(operator)
    length(separations) == length(values) || throw(DimensionMismatch(
        "separations and values must agree in length; got $(length(separations)) and $(length(values))",
    ))
    issorted(separations) || throw(ArgumentError("separations must be sorted"))

    FT = float(promote_type(eltype(separations), eltype(values), eltype(wavenumbers)))
    out = Vector{FT}(undef, length(wavenumbers))
    @inbounds for (j, K) in pairs(wavenumbers)
        acc = zero(FT)
        for i in eachindex(separations)
            acc += FT(values[i]) * bessel_kernel(Val(1), K * FT(separations[i])) *
                   _quad_width(separations, i)
        end
        out[j] = -FT(K) * acc / 2
    end
    return out
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

"""
    gridded_spectrum(u, schedule, ::Val{D}, spectral_backend; valid) -> (wavenumbers, density)

Spectral density of a gridded field, by transforming its structure function over the whole lag
space.

No direction is averaged over and no separation is binned, so this carries none of the angular
assumption [`isotropic_spectrum`](@ref) makes and is exact on a rectilinear grid. It reads the field
only through its structure function, so it accepts a field with cells missing, which is the case
where the field's own transform is meaningless while the pair average is still unbiased.

`wavenumbers` is one angular-wavenumber vector per grid direction; `density` is the `Dg`-dimensional
array over those, on the same convention as [`isotropic_spectrum`](@ref) — integrating it over
`d^D k` returns the variance, so `sum(density) * prod(step)` does, with `step` the wavenumber
spacing of each direction.
"""
function gridded_spectrum(u, schedule, ::Val{D}, spectral_backend; valid = AllValid()) where {D}
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
