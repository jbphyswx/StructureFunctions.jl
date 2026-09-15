# The soft-binned pair statistic of scattered points: type-1 non-uniform FFTs of the masked, weighted
# monomials onto a periodic mode grid make the transform engine's lag-space products the pair sums
# with the mode set's periodic kernel in place of a hard lag.

"""
    ScatteredModesSchedule(points, r_max, modes; taper = NoTaper())

Scattered points `(Dg, N)` in the box that is their extent padded by `r_max` along every direction,
so that no pair within `r_max` wraps, mapped to `[0, 2π)^Dg`, with a periodic mode grid of `modes`
per direction.

The transform engine takes the forward transforms of the masked, weighted monomials as type-1
non-uniform FFTs onto that grid, so every lag-space product is the pair sum
`Σ_ij w_i w_j m_i m_j μ_i ν_j K(θ_j − θ_i − 2πh/M)` with the periodic kernel
`K(φ) = Π_d (1/M_d) Σ_k b_k e^{i k φ_d}` of the mode set in place of a delta at lag `h`: a soft bin of
width about `L_d/M_d`, whose sidelobes `taper` trades for width (`NoTaper()` is the Dirichlet kernel;
`GaussianTaper(σ)`, `σ` a length, weights mode `k` by `exp(−|k|²σ²/2)`). The statistic is **not
exact**: it converges to the hard-binned pair sum as the modes grow, and no finite mode set reproduces
it. Its pair count is the kernel-weighted pair mass, a floating-point number, so results carry
[`ModeBinEdges`](@ref) and need floating-point counts. Every point's pair with itself deposits half a
unit of that mass on the smallest lags, as the harmonic route's self term does; bins that start above
the kernel's width exclude it. The accuracy of the transforms themselves is the provider's, set on its
tag: [`NonuniformFFTsSpectralBackend`](@ref) or [`FINUFFTSpectralBackend`](@ref).
"""
struct ScatteredModesSchedule{Dg, T, X <: AbstractMatrix{T}, B <: AbstractTaper} <: AbstractSeparableSchedule
    points::X
    origin::NTuple{Dg, T}
    box::NTuple{Dg, T}
    modes::NTuple{Dg, Int}
    taper::B
end

function ScatteredModesSchedule(points::AbstractMatrix{T}, r_max::Real, modes::NTuple{Dg, <:Integer};
                                taper::AbstractTaper = NoTaper()) where {T, Dg}
    size(points, 1) == Dg || throw(DimensionMismatch(
        "points have $(size(points, 1)) coordinates, modes name $Dg directions",
    ))
    all(m -> m >= 2, modes) || throw(ArgumentError("every direction needs at least two modes; got $modes"))
    (isfinite(r_max) && r_max > 0) || throw(ArgumentError(
        "r_max must be finite and positive: the box is the points' extent padded by it",
    ))
    taper isa Union{NoTaper, GaussianTaper} || throw(ArgumentError(
        "$(nameof(typeof(taper))) has no mode-space form; use NoTaper() or GaussianTaper(σ)",
    ))
    origin = ntuple(d -> T(minimum(view(points, d, :))), Val(Dg))
    box = ntuple(d -> T(maximum(view(points, d, :)) - origin[d] + r_max), Val(Dg))
    return ScatteredModesSchedule{Dg, T, typeof(points), typeof(taper)}(points, origin, box, Int.(modes), taper)
end

@inline n_cells(s::ScatteredModesSchedule) = size(s.points, 2)
@inline grid_dimension(::ScatteredModesSchedule{Dg}) where {Dg} = Dg
@inline uniform_axes(s::ScatteredModesSchedule{Dg}) where {Dg} =
    UniformLagSchedule(s.modes, ntuple(d -> s.box[d] / s.modes[d], Val(Dg)), ntuple(_ -> true, Val(Dg)))
@inline n_slabs(::ScatteredModesSchedule) = 1
@inline enumerated_pairs(::ScatteredModesSchedule, r_max) = ((1, 1),)
@inline separable_layout(::ScatteredModesSchedule, data, valid, weights) = (data, valid, weights)
@inline lag_limits(s::ScatteredModesSchedule, r_max) = lag_limits(uniform_axes(s), r_max)
@inline lag_limits(s::ScatteredModesSchedule, I, J, r_max) = lag_limits(uniform_axes(s), r_max)
@inline lag_transport(::ScatteredModesSchedule) = IdentityTransport()
@inline lag_frame(s::ScatteredModesSchedule, I, J, h, vD::Val, vV::Val, vK::Val) =
    lag_frame(uniform_axes(s), I, J, h, vD, vV, vK)

"""Whether a schedule's pair counts are a kernel-weighted mass read from the transforms' count column."""
@inline _soft_binned(::AbstractSeparableSchedule) = false
@inline _soft_binned(::ScatteredModesSchedule) = true

"""Points in `[0, 2π)` along each direction, as the non-uniform FFT wants them."""
@inline function mode_coordinates(s::ScatteredModesSchedule{Dg, T}, d::Integer) where {Dg, T}
    return T(2π) .* (view(s.points, d, :) .- s.origin[d]) ./ s.box[d]
end

"""The angular wavenumber squared of mode `m` (integers, one per direction) of the schedule's box."""
@inline function mode_wavenumber2(s::ScatteredModesSchedule{Dg, T}, m::NTuple{Dg}) where {Dg, T}
    k2 = zero(T)
    for d in 1:Dg
        k = T(2π) * m[d] / s.box[d]
        k2 += k * k
    end
    return k2
end

"""
    mode_taper_weights(schedule, FT, dims, to) -> Array or nothing

The schedule's taper on the real-to-complex half spectrum of size `dims` — mode integers `0:M₁÷2` along
the first direction, FFT order along the rest — in the array family `to` returns; `nothing` for `NoTaper()`.
"""
function mode_taper_weights(s::ScatteredModesSchedule{Dg, T}, ::Type{FT}, dims::NTuple{Dg, Int}, to) where {Dg, T, FT}
    s.taper isa NoTaper && return nothing
    b = Array{FT}(undef, dims)
    for I in CartesianIndices(b)
        m = ntuple(d -> _mode_integer(I[d], s.modes[d], d == 1), Val(Dg))
        b[I] = FT(mode_taper(s.taper, mode_wavenumber2(s, m)))
    end
    return to(b)
end

"""The signed mode integer at position `i` of a direction of `M` modes: `i − 1` on the half-spectrum direction, FFT order on a full one."""
@inline _mode_integer(i::Int, M::Int, half::Bool) = (half || i - 1 <= (M - 1) ÷ 2) ? i - 1 : i - 1 - M

_no_lag_sweep(::ScatteredModesSchedule) = throw(ArgumentError(
    "a ScatteredModesSchedule has no lags to sweep: its pairs are summed by a non-uniform FFT. Pass a non-uniform " *
    "FFT tag — NonuniformFFTsSpectralBackend() or FINUFFTSpectralBackend(), or NonUniformFastFourierTransformSpectralBackend() " *
    "with one provider loaded — with the provider and an AbstractFFTs implementation loaded, or call the point entry " *
    "without a schedule for the exact pair loop.",
))

function gridded_lag_sweep!(
    sums::AbstractVector{OT}, counts::AbstractVector{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::ScatteredModesSchedule, dist_be, ::Val{D}, ::Val{V}, ::Val{K}; kwargs...,
) where {OT, CT, D, V, K}
    _no_lag_sweep(s)
end

function gridded_lag_sweep!(
    sums::AbstractMatrix{OT}, counts::AbstractMatrix{CT}, sf::SFT.AbstractPairwiseStructureFunctionType,
    data::AbstractMatrix, s::ScatteredModesSchedule, dist_be, axis_be, ::Val{D}, ::Val{V}, ::Val{K}; kwargs...,
) where {OT, CT, D, V, K}
    _no_lag_sweep(s)
end

# ---------------------------------------------------------------------------------------------------
# The providers of the non-uniform FFT, as tags
# ---------------------------------------------------------------------------------------------------

"""
    NonuniformFFTsSpectralBackend(; half_support = 8)

The non-uniform FFT tag that names NonuniformFFTs.jl as the provider, on the CPU or a device; `using
NonuniformFFTs` supplies it. `half_support` is the spreading kernel's half-support in oversampled grid
points, which sets the transforms' accuracy (8 gives about `1e-12` in `Float64`) and the least number
of modes a direction of the schedule may have.
"""
struct NonuniformFFTsSpectralBackend <: SB.AbstractNonUniformFastFourierTransformSpectralBackend
    half_support::Int
end

function NonuniformFFTsSpectralBackend(; half_support::Integer = 8)
    half_support >= 1 || throw(ArgumentError("half_support must be positive; got $half_support"))
    return NonuniformFFTsSpectralBackend(Int(half_support))
end

"""
    FINUFFTSpectralBackend(; tolerance = 1e-12)

The non-uniform FFT tag that names FINUFFT.jl as the provider — cuFINUFFT for points on a CUDA device;
`using FINUFFT` supplies it, with `using CUDA` for the device. `tolerance` is the relative accuracy the
transforms are asked for.
"""
struct FINUFFTSpectralBackend <: SB.AbstractNonUniformFastFourierTransformSpectralBackend
    tolerance::Float64
end

function FINUFFTSpectralBackend(; tolerance::Real = 1e-12)
    0 < tolerance < 1 || throw(ArgumentError("tolerance must lie in (0, 1); got $tolerance"))
    return FINUFFTSpectralBackend(Float64(tolerance))
end

"""The provider tags the plain non-uniform FFT tag may resolve to."""
const NUFFT_PROVIDERS = (NonuniformFFTsSpectralBackend, FINUFFTSpectralBackend)

"""Whether a provider tag's extension is loaded; each extension answers `true` for its own tag."""
_nufft_loaded(::Type) = false

_nufft_package(::Type{NonuniformFFTsSpectralBackend}) = "NonuniformFFTs"
_nufft_package(::Type{FINUFFTSpectralBackend}) = "FINUFFT"

"""
    nufft_provider(tag) -> provider tag

The provider a non-uniform FFT tag names: a provider tag names itself, and the plain
`NonUniformFastFourierTransformSpectralBackend()` names the one loaded provider's default, refusing by
name when none or both are loaded.
"""
nufft_provider(tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend) = tag

function nufft_provider(::SB.NonUniformFastFourierTransformSpectralBackend)
    loaded = filter(_nufft_loaded, NUFFT_PROVIDERS)
    length(loaded) == 1 && return loaded[1]()
    isempty(loaded) && throw(ArgumentError(
        "the non-uniform FFT route needs a provider: `using NonuniformFFTs` or `using FINUFFT`.",
    ))
    throw(ArgumentError(
        "both non-uniform FFT providers are loaded, so NonUniformFastFourierTransformSpectralBackend() names " *
        "neither; pass NonuniformFFTsSpectralBackend() or FINUFFTSpectralBackend().",
    ))
end

"""
    nufft_monomial_transforms(tag, schedule::ScatteredModesSchedule, data, valid, weights, keys, ::Val{Pm}; to = identity) -> Vector

Type-1 non-uniform FFTs of the masked, weighted monomials `keys` of the packed point field onto the
schedule's mode grid, one array per key in the real-to-complex half-spectrum layout
`(M₁ ÷ 2 + 1, M₂, …)`, each weighted by the schedule's taper; the arrays in the family `to` returns.
`tag` is a provider tag, or the plain tag resolved by `nufft_provider`; supplied by the provider's
extension.
"""
nufft_monomial_transforms(tag::SB.NonUniformFastFourierTransformSpectralBackend, args...; kwargs...) =
    nufft_monomial_transforms(nufft_provider(tag), args...; kwargs...)

nufft_monomial_transforms(tag::SB.AbstractNonUniformFastFourierTransformSpectralBackend, args...; kwargs...) =
    throw(ArgumentError(_nufft_missing(typeof(tag))))

_nufft_missing(::Type{T}) where {T <: Union{NonuniformFFTsSpectralBackend, FINUFFTSpectralBackend}} =
    "$(nameof(T)) needs `using $(_nufft_package(T))`."
_nufft_missing(::Type{T}) where {T} = "no extension computes non-uniform FFTs for $(nameof(T))."

# A non-uniform FFT tag with no transform loaded: the route's inverse transforms come from the AbstractFFTs extension.
_no_nufft_route() = throw(ArgumentError(
    "the non-uniform FFT route needs an AbstractFFTs implementation (`using FFTW` on CPU) for the inverse " *
    "transforms, and a provider: `using NonuniformFFTs` or `using FINUFFT`.",
))

gridded_sweep!(::AbstractVector, ::AbstractVector, sf, ::AbstractMatrix, schedule, distance_bins, ::Val{D}, ::Val{V},
               ::Val{K}, ::SB.AbstractNonUniformFastFourierTransformSpectralBackend; kwargs...) where {D, V, K} =
    _no_nufft_route()

gridded_sweep!(::AbstractMatrix, ::AbstractMatrix, sf, ::AbstractMatrix, schedule, distance_bins, axis_bins, ::Val{D},
               ::Val{V}, ::Val{K}, ::SB.AbstractNonUniformFastFourierTransformSpectralBackend; kwargs...) where {D, V, K} =
    _no_nufft_route()

"""
The monomial `w_k Π_c data[c, k]` over the cells, zero where the cell holds nothing: selected by the
mask, never multiplied by it, since an empty cell may hold NaN. The weights are finite.
"""
function _held_monomial_vector(data::AbstractMatrix, valid, weights, key::Tuple, ::Type{FT}) where {FT}
    out = fill!(similar(parent(data), FT, size(data, 2)), one(FT))
    for c in key
        c == 0 && continue
        out .*= view(data, c, :)
    end
    weights isa NoWeights || (out .*= weights)
    valid isa AllValid || (out .= ifelse.(valid, out, zero(FT)))
    return out
end

"""
    calculate_structure_function(sf, schedule::ScatteredModesSchedule, u, distance_bins, spectral_backend; weights, backend, output_type, verbose)

The soft-binned structure function of scattered points by non-uniform FFT. `u` is `(D, N)` over the
`N` points of `schedule`, or a `Fields` bundle over them; `spectral_backend` is a non-uniform FFT tag —
[`NonuniformFFTsSpectralBackend`](@ref), [`FINUFFTSpectralBackend`](@ref), or the plain
`NonUniformFastFourierTransformSpectralBackend()` when exactly one provider is loaded. Counts are
`Float64`, the kernel-weighted pair mass, and the result's `distance` is a [`ModeBinEdges`](@ref)
carrying the schedule. See [`ScatteredModesSchedule`](@ref) for what is and is not exact.
"""
function calculate_structure_function(
    sf::SFT.AbstractPairwiseStructureFunctionType, s::ScatteredModesSchedule, u::Union{AbstractArray, CH.Fields},
    distance_bins::AbstractVector, spectral_backend;
    weights = nothing, backend::CB.AbstractExecutionBackend = CB.AutoBackend(),
    output_type::Type{OT} = SFO.StructureFunction, verbose::Bool = true,
) where {OT}
    data, vD, vV, vK = _packed(u)
    N = n_cells(s)
    size(data, 2) == N || throw(DimensionMismatch("u covers $(size(data, 2)) points, the schedule $N"))
    valid = field_validity(data)
    nb = n_histogram_bins(squared_digitize_plan(distance_bins))
    sums = zeros(float(eltype(data)), nb)
    counts = zeros(Float64, nb)
    verbose && @info "soft-binned structure function by non-uniform FFT: $N points onto $(s.modes) modes"
    gridded_sweep!(sums, counts, sf, data, s, distance_bins, vD, vV, vK, spectral_backend; valid, weights, backend)
    return _finalize(SFO.StructureFunctionSumsAndCounts(sf, ModeBinEdges(distance_bins, s), sums, counts), OT)
end
