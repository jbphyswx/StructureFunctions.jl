module StructureFunctionsSpectralBackendsExt

using SpectralBackends: SpectralBackends as SB
using StructureFunctions: Calculations as SFC

# The tags that name an algorithm needing no transform. `Auto` resolves here to the lag sweep because
# that is the only summation this session has; the transform extension answers `Auto` on the concrete
# tag, which out-specialises this, and weighs the two costs.

SFC.gridded_sweep!(sums::AbstractVector, counts::AbstractVector, sf, data::AbstractMatrix, schedule,
                   distance_bins, ::Val{D}, ::Val{V}, ::Val{K}, ::SB.AbstractDirectSumSpectralBackend;
                   kwargs...) where {D, V, K} =
    SFC.gridded_lag_sweep!(sums, counts, sf, data, schedule, distance_bins, Val(D), Val(V), Val(K); kwargs...)

SFC.gridded_sweep!(sums::AbstractVector, counts::AbstractVector, sf, data::AbstractMatrix, schedule,
                   distance_bins, ::Val{D}, ::Val{V}, ::Val{K}, ::SB.AbstractAutoSpectralBackend;
                   kwargs...) where {D, V, K} =
    SFC.gridded_lag_sweep!(sums, counts, sf, data, schedule, distance_bins, Val(D), Val(V), Val(K); kwargs...)

SFC.gridded_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, data::AbstractMatrix, schedule,
                   distance_bins, axis_bins, ::Val{D}, ::Val{V}, ::Val{K},
                   ::SB.AbstractDirectSumSpectralBackend; kwargs...) where {D, V, K} =
    SFC.gridded_lag_sweep!(sums, counts, sf, data, schedule, distance_bins, axis_bins, Val(D), Val(V),
                           Val(K); kwargs...)

SFC.gridded_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, data::AbstractMatrix, schedule,
                   distance_bins, axis_bins, ::Val{D}, ::Val{V}, ::Val{K},
                   ::SB.AbstractAutoSpectralBackend; kwargs...) where {D, V, K} =
    SFC.gridded_lag_sweep!(sums, counts, sf, data, schedule, distance_bins, axis_bins, Val(D), Val(V),
                           Val(K); kwargs...)

function SFC.gridded_sweep!(sums::AbstractVector, counts::AbstractVector, sf, data::AbstractMatrix,
                            schedule, distance_bins, ::Val{D}, ::Val{V}, ::Val{K},
                            backend::SB.AbstractSpectralBackend; kwargs...) where {D, V, K}
    _no_transform_loaded(backend, schedule)
end

function SFC.gridded_sweep!(sums::AbstractMatrix, counts::AbstractMatrix, sf, data::AbstractMatrix,
                            schedule, distance_bins, axis_bins, ::Val{D}, ::Val{V}, ::Val{K},
                            backend::SB.AbstractSpectralBackend; kwargs...) where {D, V, K}
    _no_transform_loaded(backend, schedule)
end

# The harmonic route's reference provider: the direct sum over the points.
SFC.harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes::SFC.HarmonicNodes, ::Val{D}, ::Val{V},
                    ::Val{K}, ::SB.AbstractDirectSumSpectralBackend; valid = SFC.AllValid()) where {D, V, K} =
    SFC._harmonic_sweep!(sums, counts, sf, geometry, x, weights, data, nodes, Val(D), Val(V), Val(K), valid,
                         SFC.direct_sum_provider)

SFC.harmonic_spectra(x::AbstractMatrix, u::AbstractMatrix, lmax::Integer, ::SB.AbstractDirectSumSpectralBackend;
                     distance_metric = SFC.DI.SphericalAngle(), weights = nothing, valid = nothing) =
    SFC._harmonic_spectra(x, u, lmax, distance_metric, weights, valid, SFC.direct_sum_provider)

_no_transform_loaded(backend, schedule::SFC.AbstractSeparableSchedule) = throw(ArgumentError(
    "$(typeof(backend)) needs a transform this session has not loaded: `using FFTW` on CPU, or " *
    "another AbstractFFTs implementation, supplies it for every grid with a uniform direction. " *
    "DirectSumSpectralBackend and AutoSpectralBackend need none.",
))

_no_transform_loaded(backend, schedule) = throw(ArgumentError(
    "$(typeof(backend)) transforms along a uniform direction, and a $(nameof(typeof(schedule))) has " *
    "none: its pairs share no lag. Omit the tag, or pass DirectSumSpectralBackend or " *
    "AutoSpectralBackend, to enumerate the pairs.",
))

end # module
