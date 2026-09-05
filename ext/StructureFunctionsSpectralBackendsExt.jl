module StructureFunctionsSpectralBackendsExt

using SpectralBackends: SpectralBackends as SB
using StructureFunctions: Calculations as SFC

# The tags that name an algorithm needing no transform. `Auto` resolves here to the lag sweep because
# that is the only summation this session has; the transform extension answers `Auto` on the concrete
# tag, which out-specialises this, and weighs the two costs.

SFC.gridded_sweep!(sums, counts, sf, u, schedule, distance_bins, ::Val{D},
                   ::SB.AbstractDirectSumSpectralBackend; valid = SFC.AllValid()) where {D} =
    SFC.gridded_lag_sweep!(sums, counts, sf, u, schedule, distance_bins, Val(D); valid)

SFC.gridded_sweep!(sums, counts, sf, u, schedule, distance_bins, ::Val{D},
                   ::SB.AbstractAutoSpectralBackend; valid = SFC.AllValid()) where {D} =
    SFC.gridded_lag_sweep!(sums, counts, sf, u, schedule, distance_bins, Val(D); valid)

function SFC.gridded_sweep!(sums, counts, sf, u, schedule, distance_bins, ::Val{D},
                            backend::SB.AbstractSpectralBackend; kwargs...) where {D}
    throw(ArgumentError(
        "$(typeof(backend)) needs a transform this session has not loaded: `using FFTW` on CPU, " *
        "or another AbstractFFTs implementation. DirectSumSpectralBackend and AutoSpectralBackend " *
        "need none.",
    ))
end

end # module
