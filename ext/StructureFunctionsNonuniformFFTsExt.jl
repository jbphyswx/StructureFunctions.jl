module StructureFunctionsNonuniformFFTsExt

using NonuniformFFTs: NonuniformFFTs as NU
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC

SFC._nufft_loaded(::Type{SFC.NonuniformFFTsSpectralBackend}) = true

# NonuniformFFTs' type-1 transform carries the minus sign and, for real data, the real-to-complex half
# spectrum `(M₁ ÷ 2 + 1, M₂, …)`; on uniform points it equals FFTW's `rfft`, so the engine reads these
# arrays exactly as it reads a slab's transforms. One plan serves every monomial of the point set.
function SFC.nufft_monomial_transforms(
    tag::SFC.NonuniformFFTsSpectralBackend, s::SFC.ScatteredModesSchedule{Dg, T},
    data::AbstractMatrix, valid, weights, keys, ::Val{Pm}; to = identity,
) where {Dg, T, Pm}
    m = tag.half_support
    all(M -> M >= m, s.modes) || throw(ArgumentError(
        "every direction needs at least half_support = $m modes for NonuniformFFTs' spreading kernel to fit its " *
        "oversampled grid; got $(s.modes). Raise the modes or lower the tag's half_support.",
    ))
    FT = float(eltype(data))
    θ = ntuple(d -> to(FT.(SFC.mode_coordinates(s, d))), Val(Dg))
    backend = KA.get_backend(θ[1])
    # The kernel and its evaluation are fixed rather than left to the backend's defaults: on CUDA the
    # default direct Kaiser–Bessel evaluation reaches only ~3e-6 in Float64 (its device Bessel I₀),
    # while the piecewise-polynomial evaluation of the backwards kernel reaches ~1e-15 on every backend.
    plan = NU.PlanNUFFT(FT, s.modes; m = NU.HalfSupport(m), backend,
                        kernel = NU.BackwardsKaiserBesselKernel(), kernel_evalmode = NU.FastApproximation())
    NU.set_points!(plan, θ)
    taper = SFC.mode_taper_weights(s, FT, size(plan), to)
    return map(keys) do key
        v = SFC._held_monomial_vector(data, valid, weights, key, FT)
        û = similar(θ[1], Complex{FT}, size(plan))
        NU.exec_type1!(û, plan, v)
        taper === nothing || (û .*= taper)
        û
    end
end

end # module
