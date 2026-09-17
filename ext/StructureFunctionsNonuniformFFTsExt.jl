module StructureFunctionsNonuniformFFTsExt

using NonuniformFFTs: NonuniformFFTs as NU
using StructureFunctions: StructureFunctions as SF, Calculations as SFC

# NonuniformFFTs' type-1 transform carries the minus sign and, for real data, the real-to-complex half
# spectrum `(M₁ ÷ 2 + 1, M₂, …)`; on uniform points it equals FFTW's `rfft`, so the engine reads these
# arrays exactly as it reads a slab's transforms. One plan serves every monomial of the point set, on
# the host or on the device the points live on.
function SFC.nufft_monomial_transforms(
    tag::SFC.NonuniformFFTsSpectralBackend, s::SFC.ScatteredModesSchedule{Dg, T},
    data::AbstractMatrix, valid, weights, keys, ::Val{Pm}; to = identity,
) where {Dg, T, Pm}
    m = SFC.nufft_half_support(tag)
    all(M -> M >= m, s.modes) || throw(ArgumentError(
        "every direction needs at least $m modes, the kernel half-support at tolerance $(tag.tolerance), for " *
        "NonuniformFFTs' spreading kernel to fit its oversampled grid; got $(s.modes). Raise the modes or the tolerance.",
    ))
    FT = float(eltype(data))
    θ = ntuple(d -> to(FT.(SFC.mode_coordinates(s, d))), Val(Dg))
    plan = SFC.nufft_plan(tag, FT, s.modes, θ[1])
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

SFC.nufft_plan(tag::SFC.NonuniformFFTsSpectralBackend, ::Type{FT}, modes, ::Array) where {FT} =
    NU.PlanNUFFT(FT, modes; m = NU.HalfSupport(SFC.nufft_half_support(tag)))

end # module
