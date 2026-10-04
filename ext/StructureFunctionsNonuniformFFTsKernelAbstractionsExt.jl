module StructureFunctionsNonuniformFFTsKernelAbstractionsExt

using NonuniformFFTs: NonuniformFFTs as NU
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC

# The device plan pins the backwards Kaiser–Bessel kernel with piecewise-polynomial evaluation.
SFC.nufft_plan(tag::SFC.NonuniformFFTsSpectralBackend, ::Type{FT}, modes, x::AbstractArray) where {FT} =
    NU.PlanNUFFT(FT, modes; backend = KA.get_backend(x), m = NU.HalfSupport(SFC.nufft_half_support(tag)),
                 kernel = NU.BackwardsKaiserBesselKernel(), kernel_evalmode = NU.FastApproximation())

end # module
