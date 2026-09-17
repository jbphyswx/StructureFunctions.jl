module StructureFunctionsNonuniformFFTsKernelAbstractionsExt

using NonuniformFFTs: NonuniformFFTs as NU
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC

# The device plan pins the kernel and its evaluation: on CUDA the backend's default direct Kaiser–Bessel
# evaluation reaches only ~3e-6 in Float64 (its device Bessel I₀), while the piecewise-polynomial
# evaluation of the backwards kernel reaches ~1e-15.
SFC.nufft_plan(tag::SFC.NonuniformFFTsSpectralBackend, ::Type{FT}, modes, x::AbstractArray) where {FT} =
    NU.PlanNUFFT(FT, modes; backend = KA.get_backend(x), m = NU.HalfSupport(SFC.nufft_half_support(tag)),
                 kernel = NU.BackwardsKaiserBesselKernel(), kernel_evalmode = NU.FastApproximation())

end # module
