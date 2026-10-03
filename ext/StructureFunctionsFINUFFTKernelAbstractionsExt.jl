module StructureFunctionsFINUFFTKernelAbstractionsExt

using FINUFFT: FINUFFT
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC

SFC.nufft_type1_plan(tag::SFC.FINUFFTSpectralBackend, θ::Tuple, modes, ntrans::Int) =
    _type1_plan(KA.get_backend(first(θ)), tag, θ, modes, ntrans)

"""A cuFINUFFT plan: its type exists once `using CUDA` has loaded FINUFFT's device interface, so it is held
here and never named in a signature."""
struct CuFINUFFTPlan{P}
    plan::P
end

function _type1_plan(backend::KA.GPU, tag, θ, modes, ntrans)
    isdefined(FINUFFT, :cufinufft_makeplan) || throw(ArgumentError(
        "FINUFFT's device transform is cuFINUFFT, loaded by `using CUDA`; the points live on " *
        "$(nameof(typeof(backend))).",
    ))
    plan = FINUFFT.cufinufft_makeplan(1, collect(Int64, modes), -1, ntrans, tag.tolerance;
                                      dtype = eltype(θ[1]), modeord = 1)
    try
        FINUFFT.cufinufft_setpts!(plan, θ...)
    catch
        FINUFFT.cufinufft_destroy!(plan)
        rethrow()
    end
    return CuFINUFFTPlan(plan)
end

_type1_plan(::KA.CPU, tag, θ, modes, ntrans) = throw(ArgumentError(
    "FINUFFT's host transform takes `Array`s; got $(typeof(first(θ))).",
))

SFC.nufft_type1_exec!(p::CuFINUFFTPlan, strengths, full) = (FINUFFT.cufinufft_exec!(p.plan, strengths, full); full)
SFC._release_plan!(p::CuFINUFFTPlan) = (FINUFFT.cufinufft_destroy!(p.plan); nothing)

end # module
