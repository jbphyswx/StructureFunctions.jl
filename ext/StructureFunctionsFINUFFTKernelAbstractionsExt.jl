module StructureFunctionsFINUFFTKernelAbstractionsExt

using FINUFFT: FINUFFT
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: Calculations as SFC

SFC.nufft_type1!(tag::SFC.FINUFFTSpectralBackend, full::AbstractArray, strengths::AbstractArray, θ::Tuple, modes) =
    _type1!(KA.get_backend(full), tag, full, strengths, θ, modes)

# cuFINUFFT's names exist once `using CUDA` has loaded FINUFFT's device interface, so they are looked up
# here and never named in a signature.
function _type1!(backend::KA.GPU, tag, full, strengths, θ, modes)
    isdefined(FINUFFT, :cufinufft_makeplan) || throw(ArgumentError(
        "FINUFFT's device transform is cuFINUFFT, loaded by `using CUDA`; the points live on " *
        "$(nameof(typeof(backend))).",
    ))
    plan = FINUFFT.cufinufft_makeplan(1, collect(Int64, modes), -1, size(strengths, 2), tag.tolerance;
                                      dtype = eltype(θ[1]), modeord = 1)
    try
        FINUFFT.cufinufft_setpts!(plan, θ...)
        FINUFFT.cufinufft_exec!(plan, strengths, full)
    finally
        FINUFFT.cufinufft_destroy!(plan)
    end
    return full
end

_type1!(::KA.CPU, tag, full, strengths, θ, modes) = throw(ArgumentError(
    "FINUFFT's host transform takes `Array`s; got $(typeof(full)).",
))

end # module
