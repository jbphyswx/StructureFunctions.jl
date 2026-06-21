# BatchPrototypes — Phase 0 isolated harness for batched trailing dimensions.
# Not wired to src/Calculations.jl. See gpu/batch_prototypes/README.md.

module BatchPrototypes

using KernelAbstractions: KernelAbstractions as KA, @index, @atomic, @Const, @kernel, @localmem, @synchronize
using Printf: Printf, @printf, @sprintf
using Random: Random
using StaticArrays: StaticArrays as SA
using StructureFunctions:
    StructureFunctions, HelperFunctions as SFH, StructureFunctionTypes as SFT, LinearBinEdges

# Production GPU ext: tiled128 schedule, digitize, pair/tile indexing (not GPUPrototypeKernels).
const SFGE = Base.get_extension(StructureFunctions, :StructureFunctionsGPUExt)

# CPU slice gold + pair enumeration only (experimental file — not used for GPU batch paths).
const _GPUP = let m = Module()
    Base.include(m, joinpath(@__DIR__, "..", "GPUPrototypeKernels.jl"))
    m
end

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "cpu_reference.jl"))
include(joinpath(@__DIR__, "batch_workspace.jl"))
include(joinpath(@__DIR__, "gpu_tiled_batch.jl"))
include(joinpath(@__DIR__, "gpu_fused_tiled_batch.jl"))
include(joinpath(@__DIR__, "gpu_kernels.jl"))
include(joinpath(@__DIR__, "harness.jl"))

end # module
