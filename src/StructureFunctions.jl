module StructureFunctions # `using StructureFunctions`` should work `@everywhere` automatically... hopefully the methods and extensinos below follow...

using StaticArrays: StaticArrays as SA
using PrecompileTools: PrecompileTools
abstract type AbstractStructureFunction end

# using Distributed
# @everywhere include("ParallelCalculations.jl") # this works w/ include("src/StructureFunctions.jl") but not w/ using StructureFunctions, and the former dumps directly into Main...
# using NaNStatistics # consider making this a strong dependency for easier use, needed for parallel but I'm sure we'll need it once we work with real data...

include("HelperFunctions.jl")
include("StructureFunctionTypes.jl")
include("StructureFunctionObjects.jl")
include("Calculations.jl")
include("SpectralAnalysis.jl")

import .StructureFunctionObjects: AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts

using .HelperFunctions
using .StructureFunctionTypes
using .StructureFunctionObjects
using .Calculations
using .SpectralAnalysis

# Re-export key APIs
export calculate_structure_function
export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts
export LongitudinalSecondOrderStructureFunctionType, TransverseSecondOrderStructureFunctionType
export SecondOrderStructureFunctionType, ThirdOrderStructureFunctionType
export DiagonalConsistentThirdOrderStructureFunctionType, DiagonalInconsistentThirdOrderStructureFunctionType
export OffDiagonalConsistentThirdOrderStructureFunctionType, OffDiagonalInconsistentThirdOrderStructureFunctionType
export L2SFType, T2SFType, L3SFType

export LongitudinalSecondOrderStructureFunction, TransverseSecondOrderStructureFunction
export SecondOrderStructureFunction, ThirdOrderStructureFunction
export DiagonalConsistentThirdOrderStructureFunction, DiagonalInconsistentThirdOrderStructureFunction
export OffDiagonalConsistentThirdOrderStructureFunction, OffDiagonalInconsistentThirdOrderStructureFunction
export L2SF, T2SF, L3SF

export SpectralAnalysis
export DirectSumBackend, FINUFFTBackend, FFTBackend
export calculate_spectrum


# ---------------------------------------------------------------------------
# Initialization & Precompilation
# ---------------------------------------------------------------------------


# function __init__()
#     # @require Distributed="8ba89e20-285c-5b6f-9357-94700520ee1b" include("ParallelCalculations.jl")
#     @require Distributed="8ba89e20-285c-5b6f-9357-94700520ee1b" begin # maybe this always works because Distributed is in the extension?
#         println(pkgdir(Distributed))
#         include("ParallelCalculations.jl") # This seems to be almost twice as fast as the extension, not sure why...
# end
# end

PrecompileTools.@setup_workload begin
    include("precompile.jl")
end

end
