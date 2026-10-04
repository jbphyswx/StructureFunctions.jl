module StructureFunctions

using StaticArrays: StaticArrays as SA
using PrecompileTools: PrecompileTools
using SpectralBackends: SpectralBackends

include("MultiFields.jl")
include("BinEdges.jl")
include("HelperFunctions.jl")
include("AuxiliaryAxes.jl")
include("StructureFunctionTypes.jl")
include("StructureFunctionObjects.jl")
include("Calculations.jl")
include("KHM.jl")

import .StructureFunctionObjects:
    AbstractStructureFunction,
    StructureFunction,
    StructureFunctionSumsAndCounts,
    StructureFunction2DSumsAndCounts,
    StructureFunctionTensor,
    StructureFunctionTensorSumsAndCounts,
    HelmholtzDecomposition2D, to_host

using .MultiFields: MultiFields
using .HelperFunctions: HelperFunctions
using .StructureFunctionTypes: StructureFunctionTypes
using .StructureFunctionObjects: StructureFunctionObjects
using .Calculations: Calculations
using .KHM: KHM

export AbstractBinEdges, BinEdges, LinearBinEdges, LogBinEdges, LogBinEdges_from_log_edges,
    InfPaddedBinEdges, ModeBinEdges, n_histogram_bins, midpoints
export AbstractTaper, NoTaper, Bartlett, GaussianTaper, HarmonicNodes, gauss_legendre
export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts,
    StructureFunction2DSumsAndCounts, StructureFunctionTensor, StructureFunctionTensorSumsAndCounts,
    HelmholtzDecomposition2D, to_host
export KHM

PrecompileTools.@setup_workload begin
    include("precompile.jl")
end

end
