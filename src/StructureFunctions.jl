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
    HelmholtzDecomposition2D

using .MultiFields: MultiFields # what is a field
using .HelperFunctions: HelperFunctions
using .StructureFunctionTypes: StructureFunctionTypes
using .StructureFunctionObjects: StructureFunctionObjects
using .Calculations: Calculations
using .KHM: KHM

# The package's own names. Everything else is reached through the submodule that owns it —
# `Calculations` for the entries, `StructureFunctionTypes` for the operators, `HelperFunctions` for
# the geometry, `MultiFields` for `Fields`.
export AbstractBinEdges, BinEdges, LinearBinEdges, LogBinEdges, LogBinEdges_from_log_edges,
    InfPaddedBinEdges, ModeBinEdges, physical_edges_vector, n_histogram_bins, midpoints
export AbstractTaper, NoTaper, Bartlett, GaussianTaper, HarmonicNodes, gauss_legendre
export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts,
    StructureFunction2DSumsAndCounts, StructureFunctionTensor, StructureFunctionTensorSumsAndCounts,
    HelmholtzDecomposition2D
export KHM

# ---------------------------------------------------------------------------
# Initialization & Precompilation
# ---------------------------------------------------------------------------

PrecompileTools.@setup_workload begin
    include("precompile.jl")
end

end
