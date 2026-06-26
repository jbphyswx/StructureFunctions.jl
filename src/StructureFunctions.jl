module StructureFunctions # `using StructureFunctions`` should work `@everywhere` automatically... hopefully the methods and extensinos below follow...

using StaticArrays: StaticArrays as SA
using PrecompileTools: PrecompileTools

# using Distributed
# @everywhere include("ParallelCalculations.jl") # this works w/ include("src/StructureFunctions.jl") but not w/ using StructureFunctions, and the former dumps directly into Main...
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

using .HelperFunctions
using .StructureFunctionTypes
using .StructureFunctionObjects
using .Calculations
using .KHM

# Re-export key APIs
export AbstractBinEdges, BinEdges, LinearBinEdges, LogBinEdges, LogBinEdges_from_log_edges,
    InfPaddedBinEdges, physical_edges_vector, n_histogram_bins
export calculate_structure_function, calculate_structure_function!, calculate_structure_functions_single_pass,
    calculate_structure_functions_single_pass!, calculate_structure_functions_single_pass_2d,
    calculate_structure_functions_single_pass_2d!, helmholtz_decompose_2d,
    append_helmholtz_rotational_divergent_rows,
    marginalize_sp2d_then_append_helmholtz_rows,
    calculate_structure_function_tensor, calculate_structure_function_tensor!,
    calculate_structure_function_slices!, calculate_structure_function_2d_slices!,
    calculate_structure_functions_single_pass_slices!, calculate_structure_functions_single_pass_2d_slices!,
    GPUSFWorkspace, reset_histogram!, release!,
    joint2d_smem_max, joint2d_smem_exact, joint2d_smem_align256
export marginalize
export AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend,
    GPUBackend, AutoBackend, AbstractThreadingBackend, AutoThreadingBackend
export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts, StructureFunction2DSumsAndCounts
export StructureFunctionTensor, StructureFunctionTensorSumsAndCounts, HelmholtzDecomposition2D
export LongitudinalSecondOrderStructureFunctionType,
    TransverseSecondOrderStructureFunctionType
export AbstractPairwiseStructureFunctionType, AbstractDerivedStructureFunctionType
export SecondOrderStructureFunctionType, ThirdOrderStructureFunctionType
export DiagonalConsistentThirdOrderStructureFunctionType,
    DiagonalInconsistentThirdOrderStructureFunctionType
export OffDiagonalConsistentThirdOrderStructureFunctionType,
    OffDiagonalInconsistentThirdOrderStructureFunctionType
export RotationalSecondOrderStructureFunctionType, DivergentSecondOrderStructureFunctionType,
    HelmholtzDecomposition2DType
export L2SFType, T2SFType, L3SFType, S2SFType, S3SFType, T3SFType, L2T1SFType, L1T2SFType
export T2ComponentSFType, L1T2ComponentSFType

export LongitudinalSecondOrderStructureFunction, TransverseSecondOrderStructureFunction
export SecondOrderStructureFunction, ThirdOrderStructureFunction
export DiagonalConsistentThirdOrderStructureFunction,
    DiagonalInconsistentThirdOrderStructureFunction
export OffDiagonalConsistentThirdOrderStructureFunction,
    OffDiagonalInconsistentThirdOrderStructureFunction
export L2SF, T2SF, L3SF, S2SF, S3SF, T3SF, L2T1SF, L1T2SF
export RotationalSecondOrderStructureFunction, DivergentSecondOrderStructureFunction,
    HelmholtzDecomposition2DOperator
export T2ComponentSF, L1T2ComponentSF
export get_structure_function_type
export KHM
export transverse_norm2,
    transverse_component_norm2,
    transverse_component,
    transverse_basis,
    transverse_basis_vector,
    AbstractTransverseBasisConvention,
    CanonicalTransverseBasis,
    ReferenceAxisTransverseBasis,
    CoordinateGaugeTransverseBasis,
    UserTransverseBasis,
    midpoints


# ---------------------------------------------------------------------------
# Initialization & Precompilation
# ---------------------------------------------------------------------------

PrecompileTools.@setup_workload begin
    include("precompile.jl")
end

end
