module StructureFunctions

using StaticArrays: StaticArrays as SA
using PrecompileTools: PrecompileTools
using SpectralBackends: SpectralBackends

include("Channels.jl")
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

using .Channels: Channels # what is a channel
using .HelperFunctions: HelperFunctions
using .StructureFunctionTypes: StructureFunctionTypes
using .StructureFunctionObjects: StructureFunctionObjects
using .Calculations: Calculations
using .KHM: KHM

# Re-export key APIs
export Fields
export AbstractTaper, NoTaper, Bartlett, GaussianTaper, HarmonicNodes, gauss_legendre
export AbstractBinEdges, BinEdges, LinearBinEdges, LogBinEdges, LogBinEdges_from_log_edges,
    InfPaddedBinEdges, ModeBinEdges, physical_edges_vector, n_histogram_bins
export ScatteredModesSchedule, NonuniformFFTsSpectralBackend, FINUFFTSpectralBackend
export calculate_structure_function, calculate_structure_function!, calculate_structure_functions_single_pass,
    calculate_structure_functions_single_pass!, calculate_structure_functions_single_pass_2d,
    calculate_structure_functions_single_pass_2d!, helmholtz_decompose_2d,
    append_helmholtz_rotational_divergent_rows,
    marginalize_sp2d_then_append_helmholtz_rows,
    calculate_structure_function_tensor, calculate_structure_function_tensor!,
    calculate_structure_function_batch!, calculate_structure_function_2d_batch!,
    calculate_structure_functions_single_pass_batch!, calculate_structure_functions_single_pass_2d_batch!,
    GPUSFWorkspace, CPUSFWorkspace, reset_histogram!, release!,
    joint2d_smem_max, joint2d_smem_exact, joint2d_smem_align256
export isotropic_spectrum, shell_spectrum, gridded_spectrum, shell_average, cell_measure,
    helmholtz_spectra, spectral_flux, enstrophy_flux, covariance, covariance_matrix
export AbstractForwardModel, SpectrumForwardModel, HelmholtzForwardModel, FluxForwardModel, forward_matrix, flux_matrix,
    AbstractFitMethod, RegularizedLeastSquares, NonNegativeLeastSquares, SegmentedPowerLaw, segmented_spectrum,
    fit_spectrum, fit_helmholtz_spectra, fit_flux, tradeoff_curve, select_segments, independent_pair_variance
export marginalize
export AbstractStructureFunction, StructureFunction, StructureFunctionSumsAndCounts, StructureFunction2DSumsAndCounts
export StructureFunctionTensor, StructureFunctionTensorSumsAndCounts, StructureFunctionTensor2DSumsAndCounts,
    HelmholtzDecomposition2D
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
export ScalarStructureFunctionType, MixedStructureFunctionType, ScalarDotStructureFunctionType,
    VectorDotStructureFunctionType, ScalarSFType, MixedSFType, ScalarDotSFType, VectorDotSFType

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
    SphericalDistance,
    midpoints


# ---------------------------------------------------------------------------
# Initialization & Precompilation
# ---------------------------------------------------------------------------

PrecompileTools.@setup_workload begin
    include("precompile.jl")
end

end
