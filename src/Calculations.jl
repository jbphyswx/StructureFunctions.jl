"""
The workhorse of this package, split into focused files under src/Calculations/
"""
module Calculations

using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using SpectralBackends: SpectralBackends as SB
using ..HelperFunctions: HelperFunctions as SFH
using ..MultiFields: MultiFields as MF
using ..StructureFunctionTypes: StructureFunctionTypes as SFT
using ..StructureFunctionObjects: StructureFunctionObjects as SFO
using ..StructureFunctions: AbstractBinEdges, BinEdges, LinearBinEdges, LogBinEdges,
    InfPaddedBinEdges, ModeBinEdges, n_histogram_bins, midpoints,
    AbstractTaper, NoTaper, Bartlett, GaussianTaper, taper_weight, harmonic_taper, mode_taper, HarmonicNodes,
    gauss_legendre,
    AbstractSquaredDigitizePlan, squared_digitize_plan, squared_digitize,
    squared_approx_index, squared_correct, squared_bin, has_vector_index, digitize_key

using StaticArrays: StaticArrays as SA
using LinearAlgebra: LinearAlgebra as LA
using Base.Threads: Threads

import ..batch_dims
import ..batch_size
import ..batch_field_slice
import ..batch_histograms_equal
import ..batch_max_abs_diff
import ..pair_from_linear
import .._pair_from_linear
import .._flatten_sums_counts

export calculate_structure_function,
    gpu_calculate_structure_function, gpu_calculate_structure_function!,
    gpu_calculate_structure_function_2d, gpu_calculate_structure_function_2d_batch,
    gpu_calculate_structure_functions_single_pass_2d,
    gpu_calculate_structure_functions_single_pass_2d!,
    serial_calculate_structure_function, threaded_calculate_structure_function,
    calculate_structure_functions_single_pass,
    calculate_structure_functions_single_pass!,
    calculate_structure_functions_single_pass_2d,
    calculate_structure_functions_single_pass_2d!,
    serial_calculate_structure_functions_single_pass_2d,
    helmholtz_decompose_2d,
    append_helmholtz_rotational_divergent_rows,
    marginalize_sp2d_then_append_helmholtz_rows,
    serial_calculate_structure_function!, threaded_calculate_structure_function!,
    calculate_structure_function_tensor, calculate_structure_function_tensor!,
    serial_calculate_structure_function_tensor!,
    gpu_calculate_structure_function!, calculate_structure_function!,
    GPUSFWorkspace, CPUSFWorkspace, reset_histogram!, release!,
    joint2d_smem_max, joint2d_smem_exact, joint2d_smem_align256,
    isotropic_spectrum, shell_spectrum, gridded_spectrum, shell_average, cell_measure,
    ScatteredModesSchedule, NonuniformFFTsSpectralBackend, FINUFFTSpectralBackend, nufft_half_support,
    nufft_monomial_transforms,
    helmholtz_spectra, spectral_flux, enstrophy_flux, covariance, covariance_matrix, harmonic_sweep!, harmonic_spectra,
    AbstractForwardModel, SpectrumForwardModel, HelmholtzForwardModel, FluxForwardModel, forward_matrix, flux_matrix,
    AbstractFitMethod, RegularizedLeastSquares, NonNegativeLeastSquares, SegmentedPowerLaw, segmented_spectrum,
    fit_spectrum, fit_helmholtz_spectra, fit_flux, tradeoff_curve, select_segments, independent_pair_variance,
    calculate_structure_function_batch!, calculate_structure_function_2d_batch!,
    calculate_structure_functions_single_pass_batch!,
    calculate_structure_functions_single_pass_2d_batch!,
    auxiliary_shared_positions!, auxiliary_varying_positions!,
    serial_calculate_structure_functions_single_pass!,
    serial_calculate_structure_functions_single_pass_2d!,
    auxiliary_joint2d!, cpu_slice_baseline!

# Re-include backend types, GPU stubs, batch CPU drivers, serial solvers, and main entry dispatch.
include("Calculations/backends.jl")
include("Calculations/shapes.jl")
include("Calculations/culling.jl")
include("Calculations/pair_schedule.jl")
include("Calculations/second_axis.jl")
include("Calculations/gridded.jl")
include("Calculations/gridded_zonal.jl")
include("Calculations/scattered_modes.jl")
include("Calculations/lag_moments.jl")
include("Calculations/sorted_line.jl")
include("Calculations/transforms.jl")
include("Calculations/fits.jl")
include("Calculations/batch_api.jl")
include("Calculations/gpu_stubs.jl")
include("Calculations/batch_leading.jl")
include("Calculations/workspace.jl")
include("Calculations/batch.jl")
include("Calculations/serial.jl")
include("Calculations/serial_2d.jl")
include("Calculations/serial_single_pass.jl")
include("Calculations/tensor.jl")
include("Calculations/dispatch.jl")
include("Calculations/multifields.jl")
include("Calculations/harmonic.jl")

end
