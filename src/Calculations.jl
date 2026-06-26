"""
The workhorse of this package, split into focused files under src/Calculations/
"""
module Calculations

using ProgressMeter: ProgressMeter as PM
using Distances: Distances as DI
using ..HelperFunctions: HelperFunctions as SFH
using ..StructureFunctionTypes: StructureFunctionTypes as SFT
using ..StructureFunctionObjects: StructureFunctionObjects as SFO
using ..StructureFunctions: AbstractBinEdges, BinEdges, LinearBinEdges, LogBinEdges,
    InfPaddedBinEdges, n_histogram_bins

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
    AbstractExecutionBackend, SerialBackend, ThreadedBackend, DistributedBackend,
    GPUBackend, AutoBackend, AbstractThreadingBackend, AutoThreadingBackend,
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
    GPUSFWorkspace, reset_histogram!, release!,
    joint2d_smem_max, joint2d_smem_exact, joint2d_smem_align256,
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
include("Calculations/batch_api.jl")
include("Calculations/gpu_stubs.jl")
include("Calculations/batch_leading.jl")
include("Calculations/batch.jl")
include("Calculations/serial.jl")
include("Calculations/serial_2d.jl")
include("Calculations/serial_single_pass.jl")
include("Calculations/tensor.jl")
include("Calculations/dispatch.jl")

end
