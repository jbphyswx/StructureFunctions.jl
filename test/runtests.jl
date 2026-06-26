using Test: Test
using StructureFunctions: StructureFunctions as SF

include("test_synthetic_data.jl")
using .SyntheticData: SyntheticData

Test.@testset "StructureFunctions.jl" begin
    println("--- Running Baseline Correctness Test ---")
    include("baseline_correctness.jl")

    println("--- Running Helpers Test ---")
    include("test_helpers.jl")

    println("--- Running BinEdges Test ---")
    include("test_bin_edges.jl")

    println("--- Running Core Correctness Test ---")
    include("test_core_correctness.jl")

    println("--- Running Single-Pass & Helmholtz Test ---")
    include("test_single_pass.jl")

    println("--- Running Single-Pass 2D Test ---")
    include("test_single_pass_2d.jl")

    println("--- Running E2E Test ---")
    include("test_e2e.jl")

    println("--- Running Stability & Inference Test ---")
    include("test_stability.jl")

    println("--- Running Shorthands Test ---")
    include("test_shorthands.jl")

    println("--- Running Shape Contract Test ---")
    include("test_shape_contract.jl")

    println("--- Running GPU Shape Contract Test ---")
    include("test_gpu_shape_contract.jl")

    println("--- Running Tensor and KHM Test ---")
    include("test_tensor_khm.jl")

    println("--- Running Threading Backend Test ---")
    include("test_threads.jl")

    println("--- Running Triangle Outer Chunks Test ---")
    include("test_triangle_outer_chunks.jl")


    # println("--- Running Performance Benchmark Test ---") # This need not run all the time, but it's here for reference
    # include("benchmark_performance.jl")

    # Enable Parallel/Distributed Test
    println("--- Running Parallel Equivalence Test ---")
    include("test_parallel_equivalence.jl")

    println("--- Running MPI Backend Test ---")
    include("test_mpi.jl")

    println("--- Running GPU Parity Test ---")
    include("test_gpu_parity.jl")

    println("--- Running GPU Tiled Parity Test ---")
    include("test_gpu_tiled_parity.jl")

    println("--- Running GPU Single-Pass Tiled Parity Test ---")
    include("test_gpu_single_pass_tiled.jl")

    println("--- Running GPU sp2d HTP-EJ Partitioned Test ---")
    include("test_gpu_sp2d_partitioned.jl")

    println("--- Running GPU joint2d smem Test ---")
    include("test_gpu_joint2d_smem.jl")

    println("--- Running GPU Workspace & Slice Batch Test ---")
    include("test_gpu_workspace.jl")

    println("--- Running GPU Script Hygiene Test ---")
    include("test_gpu_script_hygiene.jl")

    println("--- Running Batch Matrix Parity Test ---")
    include("test_batch_matrix.jl")

    println("--- Running 2D Joint-Probability Binning Test ---")
    include("test_2d_binning.jl")

    println("--- Running Pre-allocated In-place Buffer Test ---")
    include("test_inplace.jl")

    println("--- Running Aqua Test ---")
    include("test_aqua.jl")

    println("--- Running JET Test ---")
    include("test_jet.jl")
end
