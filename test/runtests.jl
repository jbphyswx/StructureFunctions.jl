using Test: Test
using StructureFunctions: StructureFunctions as SF

include("test_synthetic_data.jl")
using .SyntheticData: SyntheticData

Test.@testset "StructureFunctions.jl" begin
    println("--- Running Baseline Correctness Test ---")
    include("baseline_correctness.jl")

    println("--- Running Helpers Test ---")
    include("test_helpers.jl")

    println("--- Running Core Correctness Test ---")
    include("test_core_correctness.jl")

    println("--- Running Inputs Test ---")
    include("test_inputs.jl")

    println("--- Running E2E Test ---")
    include("test_e2e.jl")

    println("--- Running Stability & Inference Test ---")
    include("test_stability.jl")

    println("--- Running Shorthands Test ---")
    include("test_shorthands.jl")

    println("--- Running Spectral Test ---")
    include("test_spectral.jl")

    println("--- Running Spectral GPU Parity Test ---")
    include("test_spectral_gpu_parity.jl")

    # println("--- Running Performance Benchmark Test ---") # This need not run all the time, but it's here for reference
    # include("benchmark_performance.jl")

    # Enable Parallel/Distributed Test
    println("--- Running Parallel Equivalence Test ---")
    include("test_parallel_equivalence.jl")

    println("--- Running Real Data Extensions Test ---")
    include("test_real_data_extensions.jl")

    println("--- Running GPU Parity Test ---")
    include("test_gpu_parity.jl")

    println("--- Running Aqua Test ---")
    include("test_aqua.jl")

    println("--- Running JET Test ---")
    include("test_jet.jl")
end
