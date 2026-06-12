using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges, LogBinEdges
using Random: Random
using LinearAlgebra: LinearAlgebra as LA
using StaticArrays: StaticArrays as SA

Random.seed!(42)

function _cpu_ref(sft, x_tup, u_tup, bin_edges)
    return SFC.calculate_structure_function(
        sft, x_tup, u_tup, bin_edges;
        verbose = false, show_progress = false, return_sums_and_counts = true,
    )
end

function _gpu_tiled(sft, x, u, bin_edges)
    return SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges;
        return_sums_and_counts = true,
    )
end

Test.@testset "GPU tiled parity — linear 2D N=50" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, (x[1, :], x[2, :]), (u[1, :], u[2, :]), bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — linear 3D N=50" begin
    N = 50
    FT = Float64
    x = rand(FT, 3, N)
    u = rand(FT, 3, N)
    bin_edges = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(
        sft,
        (x[1, :], x[2, :], x[3, :]),
        (u[1, :], u[2, :], u[3, :]),
        bin_edges,
    )
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — log bins 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = rand(FT, 2, N)
    log_vec = exp.(range(log(FT(0.05)), log(FT(1.4)); length = 11))
    bin_edges = LogBinEdges(log_vec)
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, (x[1, :], x[2, :]), (u[1, :], u[2, :]), log_vec)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — general monotone bins 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    # Non-uniform, non-log monotone edges
    bin_edges = FT[0.0, 0.05, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.35]
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, (x[1, :], x[2, :]), (u[1, :], u[2, :]), bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — medium N linear 2D" begin
    N = 500
    FT = Float32
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0f0, 1.4f0; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, (x[1, :], x[2, :]), (u[1, :], u[2, :]), bin_edges)
    gpu = _gpu_tiled(sft, x, u, bin_edges)
    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    max_Δ = maximum(abs, gpu.sums .- ref.sums)
    Test.@test max_Δ < 0.05f0
end

Test.@testset "GPU in-place !() parity — linear 2D" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    sft = SFT.L2SFType()
    ref = _cpu_ref(sft, (x[1, :], x[2, :]), (u[1, :], u[2, :]), bin_edges)
    n_bins = length(bin_edges) - 1
    sums = zeros(FT, n_bins)
    counts = zeros(UInt32, n_bins)
    SFC.gpu_calculate_structure_function!(sums, counts, sft, KA.CPU(), x, u, bin_edges)
    Test.@test counts == ref.counts
    Test.@test sums ≈ ref.sums atol = 1e-10
    SFC.gpu_calculate_structure_function!(sums, counts, sft, KA.CPU(), x, u, bin_edges)
    Test.@test counts == ref.counts .* 2
    Test.@test sums ≈ ref.sums .* 2 atol = 1e-10
end

Test.@testset "GPU joint 2D parity — L2SF linear bins N=50" begin
    N = 50
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    distance_bins = collect(FT, range(0.0, 1.4; length = 11))
    value_bins = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()
    ref = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = SFC.SerialBackend(), verbose = false, show_progress = false,
    )
    gpu = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = SF.GPUBackend(KA.CPU()), verbose = false, show_progress = false,
    )
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU joint 2D parity — L3SF log distance bins" begin
    N = 40
    FT = Float64
    x = rand(FT, 2, N) .+ FT(0.01)
    u = randn(FT, 2, N)
    distance_bins = exp.(range(log(0.05), log(2.0); length = 8))
    value_bins = collect(FT, range(-1.0, 1.0; length = 9))
    sft = SFT.L3SFType()
    ref = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = SFC.SerialBackend(), verbose = false, show_progress = false,
    )
    gpu = SFC.calculate_structure_function(
        sft, x, u, distance_bins, value_bins;
        backend = SF.GPUBackend(KA.CPU()), verbose = false, show_progress = false,
    )
    Test.@test gpu.counts == ref.counts
    Test.@test gpu.sums ≈ ref.sums atol = 1e-10
end

Test.@testset "GPU tiled parity — NB > 64 errors" begin
    N = 20
    FT = Float64
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    bin_edges = collect(FT, range(0.0, 2.0; length = 67))  # 66 bins
    sft = SFT.L2SFType()
    Test.@test_throws ErrorException SFC.gpu_calculate_structure_function(
        sft, KA.CPU(), x, u, bin_edges; return_sums_and_counts = true,
    )
end
