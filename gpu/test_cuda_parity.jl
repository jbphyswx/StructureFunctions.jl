"""
    test_cuda_parity.jl

CUDA device parity: compare `gpu_calculate_structure_function` on `CUDABackend`
against the serial CPU reference.

Included by `runtests.jl`; not run from the main `test/` suite.
"""

using Test: Test
using CUDA: CUDA
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionTypes as SFT,
    LinearBinEdges, LogBinEdges
using Random: Random

Random.seed!(42)

Test.@testset "CUDA structure-function parity" begin
    Test.@test CUDA.functional()

    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)

    bin_edges = collect(FT, range(0.0, 1.4; length = 11))
    sft = SFT.L2SFType()

    res_ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, bin_edges;
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    res_cuda = SFC.gpu_calculate_structure_function(
        sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges;
        return_sums_and_counts = true,
    )
    CUDA.synchronize()

    Test.@test res_cuda.counts ≈ res_ref.counts atol = 0.0

    max_Δ = maximum(abs, res_cuda.sums .- res_ref.sums)
    # Float32 GPU path uses atomic adds; order differs from serial CPU → small drift.
    Test.@test max_Δ < 0.05f0

    println("CUDA linear parity OK  max |Δ sums| = ", max_Δ)
end

Test.@testset "CUDA log-spaced bin parity" begin
    Test.@test CUDA.functional()

    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N) .+ FT(0.01)
    u_cpu = rand(FT, 2, N)
    x_gpu = CUDA.cu(x_cpu)
    u_gpu = CUDA.cu(u_cpu)

    log_edge_vec = exp.(range(log(FT(0.05)), log(FT(1.4)); length = 11))
    bin_edges = LogBinEdges(log_edge_vec)
    sft = SFT.L2SFType()

    res_ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, bin_edges;
        verbose = false,
        show_progress = false,
        return_sums_and_counts = true,
    )

    res_cuda = SFC.gpu_calculate_structure_function(
        sft, CUDA.CUDABackend(), x_gpu, u_gpu, bin_edges;
        return_sums_and_counts = true,
    )
    CUDA.synchronize()

    Test.@test res_cuda.counts ≈ res_ref.counts atol = 0.0
    max_Δ = maximum(abs, res_cuda.sums .- res_ref.sums)
    Test.@test max_Δ < 0.05f0
    println("CUDA log-bin parity OK  max |Δ sums| = ", max_Δ)
end

Test.@testset "CUDA joint 2D structure-function parity" begin
    Test.@test CUDA.functional()

    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N)
    u_cpu = rand(FT, 2, N)

    distance_bins = collect(FT, range(0.0, 1.4; length = 11))
    value_bins = collect(FT, range(0.0, 2.0; length = 11))
    sft = SFT.L2SFType()

    ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, distance_bins, value_bins;
        verbose = false, show_progress = false,
    )
    gpu = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, distance_bins, value_bins;
        backend = SF.GPUBackend(CUDA.CUDABackend()),
        verbose = false, show_progress = false,
    )
    CUDA.synchronize()

    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    max_Δ = maximum(abs, gpu.sums .- ref.sums)
    Test.@test max_Δ < 0.05f0
    println("CUDA joint 2D linear parity OK  max |Δ sums| = ", max_Δ)
end

Test.@testset "CUDA joint 2D log distance bins" begin
    Test.@test CUDA.functional()

    N = 500
    FT = Float32
    x_cpu = rand(FT, 2, N) .+ FT(0.01)
    u_cpu = rand(FT, 2, N)

    distance_bins = exp.(range(log(FT(0.05)), log(FT(1.4)); length = 11))
    value_bins = collect(FT, range(-1.0, 1.0; length = 9))
    sft = SFT.L3SFType()

    ref = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, distance_bins, value_bins;
        backend = SFC.SerialBackend(),
        verbose = false, show_progress = false,
    )
    gpu = SFC.calculate_structure_function(
        sft, x_cpu, u_cpu, distance_bins, value_bins;
        backend = SF.GPUBackend(CUDA.CUDABackend()),
        verbose = false, show_progress = false,
    )
    CUDA.synchronize()

    Test.@test gpu.counts ≈ ref.counts atol = 0.0
    max_Δ = maximum(abs, gpu.sums .- ref.sums)
    Test.@test max_Δ < 0.05f0
    println("CUDA joint 2D log-distance parity OK  max |Δ sums| = ", max_Δ)
end

"""Wide synthetic value-bin edges (matches test/test_single_pass_2d.jl)."""
function _cuda_synthetic_value_bins_ntuple(n_bins::Int, ::Type{FT}) where {FT}
    edges = collect(FT, range(-1.0, 2.0; length = n_bins + 1))
    template = vcat(FT(-Inf), edges, FT(Inf))
    return ntuple(_ -> copy(template), 6)
end

Test.@testset "CUDA single-pass 2D parity" begin
    Test.@test CUDA.functional()

    N = 200
    FT = Float32
    x_cpu = rand(FT, 2, N) .* FT(50000)
    u_cpu = randn(FT, 2, N) .* FT(0.5)
    distance_bins = exp.(range(log(FT(1000)), log(FT(50000)); length = 6))
    value_bins = _cuda_synthetic_value_bins_ntuple(10, FT)

    sums_ref = zeros(Float64, 6, length(distance_bins) - 1, length(value_bins[1]) - 1)
    counts_ref = zeros(UInt32, size(sums_ref))
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, counts_ref, x_cpu, u_cpu, distance_bins, value_bins;
        backend = SFC.SerialBackend(),
    )

    inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
    sp_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x_cpu, u_cpu, distance_bins, value_bins;
        backend = SF.GPUBackend(CUDA.CUDABackend()),
    )
    CUDA.synchronize()

    max_Δ = 0.0
    for (t, k) in enumerate(inv)
        Test.@test sp_gpu[k].counts == counts_ref[t, :, :]
        max_Δ = max(max_Δ, maximum(abs, sp_gpu[k].sums .- sums_ref[t, :, :]))
    end
    Test.@test max_Δ < 0.1f0
    println("CUDA single-pass 2D parity OK  max |Δ sums| = ", max_Δ)
end
