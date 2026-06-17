using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using Random: Random

Random.seed!(42)

function _value_bins_uniform(n_val::Int, ::Type{FT}) where {FT}
    edges = collect(range(FT(-1), FT(2); length = n_val + 1))
    return [copy(edges) for _ in 1:8]
end

Test.@testset "GPU single-pass tiled parity — 1D log bins" begin
    N = 64
    FT = Float32
    x = rand(FT, 2, N) .* FT(5000)
    u = randn(FT, 2, N) .* FT(0.3)
    dist_vec = LogBinEdges(Vector{FT}(exp.(range(log(FT(10)), log(FT(5000)); length = 11))))

    sums_cpu = zeros(FT, 8, length(dist_vec) - 1)
    counts_cpu = zeros(Int32, 8, length(dist_vec) - 1)
    SFC.calculate_structure_functions_single_pass!(
        sums_cpu, counts_cpu, x, u, dist_vec; backend = SFC.SerialBackend(),
    )

    sums_gpu = zeros(FT, 8, length(dist_vec) - 1)
    counts_gpu = zeros(Int32, 8, length(dist_vec) - 1)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist_vec; kind = :single_pass)
    SFC.calculate_structure_functions_single_pass!(
        sums_gpu, counts_gpu, x, u, dist_vec;
        backend = SF.GPUBackend(KA.CPU()), workspace = ws,
    )

    Test.@test sums_gpu ≈ sums_cpu rtol = FT(1e-4)
    Test.@test counts_gpu == counts_cpu
end

Test.@testset "GPU single-pass tiled parity — 2D log dist + linear value" begin
    N = 48
    FT = Float32
    x = rand(FT, 2, N) .* FT(5000)
    u = randn(FT, 2, N) .* FT(0.3)
    dist_vec = LogBinEdges(Vector{FT}(exp.(range(log(FT(10)), log(FT(5000)); length = 11))))
    n_val = 8
    value_bins = ntuple(_ -> LinearBinEdges(range(FT(-1), FT(2); length = n_val + 1)), 8)
    n_dist = length(dist_vec) - 1

    sums_cpu = zeros(FT, 8, n_dist, n_val)
    counts_cpu = zeros(Int32, 8, n_dist, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_cpu, counts_cpu, x, u, dist_vec, value_bins;
        backend = SFC.SerialBackend(),
    )

    sums_gpu = zeros(FT, 8, n_dist, n_val)
    counts_gpu = zeros(Int32, 8, n_dist, n_val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist_vec, value_bins; kind = :single_pass_2d)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, counts_gpu, x, u, dist_vec, value_bins;
        backend = SF.GPUBackend(KA.CPU()), workspace = ws,
    )

    Test.@test sums_gpu ≈ sums_cpu rtol = FT(1e-4)
    Test.@test counts_gpu == counts_cpu
end

Test.@testset "GPU single-pass tiled parity — 2D InfPadded linear value catch-alls" begin
    N = 48
    FT = Float32
    x = rand(FT, 2, N) .* FT(5000)
    u = randn(FT, 2, N) .* FT(0.3)
    dist_vec = LogBinEdges(Vector{FT}(exp.(range(log(FT(10)), log(FT(5000)); length = 11))))
    n_val = 8
    value_bins = ntuple(_ -> InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_val - 1))), 8)
    n_dist = length(dist_vec) - 1

    sums_cpu = zeros(FT, 8, n_dist, n_val)
    counts_cpu = zeros(Int32, 8, n_dist, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_cpu, counts_cpu, x, u, dist_vec, value_bins;
        backend = SFC.SerialBackend(),
    )

    sums_gpu = zeros(FT, 8, n_dist, n_val)
    counts_gpu = zeros(Int32, 8, n_dist, n_val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist_vec, value_bins; kind = :single_pass_2d, n_val = n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, counts_gpu, x, u, dist_vec, value_bins;
        backend = SF.GPUBackend(KA.CPU()), workspace = ws,
    )

    Test.@test sums_gpu ≈ sums_cpu rtol = FT(1e-4)
    Test.@test counts_gpu == counts_cpu
end
