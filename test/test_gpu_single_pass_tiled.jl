using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using Random: Random

Random.seed!(42)

function _value_bins_uniform(n_val::Int, ::Type{FT}) where {FT}
    edges = collect(range(FT(-1), FT(2); length = n_val + 1))
    return [copy(edges) for _ in 1:6]
end

Test.@testset "GPU single-pass tiled parity — 1D log bins" begin
    N = 64
    FT = Float32
    x = rand(FT, 2, N) .* FT(5000)
    u = randn(FT, 2, N) .* FT(0.3)
    dist_vec = LogBinEdges(Vector{FT}(exp.(range(log(FT(10)), log(FT(5000)); length = 11))))

    sums_cpu = zeros(FT, 6, length(dist_vec) - 1)
    counts_cpu = zeros(Int32, 6, length(dist_vec) - 1)
    SFC.calculate_structure_functions_single_pass!(
        sums_cpu, counts_cpu, x, u, dist_vec; backend = CB.SerialBackend(),
    )

    sums_gpu = zeros(FT, 6, length(dist_vec) - 1)
    counts_gpu = zeros(Int32, 6, length(dist_vec) - 1)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist_vec; kind = :single_pass)
    SFC.calculate_structure_functions_single_pass!(
        sums_gpu, counts_gpu, x, u, dist_vec;
        backend = CB.GPUBackend(KA.CPU()), workspace = ws,
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
    value_bins = ntuple(_ -> LinearBinEdges(range(FT(-1), FT(2); length = n_val + 1)), 6)
    n_dist = length(dist_vec) - 1

    sums_cpu = zeros(FT, 6, n_dist, n_val)
    counts_cpu = zeros(Int32, 6, n_dist, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_cpu, counts_cpu, x, u, dist_vec, value_bins;
        backend = CB.SerialBackend(),
    )

    sums_gpu = zeros(FT, 6, n_dist, n_val)
    counts_gpu = zeros(Int32, 6, n_dist, n_val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist_vec, value_bins; kind = :single_pass_2d)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, counts_gpu, x, u, dist_vec, value_bins;
        backend = CB.GPUBackend(KA.CPU()), workspace = ws,
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
    value_bins = ntuple(_ -> InfPaddedBinEdges(LinearBinEdges(range(FT(-1), FT(2); length = n_val - 1))), 6)
    n_dist = length(dist_vec) - 1

    sums_cpu = zeros(FT, 6, n_dist, n_val)
    counts_cpu = zeros(Int32, 6, n_dist, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_cpu, counts_cpu, x, u, dist_vec, value_bins;
        backend = CB.SerialBackend(),
    )

    sums_gpu = zeros(FT, 6, n_dist, n_val)
    counts_gpu = zeros(Int32, 6, n_dist, n_val)
    ws = SFC.GPUSFWorkspace(KA.CPU(), dist_vec, value_bins; kind = :single_pass_2d, n_val = n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu, counts_gpu, x, u, dist_vec, value_bins;
        backend = CB.GPUBackend(KA.CPU()), workspace = ws,
    )

    Test.@test sums_gpu ≈ sums_cpu rtol = FT(1e-4)
    Test.@test counts_gpu == counts_cpu
end

Test.@testset "GPU single-pass global fallback parity — 2D and 3D" begin
    FT = Float32
    backend = CB.GPUBackend(KA.CPU())

    for D in (2, 3)
        N = 18
        x = rand(FT, D, N)
        u = rand(FT, D, N)
        bin_sets = (
            collect(FT, range(0, 2; length = 75)),
            LogBinEdges(collect(FT, exp.(range(log(FT(0.01)), log(FT(2)); length = 75)))),
            begin
                edges = sort!(vcat(FT(0), cumsum(rand(FT, 74))))
                edges ./= edges[end] / FT(2)
                edges
            end,
        )

        inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
        for bins in bin_sets
            sp_cpu = SFC.calculate_structure_functions_single_pass(
                x, u, bins; backend = CB.SerialBackend(),
                output_type = SF.StructureFunctionSumsAndCounts,
            )
            sp_gpu = SFC.calculate_structure_functions_single_pass(
                x, u, bins; backend, output_type = SF.StructureFunctionSumsAndCounts,
            )
            for k in inv
                Test.@test sp_gpu[k].counts == sp_cpu[k].counts
                Test.@test sp_gpu[k].sums ≈ sp_cpu[k].sums atol = FT(1e-4)
            end
        end
    end
end

Test.@testset "GPU single-pass 2D global fallback parity" begin
    FT = Float32
    backend = CB.GPUBackend(KA.CPU())
    N = 16
    x = rand(FT, 2, N)
    u = rand(FT, 2, N)
    value_bins = collect(FT, range(-0.5f0, 1.5f0; length = 9))
    bin_sets = (
        collect(FT, range(0, 2; length = 75)),
        LogBinEdges(collect(FT, exp.(range(log(FT(0.01)), log(FT(2)); length = 75)))),
        begin
            edges = sort!(vcat(FT(0), cumsum(rand(FT, 74))))
            edges ./= edges[end] / FT(2)
            edges
        end,
    )

    inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)
    for bins in bin_sets
        sp_cpu = SFC.calculate_structure_functions_single_pass_2d(
            x, u, bins, value_bins; backend = CB.SerialBackend(),
        )
        # `force_global_atomic` is a GPU routing override, so it goes to the GPU entry rather
        # than the backend-generic one, which would have to ignore it on a CPU backend.
        gs, gc = SFC.gpu_calculate_structure_functions_single_pass_2d(
            backend.backend, x, u, bins, value_bins; force_global_atomic = true,
        )
        for (t, k) in enumerate(inv)
            Test.@test gc[t, :, :] == sp_cpu[k].counts
            Test.@test gs[t, :, :] ≈ sp_cpu[k].sums atol = FT(1e-4)
        end
    end
end
