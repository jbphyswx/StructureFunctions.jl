using ComputationalBackends: ComputationalBackends as CB
using Test: Test
using KernelAbstractions: KernelAbstractions as KA
using StructureFunctions: StructureFunctions as SF, Calculations as SFC
using StructureFunctions: InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using Random: Random

Random.seed!(42)

const GPU_SP_TILED_EXT = Base.get_extension(SF, :StructureFunctionsKernelAbstractionsExt)

# Single-pass log-bin histograms through a workspace into Int32 counts match serial.
Test.@testset "GPU single-pass tiled parity — 1D log bins" begin
    N = 64
    FT = Float32
    x = rand(FT, 2, N) .* FT(5000)
    u = randn(FT, 2, N) .* FT(0.3)
    dist_vec = LogBinEdges(FT(10), FT(5000), 11)

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

# Inf-padded value bins put out-of-range values in their catch-all bins as serial does.
Test.@testset "GPU single-pass tiled parity — 2D InfPadded linear value catch-alls" begin
    N = 48
    FT = Float32
    x = rand(FT, 2, N) .* FT(5000)
    u = randn(FT, 2, N) .* FT(0.3)
    dist_vec = LogBinEdges(FT(10), FT(5000), 11)
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

const GPU_SP1D_CAP_CASES = ((2, :uniform, :within_cap), (3, :log, :within_cap), (2, :irregular, :past_cap))

# Point-list single-pass histograms match serial with bin counts within and past the tiled kernel's cap.
Test.@testset "GPU single-pass parity within and past the tiled bin cap — 2D and 3D" begin
    FT = Float32
    backend = CB.GPUBackend(KA.CPU())
    inv = (:S2, :L2, :T2, :S3, :L3, :L1T2)

    Test.@testset "D = $D, $kind bins, $side" for (D, kind, side) in GPU_SP1D_CAP_CASES
        N = 18
        n_bins = side === :within_cap ? 74 : GPU_SP_TILED_EXT.SF_GPU_MAX_BINS + 1
        x = rand(FT, D, N)
        u = rand(FT, D, N)
        bins = if kind === :uniform
            collect(FT, range(0, 2; length = n_bins + 1))
        elseif kind === :log
            LogBinEdges(FT(0.01), FT(2), n_bins + 1)
        else
            edges = sort!(vcat(FT(0), cumsum(rand(FT, n_bins))))
            edges ./= edges[end] / FT(2)
            edges
        end
        sp_cpu = SFC.calculate_structure_functions_single_pass(
            x, u, bins, SF.StructureFunctionSumsAndCounts; backend = CB.SerialBackend(),
        )
        sp_gpu = SFC.calculate_structure_functions_single_pass(
            x, u, bins, SF.StructureFunctionSumsAndCounts; backend,
        )
        Test.@test all(k -> sp_gpu[k].counts == sp_cpu[k].counts, inv)
        Test.@test all(k -> isapprox(sp_gpu[k].sums, sp_cpu[k].sums; atol = FT(1e-4)), inv)
    end
end
