using StructureFunctions:
    StructureFunctions as SF, Calculations as SFC, StructureFunctionObjects as SFO,
    StructureFunctionTypes as SFT, InfPaddedBinEdges, LinearBinEdges, LogBinEdges
using OhMyThreads: OhMyThreads  # load extension for ThreadedBackend / AutoBackend when nthreads() > 1
using KernelAbstractions: KernelAbstractions as KA
using Test
using Random

"""Wide synthetic value-bin edges for unit tests only."""
function _synthetic_value_bins(n_bins::Int; pad_infinite::Bool = true)
    inner = LinearBinEdges(range(-1.0, 2.0, length = n_bins + 1))
    return pad_infinite ? InfPaddedBinEdges(inner) : inner
end

function _synthetic_value_bins_ntuple(n_bins::Int; pad_infinite::Bool = true)
    template = _synthetic_value_bins(n_bins; pad_infinite = pad_infinite)
    return ntuple(_ -> copy(template), 6)
end

Test.@testset "Single-Pass 2D Core Correctness & Parity" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5

    distance_bins = LogBinEdges(collect(exp.(range(log(1000.0), log(50000.0), length = 6))))
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1

    sums_2d = zeros(Float64, 6, n_bins, n_val)
    counts_2d = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins;
        backend = SFC.SerialBackend(),
    )

    Test.@test size(sums_2d) == (6, n_bins, n_val)
    Test.@test size(counts_2d) == (6, n_bins, n_val)

    sft_types = [
        SFT.SecondOrderStructureFunctionType(),
        SFT.LongitudinalSecondOrderStructureFunctionType(),
        SFT.TransverseSecondOrderStructureFunctionType(),
        SFT.ThirdOrderStructureFunctionType(),
        SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
    ]

    x_mat = x
    u_mat = u
    per_type_indices = 1:6

    for t in per_type_indices
        vb = value_bins isa Tuple ? value_bins[t] : value_bins
        sf2d = SFC.calculate_structure_function(
            sft_types[t],
            x_mat,
            u_mat,
            distance_bins,
            vb;
            backend = SFC.SerialBackend(),
            verbose = false,
            show_progress = false,
        )
        Test.@test sf2d isa SFO.StructureFunction2DSumsAndCounts
        Test.@test sums_2d[t, :, :] ≈ sf2d.sums
        Test.@test counts_2d[t, :, :] ≈ sf2d.counts
    end

    sums_1d, counts_1d = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend(),
    )

    for t in 1:6
        marg_sums = vec(dropdims(sum(sums_2d[t:t, :, :], dims = 3), dims = 1))
        marg_counts = vec(dropdims(sum(counts_2d[t:t, :, :], dims = 3), dims = 1))
        Test.@test marg_sums ≈ vec(sums_1d[t, :])
        Test.@test marg_counts == vec(counts_1d[t, :])
    end

    sums_post, counts_post = SFC.marginalize_sp2d_then_append_helmholtz_rows(
        sums_2d, counts_2d, distance_bins,
    )
    Test.@test sums_post ≈ sums_1d
    Test.@test counts_post == counts_1d

    fill!(sums_2d, 0.0)
    fill!(counts_2d, 0)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins;
        backend = SFC.AutoBackend(),
    )
    t_sums, t_counts = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins;
        backend = SFC.AutoBackend(),
    )
    Test.@test t_sums ≈ sums_2d
    Test.@test t_counts == counts_2d
end

Test.@testset "Single-Pass 2D with Custom Distance Metric (Cityblock)" begin
    using Distances: Distances as DI

    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5

    distance_bins = LogBinEdges(collect(exp.(range(log(1000.0), log(50000.0), length = 6))))
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1
    metric = DI.Cityblock()

    sums_2d = zeros(Float64, 6, n_bins, n_val)
    counts_2d = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_2d, counts_2d, x, u, distance_bins, value_bins;
        backend = SFC.SerialBackend(),
        distance_metric = metric
    )

    sft_types = [
        SFT.SecondOrderStructureFunctionType(),
        SFT.LongitudinalSecondOrderStructureFunctionType(),
        SFT.TransverseSecondOrderStructureFunctionType(),
        SFT.ThirdOrderStructureFunctionType(),
        SFT.DiagonalConsistentThirdOrderStructureFunctionType(),
        SFT.OffDiagonalInconsistentThirdOrderStructureFunctionType(),
    ]

    x_mat = x
    u_mat = u
    per_type_indices = 1:6

    for t in per_type_indices
        vb = value_bins isa Tuple ? value_bins[t] : value_bins
        sf2d = SFC.calculate_structure_function(
            sft_types[t],
            x_mat,
            u_mat,
            distance_bins,
            vb;
            backend = SFC.SerialBackend(),
            distance_metric = metric,
            verbose = false,
            show_progress = false,
        )
        Test.@test sums_2d[t, :, :] ≈ sf2d.sums
        Test.@test counts_2d[t, :, :] ≈ sf2d.counts
    end

    sums_1d, counts_1d = SFC.calculate_structure_functions_single_pass(
        x, u, distance_bins;
        backend = SFC.SerialBackend(),
        distance_metric = metric
    )

    for t in 1:6
        marg_sums = vec(dropdims(sum(sums_2d[t:t, :, :], dims = 3), dims = 1))
        marg_counts = vec(dropdims(sum(counts_2d[t:t, :, :], dims = 3), dims = 1))
        Test.@test marg_sums ≈ vec(sums_1d[t, :])
        Test.@test marg_counts == vec(counts_1d[t, :])
    end

    # Compare ThreadedBackend with SerialBackend
    t_sums, t_counts = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins;
        backend = SFC.AutoBackend(),
        distance_metric = metric
    )
    Test.@test t_sums ≈ sums_2d
    Test.@test t_counts == counts_2d
end

Test.@testset "Single-Pass 2D value-bin accepted shapes" begin
    Random.seed!(15)
    x = rand(2, 12)
    u = randn(2, 12)
    distance_bins = LinearBinEdges(range(0.0, 2.0; length = 5))

    shared_range_bins = range(-2.0, 3.0; length = 9)
    s_shared, c_shared = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, shared_range_bins; backend = SFC.SerialBackend(),
    )
    Test.@test size(s_shared) == (6, length(distance_bins) - 1, length(shared_range_bins) - 1)
    Test.@test size(c_shared) == size(s_shared)

    mixed_bins = ntuple(6) do t
        isodd(t) ? range(-2.0, 3.0; length = 9) :
        LinearBinEdges(range(-2.0, 3.0; length = 9))
    end
    s_mixed, c_mixed = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, mixed_bins; backend = SFC.SerialBackend(),
    )
    Test.@test size(s_mixed) == size(s_shared)
    Test.@test size(c_mixed) == size(c_shared)

    s_gpu, c_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, mixed_bins; backend = SF.GPUBackend(KA.CPU()),
    )
    Test.@test s_gpu ≈ s_mixed
    Test.@test c_gpu == c_mixed
end

Test.@testset "Single-Pass 2D value bins with 3D point fields" begin
    x = Float32[0.0 1.0 0.0 0.3;
                0.0 0.0 1.0 0.4;
                0.0 0.0 0.0 1.0]
    u = Float32[0.0 0.5 0.0 0.1;
                0.0 0.0 0.5 0.2;
                0.0 0.0 0.0 0.5]
    distance_bins = LinearBinEdges(Float32[0.1, 1.0, 2.0])
    value_bins = range(-2.0f0, 2.0f0; length = 9)

    sums_ref, counts_ref = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins; backend = SFC.SerialBackend(),
    )
    sums_gpu, counts_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins; backend = SF.GPUBackend(KA.CPU()),
    )
    Test.@test isapprox(sums_gpu, sums_ref; atol = 1f-4)
    Test.@test counts_gpu == counts_ref
end

Test.@testset "Single-Pass 2D GPU (KA.CPU) parity vs Serial" begin
    Random.seed!(42)
    n_points = 40
    x = rand(n_points, 2)' .* 50000.0
    u = randn(2, n_points) .* 0.5
    distance_bins = LogBinEdges(collect(exp.(range(log(1000.0), log(50000.0), length = 6))))
    value_bins = _synthetic_value_bins(10; pad_infinite = true)
    n_val = length(value_bins) - 1
    n_bins = length(distance_bins) - 1

    sums_ref = zeros(Float64, 6, n_bins, n_val)
    counts_ref = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_ref, counts_ref, x, u, distance_bins, value_bins;
        backend = SFC.SerialBackend(),
    )

    sums_gpu, counts_gpu = SFC.calculate_structure_functions_single_pass_2d(
        x, u, distance_bins, value_bins;
        backend = SF.GPUBackend(KA.CPU()),
    )
    Test.@test sums_gpu ≈ sums_ref
    Test.@test counts_gpu == counts_ref

    sums_gpu2 = zeros(Float64, 6, n_bins, n_val)
    counts_gpu2 = zeros(UInt32, 6, n_bins, n_val)
    SFC.calculate_structure_functions_single_pass_2d!(
        sums_gpu2, counts_gpu2, x, u, distance_bins, value_bins;
        backend = SF.GPUBackend(KA.CPU()),
    )
    Test.@test sums_gpu2 ≈ sums_ref
    Test.@test counts_gpu2 == counts_ref
end
